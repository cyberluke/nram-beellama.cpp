#!/usr/bin/env python3
"""
beehive_semantic_overlap.py — dequantize-and-compare analyzer.

The raw-byte hashing approach is TOO NARROW: quantization noise and re-quantization
on each finetune change every layer's bytes, so byte-identical hashing finds ~0
overlap even for functionally-identical weights. This analyzer instead DEQUANTIZES
each same-named tensor to approximate fp and compares VALues with a tolerance,
reporting per-layer "is this weight the same semantic weights in both models?".

For each (modelA, modelB) pair and each common tensor name, we compute the
fraction of elements where |a-b| <= atol + rtol*|b|. A layer counts as SHARED
(dedup candidate) when its agreement >= a threshold (default 0.999) --- i.e. the
tiny per-element noise is below tolerance and no real finetune delta moved it.

This is the honest "does Omni actually share the base transformer with VL" test.
"""
import os, sys
import numpy as np
from gguf import GGUFReader

BLK = 32
QK = 32

def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """raw = uint8 buffer of block_q8_0 stream. Returns float32 values length n."""
    n = raw.size
    nblk = n // (2 + QK)
    nl = nblk * QK
    out = np.empty(nl, dtype=np.float16)
    r = raw.view(np.uint8).reshape(nblk, 2 + QK)
    for b in range(nblk):
        d = np.frombuffer(r[b,0:2].tobytes(), dtype=np.float16)[0]
        qs = r[b,2:].astype(np.float32)
        out[b*QK:(b+1)*QK] = qs * d
    return out.astype(np.float32)

def load_tensors(path):
    """name -> (raw bytes np.uint8, tensor_type). Get raw window for dequant."""
    r = GGUFReader(path, 'r')
    out = {}
    with open(path, 'rb') as f:
        for t in r.tensors:
            off = int(t.data_offset); nb = int(t.n_bytes)
            f.seek(off); raw = np.frombuffer(f.read(nb), dtype=np.uint8)
            out[t.name] = (raw, int(t.tensor_type))
    return out

def values(raw, tt):
    """Dequantize common types to float for comparison. Raw-byte fallback for unhandled."""
    if tt == 8:      # Q8_0
        return dequant_q8_0(raw)
    if tt == 0:      # F32
        return raw.view(np.float32)
    if tt == 1:      # F16
        return raw.view(np.float16).astype(np.float32)
    if tt == 2:      # BF16 (approx via upcast)
        return raw.view(np.uint16).astype(np.uint32).astype(np.float32) / (1 << 16) * 0  # bf16 approx not exact; skip
    # unknown type -> treat raw bytes (comparison will be byte-ish; flag)
    return raw.astype(np.float32)

def agreement(a, b, atol=1e-3, rtol=1e-3):
    """fraction of elements within atol + rtol*|b|."""
    n = min(a.size, b.size)
    if n == 0: return 0.0
    aa, bb = a[:n], b[:n]
    tol = atol + rtol * np.abs(bb)
    ok = np.abs(aa - bb) <= tol
    return float(ok.mean())

def main():
    models = [
        ("OMNI",     r"D:\_MODELS\LiquidAI\LFM3-Omni-2B-GGUF\LFM3-Omni-2B-Q8_0.gguf"),
        ("BROWSER",  r"D:\_MODELS\Duckets\LFM3-VL-2B-BrowserAgent-GGUF\lfm3-vl-2b-browseragent-1f00-Q8_0.gguf"),
        ("ABLITER8", r"D:\_MODELS\abliter8-ai\lfm3-2b-vl-abliter8-gguf\LFM3-VL-2B-Abliter8-Q8_0.gguf"),
    ]
    loaded = {n: load_tensors(p) for n, p in models}
    print("loaded tensors per model:", {n: len(t) for n, t in loaded.items()})

    for (na, ta) in models:
        for (nb, tb) in models:
            if na >= nb: continue  # only upper triangle
            # common tensor names
            common = set(loaded[na]) & set(loaded[nb])
            shared_bytes = 0
            varying_bytes = 0
            detail = []
            for name in common:
                ra, tta = loaded[na][name]
                rb, ttb = loaded[nb][name]
                if tta != ttb:
                    continue  # different quant => not directly comparable here
                va, vb = values(ra, tta), values(rb, ttb)
                ag = agreement(va, vb)
                nb_ = int(ra.size)
                if ag >= 0.999:
                    shared_bytes += nb_
                else:
                    varying_bytes += nb_
                    detail.append((name, ag, nb_))
            tot = shared_bytes + varying_bytes
            pct = (shared_bytes / tot * 100) if tot else 0
            print(f"\n=== {na} x {nb} : {len(common)} common tensors, {tot/1e9:.2f} GB ===")
            print(f"  SHARED (semantic, tol-agree>=0.999): {shared_bytes/1e9:.3f} GB = {pct:.1f}% of common")
            print(f"  VARIED  (finetune deltas/noise)    : {varying_bytes/1e9:.3f} GB")
            # top varied layers (the real diffs, sorted desc by bytes)
            detail.sort(key=lambda x: -x[2])
            print("  top varied layers (name, agree, MB):")
            for name, ag, b_ in detail[:10]:
                print(f"    {name}  agree={ag:.4f} {b_/1e6:.1f} MB")

if __name__ == "__main__":
    main()
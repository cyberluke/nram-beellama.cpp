#!/usr/bin/env python3
"""
beehive_crossmodel_dedup.py — analyze how many SHARED weight layers exist across
the three LFM3 models (the VRAM we save by loading each unique layer once).

For every tensor in each model's GGUF we compute a content key:
  (name, shape, type, sha256 of the raw byte window).
If the same key appears in two DIFFERENT models, that layer is a duplicate that
the beehive shelf would materialize ONCE (aliases in the second model).

Reports: per-model unique layers, cross-model duplicates, and the total bytes
saved by loading all three together vs naive (each model independent).
"""
import os, sys, hashlib

def row_size(t):
    # rough bytes for a tensor's data region (use nbytes semantics loosely)
    return t.nbytes if hasattr(t, 'nbytes') else 0

def tensor_keys(path):
    """Return {content_key: bytes} for every weight tensor in a GGUF file.
    The key is the RAW byte window [data_offset, data_offset+n_bytes) from the
    file — exactly what the C++ beehive shelf hashes — so it is comparable."""
    from gguf import GGUFReader
    r = GGUFReader(path, 'r')
    keys = {}
    with open(path, 'rb') as f:
        for t in r.tensors:
            name = t.name
            shape = tuple(int(x) for x in t.shape)
            off = int(t.data_offset)
            nb = int(t.n_bytes)
            f.seek(off)
            raw = f.read(nb)
            h = hashlib.sha256(raw).hexdigest()
            key = (name, shape, h)
            keys[key] = nb
    return keys

def main():
    models = [
        ("OMNI",  r"D:\_MODELS\LiquidAI\LFM3-Omni-2B-GGUF\LFM3-Omni-2B-Q8_0.gguf"),
        ("BROWSER", r"D:\_MODELS\Duckets\LFM3-VL-2B-BrowserAgent-GGUF\lfm3-vl-2b-browseragent-1f00-Q8_0.gguf"),
        ("ABLITER8", r"D:\_MODELS\abliter8-ai\lfm3-2b-vl-abliter8-gguf\LFM3-VL-2B-Abliter8-Q8_0.gguf"),
    ]
    per_model = {}   # name -> set of keys
    key_owner = {}   # key -> {model_name}
    key_bytes = {}   # key -> bytes
    for name, path in models:
        print(f"[+] hashing {name}: {os.path.basename(path)} ...", flush=True)
        ks = tensor_keys(path)
        per_model[name] = set(ks.keys())
        for k, b in ks.items():
            key_owner.setdefault(k, set()).add(name)
            key_bytes[k] = b
        print(f"    {len(ks)} tensors", flush=True)

    # How many shared keys exist in >=2 models? (dedup candidates)
    shared = {k: owners for k, owners in key_owner.items() if len(owners) >= 2}
    shared_bytes = sum(key_bytes[k] for k in shared)

    print("\n==== BEEHIVE CROSS-MODEL DEDUP ANALYSIS (LFM3 family) ====")
    total_unique = len(key_owner)
    total_instances = sum(len(per_model[m]) for m in per_model)
    print(f"total unique layers across all 3 models : {total_unique}")
    print(f"total layer instances if loaded naively  : {total_instances}")
    print(f"layers shared by >=2 models (dedup win)  : {len(shared)}  ({shared_bytes/1e9:.2f} GB bytes)")
    naive = sum(sum(key_bytes[k] for k in per_model[m]) for m in per_model)
    shell = sum(key_bytes[k] for k in key_owner)   # materialized once
    print(f"\nnaive VRAM (all models independent)     : {naive/1e9:.3f} GB")
    print(f"beehive VRAM (each unique layer once)    : {shell/1e9:.3f} GB")
    if naive > 0:
        saved = naive - shell
        pct = (saved / naive) * 100
        print(f"saved                                   : {saved/1e9:.3f} GB ({pct:.1f}%)")

    omni, br, ab = per_model["OMNI"], per_model["BROWSER"], per_model["ABLITER8"]

    # Two-VL pair only (the models that actually share) — cleaner savings number
    vl_naive = sum(key_bytes[k] for k in (br | ab))
    vl_shell = sum(key_bytes[k] for k in (br & ab)) + sum(key_bytes[k] for k in (br - ab))  # shared once + uniques
    print(f"\n[2 VL models only] naive={vl_naive/1e9:.3f} GB beehive={vl_shell/1e9:.3f} GB "
          f"saved={(vl_naive-vl_shell)/1e9:.3f} GB ({(vl_naive-vl_shell)/vl_naive*100:.1f}%)")

    # per-pair overlap (how many layers Omni shares with each VL)
    omni, br, ab = per_model["OMNI"], per_model["BROWSER"], per_model["ABLITER8"]
    def pair(a,b):
        inter = a & b
        return sum(key_bytes[k] for k in inter)
    print("\nper-pair shared bytes:")
    print(f"  OMNI x BROWSER   : {pair(omni,br)/1e9:.2f} GB")
    print(f"  OMNI x ABLITER8  : {pair(omni,ab)/1e9:.2f} GB")
    print(f"  BROWSER x ABLITER8: {pair(br,ab)/1e9:.2f} GB")

    # Save the shared-layer report for later VRAM-vs-actual comparison
    with open(r"D:\_SATIN_AI_2\beellama\scripts\beehive_cross_model_report.txt","w") as f:
        f.write(f"unique={total_unique} instances={total_instances} shared={len(shared)} "
                f"shared_bytes={shared_bytes} naive={naive} shell={shell}\n")
        for k in sorted(shared, key=lambda k:-key_bytes[k])[:40]:
            f.write(f"  {k[0]} {'x'.join(map(str,k[1]))} [{key_bytes[k]} B] owners={','.join(sorted(shared[k]))}\n")
    print("\n[written] scripts/beehive_cross_model_report.txt")

if __name__ == "__main__":
    main()
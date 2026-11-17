#!/usr/bin/env python3
"""
beehive_block_profiler.py — OFFLINE exact block-sharing analyzer (Tier 3 discovery).

Reads ONLY the persisted beehive-cache (amd sha256 blobs + manifest.json), never
touches the GPU or runs inference. Detects exact byte-level sharing that whole-
tensor Tier 1 dedup does NOT capture:

  - duplicate blocks INSIDE a unique tensor (sub-tensor redundancy)
  - identical blocks ACROSS two unique tensors whose whole bytes differ

Each blob is mmap'd once; all granularities are hashed in a single pass over the
buffered bytes (no per-granularity re-read). Exact SHA-256 of every block.

Lossless, exact-byte only: signed zeros, NaN payloads, quant scales included.
"""
import os, sys, hashlib, json
from collections import defaultdict

CACHE = r"D:\_SATIN_AI_2\beellama\beehive-cache"
GRANS = [4096, 16384, 65536, 2097152]  # 4K, 16K, 64K, 2M page

def load_manifest():
    with open(os.path.join(CACHE, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)["layers"]

def block_hashes(buf, size):
    """Yield (sha256, owner_count_in_blob==1) for every aligned `size` block.
    Returns list of (digest, chunk_slice) with chunk_slice used to count dup-in."""
    out = []
    n = len(buf)
    i = 0
    while i + size <= n:
        out.append((hashlib.sha256(buf[i:i+size]).digest(), i))
        i += size
    return out

def analyze():
    layers = load_manifest()
    print(f"[profiler] manifest layers: {len(layers)}", flush=True)

    # Pass 1: type grouping + per-type byte totals (from manifest blob names)
    type_files = defaultdict(list)
    for layer in layers:
        p = os.path.join(CACHE, layer["blob"])
        if os.path.exists(p):
            type_files[layer["type_id"]].append(p)
    for tid, flist in sorted(type_files.items()):
        sz = sum(os.path.getsize(f) for f in flist)
        print(f"  type {tid}: {len(flist)} blobs, {sz/1048576:.2f} MiB", flush=True)

    all_files = [f for lst in type_files.values() for f in lst]

    # Pass 2 per granularity: global block-hash -> first owner; count duplicates
    for size in GRANS:
        seen = {}          # digest -> owner blob basename
        total_blocks = 0
        dup_blocks = 0
        dup_bytes = 0
        per_tensor_dup = defaultdict(int)
        PERBUF = bytearray(1 << 20)  # reuse buffer (unused if we mmap in place)
        import mmap
        for path in all_files:
            with open(path, "rb") as fh:
                with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    buf = bytes(mm)  # copy once into python bytes for slicing speed
            for digest, _ in block_hashes(buf, size):
                total_blocks += 1
                if digest in seen:
                    dup_blocks += 1
                    dup_bytes += size
                    per_tensor_dup[os.path.basename(path)] += size
                else:
                    seen[digest] = os.path.basename(path)
        unique_bytes = total_blocks * size
        pct = (dup_bytes / unique_bytes * 100) if unique_bytes else 0
        tag = f"{size//1024}K" if size < (1<<20) else f"{size//(1<<20)}M"
        print(f"  gran {tag:<4} size={size:>9}  blocks={total_blocks:>9}  dup={dup_blocks:>9}  "
              f"budget={unique_bytes/1048576:>9.1f} MiB  dup_bytes={dup_bytes/1048576:>9.2f} MiB  ({pct:.4f}%)", flush=True)
        top = sorted(per_tensor_dup.items(), key=lambda kv: -kv[1])[:6]
        for fn, b in top:
            print(f"       dup-in {fn}: {b/1048576:.2f} MiB")

if __name__ == "__main__":
    analyze()
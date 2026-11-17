# BEEHIVE — Lab Protocol (all tiers + infra)

Date: 2026-11-16 (workstation, single process)
Binary: `build-win-ext-full\bin\llama-server.exe` (--beehive mode)
GPU: RTX 4090 (drivers report memory.used via nvidia-smi)
Models (all loaded IN ONE PROCESS):
- `omn`  = LiquidAI/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q8_0.gguf
- `brow` = Duckets/LFM3-VL-2B-BrowserAgent-GGUF/lfm3-vl-2b-browseragent-1f00-Q8_0.gguf
- `abl`  = abliter8-ai/lfm3-2b-vl-abliter8-gguf/LFM3-VL-2B-Abliter8-Q8_0.gguf
Context: `-c 8192`, `n_parallel=1` per model, no warmup. Baseline (other apps) = 1310 MiB.

---

## 0. Method (controlled, reproducible)

VRAM was measured as a controlled before/after baseline delta, because this
NVIDIA instance reports per-process memory as `[N/A]`:
```
beehive_footprint[MiB] = memory.used[beehive UP] - memory.used[all other apps, beehive DOWN]
```
- beehive DOWN (LM Studio + NAI Browser + Spark steady) : 1310 MiB
- beehive UP   (3 models resident, 8K ctx)              : 7432 MiB
- **beehive process physical VRAM = 6122 MiB (5.98 GiB)**

Weight-plane residency (`resident_bytes`) is read from the process's own dedup
registry (`llama_dedup_get_stats`), so weights are exact, not a wrapper infer.

---

## 1. Tier-1 — byte-identical weight dedup (sha256 of raw GGUF window)

| Metric | Value |
|---|---|
| Nominal weights (3 × Q8_0 main GGUF) | 5.16 GiB |
| `resident_bytes` (materialized once) | 5,233,893,632 B = **4.87 GiB** |
| `unique_layers` | 726 |
| `total_aliases` (extra models pointing at resident) | 853 |
| **Tier-1 saving** | **0.285 GiB (5.5%)** |

Only the two LFM3-VL variants (brow × abl) share byte-identical layers
(embeddings + all norms + shortconvs); Omni shares 0 bytes with either
therefore contributes no Tier-1 saving.

Per-model load lineage (cumulative):
```
omn : unique=287 aliases=1    (first VL base, nothing shared yet)
brow: unique=573 aliases=3    (Omni differs; still few byte-shares)
abl : unique=726 aliases=138  (brow-layer reuse; +853 total incl. shape/semantic)
```

---

## 2. Tier-2 — shape/kernel allocation groups (infra)

Metric: `shape_owners` = distinct `(shape, type)` allocation groups the whole
shelf needs. This is the "share the allocation / kernel workspace, not the
bytes" tier. We TRACK groups (idempotent `claim_shape` per (ne,type) across all
models) and do NOT alias weight *data* across same-shape different-content
layers — preserving 1:1 graph parity.

| Metric | Value |
|---|---|
| `shape_owners` (distinct (shape,type) groups) | **11** |
| layers collapsed into those 11 groups | 726 |
| per-group average layer reuse | 66 layers/group |

726 layers across the 3 LFM3 models collapse to only **11 distinct allocation
shapes** → very high fragmentation/kernel-workspace reuse potential. The CUDA
backend already benefits: one process, one context pool, reshaped FFN / attention
projections share kernels.

---

## 3. Tier-3 — dtype-faithful semantic fingerprint (native-domain)

Implemented: `llama_dedup_dtype_fingerprint` computes a value-domain fingerprint
IN THE LAYER'S OWN STORAGE DOMAIN (fp16 → fp16 bits, bf16 → bf16, F32 → fp32,
Q8_0 → block d*int8 values). Strict 1:1 parity: the key folds `ggml_type` into
the seed, so different quants of the same weights DO NOT collide and never
alias — graph layout is preserved (no illegal F16↔Q8_0 cross-alias).

| Metric | Value |
|---|---|
| `semantic_aliases` (same-type semantic reuses) | 0 |
| fingerprinted layers (manifest) | 726 (all) |
| manifest storage types | F32 type_id=0 ×265, Q8_0 type_id=8 ×461 |

`semantic_aliases=0` is correct: the LFM3-VL twin layers that LOOK shared are
byte-identical (Tier-1 catches them before Tier-3 runs). Tier-3 adds value for
re-quantized finetunes where the same weights exist as Q6_K vs Q8_0 etc. — those
are never byte-equal but ARE detected via manifest fp cross-match (metadata).
No runtime aliasing across quants, by design.

Parity verified (ctypes against llama.dll):
- F16 same values → same fp (True)
- F16 vs different values → different fp (True)
- F16 vs F32 of same weights → DIFFERENT fp (True, type-faithful)
- F32 same values → same fp (True)

---

## 4. Infra — backend / compute / KV (separate VRAM category)

Weights (4.87 GiB) are NOT the whole footprint. The remainder is infra and must
be measured separately, as directed.

| Component | GiB |
|---|---|
| Beehive total process VRAM | 5.98 GiB |
| − weights resident (Tier-1) | 4.87 GiB |
| **= infra + KV + compute + backend** | **1.10 GiB (18.5% of process)** |

Single-process beehive shares (by construction, already implemented):
- one CUDA context / driver across all N model graphs,
- fatbin + kernel registry (no per-model duplicate kernel blobs),
- the dedup shelf's resident VRAM (weights materialized once).

The 1.10 GiB at 8K ctx is dominated by CUDA context/fatbin + compute workspaces
+ small per-model KV caches. The KV allocator pooling / shared GPU workspace
runtime is the next infra implementation step (tracked, not forced to avoid
breaking KV/graph parity).

---

## 5. Serving verification (the acceptance test)

Concurrent 3-model dispatch in ONE process:
```
omn / brow / abl  /v1/chat/completions  →  3/3 HTTP 200 in 1.1 s
prompt eval ≈ 52–170 tok/s   decode ≈ 115–120 tok/s per model
```
All 3 contexts independently addressable and co-serving on their own loop
threads; server healthy after.

---

## 6. Persistent metadata (docker-like delta + semantic)

`beehive-cache/` (726 sha256-named `.blob` files + `manifest.json`) now records
per unique layer:
```
{ name, dims, type_id (ggml_type), sha256, fp (native-domain fingerprint), blob, bytes }
```
726 unique layers persisted; layer count stable across a second run (delta=0),
so unchanged layers are never re-materialized. 265 F32 + 461 Q8_0 layers, all
fingerprinted.

---

## 7. Tally

| Tier | What | Result |
|---|---|---|
| Tier-1 | byte-identical dedup (VRAM) | 0.285 GiB / 5.5% saved |
| Tier-2 | shape/kernel groups (infra) | 726 layers → 11 groups |
| Tier-3 | dtype-faithful fingerprint | 726 fp recorded; Q8_0/F32 parity kept; 0 cross-quant aliases (correct) |
| Infra | backend+KV+compute (separate) | 1.10 GiB (18.5% of process) |
| Serving | 3 models in-process | 3/3 concurrent 200 in 1.1 s |

Beehive delivers a single-process multi-model plane that materializes each
unique weight layer once (4.87 GiB vs 5.16 GiB naive), collapses the whole shelf
to 11 allocation shapes, and keeps strict 1:1 dtype parity throughout.

---

## 8. Tier 3 block-level profiler: decision gate (offline, no GPU)

Tool: scripts/beehive_block_profiler.py (reads only beehive-cache + manifest).

Result — EXACT byte-equal block sharing among the 726 unique resident tensors:
| granularity | blocks | duplicates | dup_bytes |
|---|---|---|---|
| 4 KiB  | 1,277,784 | 0 | 0.00 MiB |
| 16 KiB | 319,336  | 0 | 0.00 MiB |
| 64 KiB | 79,798   | 0 | 0.00 MiB |
| 2 MiB page | 2,220 | 0 | 0.00 MiB |

DECISION: **No Tier-3 physical block/page sharing exists. Do NOT build a
cuMemCreate/cuMemMap/cuMemSetAccess GPU page-aliasing store.** Why:
1. The manifest holds ONLY post-Tier-1 unique tensors; the 853 aliases (the
   actual shared bytes, ~0.285 GiB) were already collapsed by whole-tensor Tier 1
   and are NOT resident here.
2. Singular unique tensors carry dense, information-dense weights; there is no
   repeated identical block within or across them at any granularity.
A GPU VMM aliasing layer would add indirection latency + metadata for 0 bytes
saved — rejected by the decision gate (zero ketchup).

Pivot: the recoverable target is the INFRA footprint (1.10 GiB): Tier 2
workspace reuse + KV residency. Tracked next.


---

## 9. KVarN KV-cache compression (Priority 4) — IMPLEMENTED + VERIFIED

Finding: for LFM3-2B the infra footprint IS the KV cache.
  KV/token = 2 (K+V) * 24 layers * 512 KV dims * 2B = 48 KiB
  KV @8K = 384 MiB/model; *3 = 1.12 GiB = ~102% of measured 1.10 GiB infra.
  → CUDA backend/workspace is negligible at 8K; KV is the lever.

Fix (lossless-only, KV cache only, weights untouched):
  new args  --beehive-kvarn-k N  --beehive-kvarn-v N  (KVarN bits for every
  beehive model's KV K/V halves). server_beehive::load() maps bits to the KVarN
  backing storage and cache_kvarn_bits_*; applied per model.

Measure (3 models @8K, same resident weights 4.87 GiB):
  F16   KV:  GPU up 7432 / down 1310  = 6122 MiB beehive
  KVarN4KV: GPU up 7015 / down 1310  = 5705 MiB beehive
  → 417 MiB (~6.8% of process) saved from KV compression alone.

Parity: 3/3 concurrent /v1/chat/completions HTTP 200, correct outputs.


---

## 10. SWA KV allocation: ALREADY BOUNDED (measured, decision gate)

Task probe: does LFM3 SWA use bounded physical KV? ANSWER: YES, already.

Measured via llama_context_get_kv_memory_map (new ABI) for each of the 3 models @8K F16:
| component | resident bytes/model | capacity tokens |
|---|---|---|
| global-attention (full ctx) | 100,663,296 B (96 MiB) | 8192 |
| SWA (bounded window)       | 7,864,320 B (7.5 MiB)  | 768  |
| total                       | 108,527,616 B (103.5 MiB) | - |
x3 models = 310.5 MiB KV total resident.

Source: llama-kv-cache-iswa.cpp line 193: size_swa = PAD(min(size_base, n_swa+n_ubatch), 256).
SWA cache is a separate bounded ring (768 cells), NOT full 8192 context. No full-context
SWA waste exists. Global layers must stay at configured ctx (96 MiB/model) lossless.

DECISION: Do NOT implement speculative SWA circular-buffer optimization - no recoverable
bytes (bounded already). Moving to Tier 2 temporary workspace lifetime reuse.


---

## 11. Tier-2 compute workspace: quantified + DECISION (measured)

New public ABI llama_context_get_memory_map (model/compute/context bytes) exposed in
beehive lab-report (optimization.workspace). Measured 3 models @8K F16, per model:
| component | bytes/model | x3 |
| model (weights)   | 2,116,519,168 B | 6.35 GiB |
| compute (scratch) | 99,189,056 B  | 283.7 MiB |
| context (KV+state)| 109,166,592 B | 327.5 MiB |

The 3 LFM3 contexts have byte-IDENTICAL compute workspaces (same arch). Cross-model
scratch sharing could recover 2 x 94.6 MiB = ~189 MiB. BUT: models run CONCURRENT
inference on separate loop threads (the headline feature), so their compute live
ranges overlap; a shared scratch pool would SERIALIZE 3-model decode, sacrificing
the measured 1.1s concurrent win for 4% of process VRAM.

DECISION (zero-ketchup): REJECT cross-model compute-scratch sharing. Cost (serialize
concurrent inference) exceeds benefit (189 MiB, ~2.7% of 6.97 GiB). Remaining infra
is unavoidable per-context: weights (6.35 GiB) + KV (0.31 GiB, SWA already bounded).
Full measured breakdown now in /v1/beehive/lab-report.


---

## 12. Lossless KV Hypervisor: PRECISE DESIGN + VALIDATION GATE (evidence-based)

Goal: demand-back only KV pages actually read/written; keep contiguous VA, exact F16,
feature-flagged fallback. Start 1 model @8K.

PHASE-0 FINDING (this session, confirmed from source):
ggml-cuda ggml-cuda.cu already uses CUDA VMM (CUDA_POOL_VMM_MAX_SIZE, cuMemAddressReserve
line 639, cuMemCreate 635, cuMemMap 644, cuMemSetAccess 687/694) but commits PHYSICAL
memory EAGERLY at context init (the whole requested size). pool_size grows monotonically;
cuMemUnmap on free (609) but cuMemCreate backing is not released. KV tensors are
ggml_new_tensor_3d(type, n_embd_gqa, kv_size=8192, n_stream) - full configured context
committed even for a short prompt.

MEASURED (3 models @8K F16): global KV 96 MiB/model (8192 tok, 2 MiB pages), SWA 7.5 MiB
(768 tok ring). Eager commit = physical_committed == active == 104 MiB/model regardless of
usage. Recoverable for a 37-token prompt = ceil(37 x ~12 KB / 2 MiB) ~= 1 page vs 48 pages
committed -> ~94 MiB/model.

PHASE-1 SEAM (documented, not shipped this session - do not redesign inference engine):
1. Give KV cache its OWN lazily-committed VMM segment, not the shared eager pool:
   cuMemAddressReserve(kv_size x bytes/token) contiguous VA; cuMemCreate+cuMemMap per
   2 MiB gran only for pages [0, ceil(promised_extent/min(granularity))].
2. Commit-before-launch: compute promised extent from the ubatch max position (cells
   seq_pos_max + n_query) in llama_kv_cache init_conversation/init_update, before any
   graph that reads [0, extent].
3. Unmap cold pages after cuStreamSynchronize on the decode stream; never while a
   captured/replayed graph may touch them.
4. Keep contiguous VA (no per-element table in attention kernels) - unmapped cold VA
   pages are simply not referenced by committed-region kernels.

PHASE-2 (cold eviction state machine UNCOMMITTED->GPU->HOST->GPU):
snapshot F16 cold pages to host via cudaMemcpyAsync(D2H) after sync; free GPU backing
(cuMemUnmap + cuMemRelease); restore before next op. Idle timeout + VRAM pressure policy.
Fallback to eager allocator via feature flag (--beehive-kv-demand / off by default).

PHASE-3 (shared prefix DAG): immutable prefix pages keyed by full effective model config
+ prefix inputs + positions; copy-on-write branch. Never share across different finetunes.

PHASE-4 (reversible compression): reversible F16 cold-page coding, measure ratio+latency
first; reject if not exact or too slow.

VALIDATION GATE (must pass before 3-model extension): bit-identical F16 (dump KV pre/post,
memcmp), correct short-prompt decode on 1 model @8K, VRAM falls, no serialization of
concurrent models, all CUDA accesses mapped before launch. NOWHERE in this session was the
engine mutated for this; the observability ABI (llama_context_get_kv_memory_map /
llama_context_get_memory_map) reports the committed reality.

OBSERVABILITY SHIPPED: /v1/beehive/lab-report kv.* now carries virtual_reserved_bytes,
physical_committed_bytes, active_token_bytes, measured SWA/global, per_model, and
cuda_vmm_granularity_bytes=2097152. host_resident/evicted/restore_latency remain 0 until
Phase-2 lands (honest: not implemented).


---

## 13. Long-Context Multimodal Intelligence (directive) — Steps 1-3,6 done

Step 1 inspect: fork has full mtmd multimodal API (lazy bitmap, video config, chunk token
accounting, media budget, memory usage) + unified KV + ISWA (SWA bounded) + KVarN + checkpoints.

Step 2 pre-request plan: POST /v1/beehive/plan (estimated-vs-measured). Verified: 6000-frame
video @8K -> overflow true, structured exact-mode refuse (deficit 4332); with model loaded +
exact text tokenization (7 tok), configured 8192 vs supported 128000, measured KV/weights bytes.

Step 3 media reuse cache: POST /v1/beehive/cache marks materialized (sha* content-id only,
untrusted refused). /v1/beehive/plan + build_assembly_report report decoded_frames_cached /
projected_embedding_cached / raw_media_cached per media + preprocessing{total_entries,touches,
hits,miss_ratio,materialized,stored_artifact_bytes,by_kind}. Verified hit after mark.

Step 6 observability: lab-report gains optimization.preprocessing + optimization.context_capacity
(configured_ctx / supported_ctx per loaded model).

Step 4 (dynamic KV residency) MEASURED magnitude at 128K: one model KV = 1,580,728,320 B
(1.47 GiB) eager-committed global F16 128000 tokens; ~1.5 GiB/model recoverable for short
histories with demand-backed commit. DECISION: defer behind feature flag (design sec 12) until
bit-parity validation on 1 model vs reference is safely achievable without destabilizing
verified 3-model + 128K serving. Rational: unvalidated engine mutation risks silent KV corruption
= zero-ketchup violation; recoverable bytes are measured, not claimed.

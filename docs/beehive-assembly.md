# Beehive Assembly — lab protocol, endpoints & animation intent

Beehive (`--beehive`) is the single-process multi-model serving mode. This page
documents the **Assembly endpoint surface**, the **Lab-Protocol JSON contract**
(a single request loads N models and returns a report), and the **animation
intent** for NAI Browser's "Wukong Tetris" cinematic. It is the authoritative
contract for the `server-beehive` HTTP surface.

Build & binary: `build-win-ext-full\bin\llama-server.exe`, beehive mode (`--beehive`).

---

## 1. Concept: Truth Engine + Showmanship Engine

The Assembly output is deliberately split so **measured data is never faked** and
the **visualization is never presented as measurement**:

- **Truth Engine (100 % real).** The Lab-Protocol JSON carries only what the
  process actually measured: dedup registry stats, per-model load status, the
  persistent layer-manifest metadata (storage type + native-domain fingerprint +
  sha256 + dims), and a load timeline. Nothing synthetic enters these fields.
- **Showmanship Engine (procedural, renderer-side).** The Wukong-Tetris animation
  interpolates between real data points. Identical layers collapse into one block,
  shared allocations change color, blocks assemble into a cube over
  `presentation.cinematic_duration_ms`. These are MIDPOINTS computed in the
  renderer, never injected back into the report as if they were measured IR.

Inference is never delayed by animation. The Assembly POST returns the report as
soon as the models are resident; the animation can play, be skipped, or be
replayed later from a saved JSON.

Two viewer modes for NAI Browser:

- **Cinematic** — ~50 % measured reality + ~50 % visual choreography (max WOW).
- **Engineering X-Ray** — actual tensors, shared-allocation addresses, timing,
  exact savings; no artistic interpolation.

---

## 2. Endpoints

Base: `http://<host>:<port>` (beehive mode). JSON in/out unless noted.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/beehive/assemblies` | Load the named models in one run and return the Lab-Protocol JSON. |
| `GET` | `/v1/beehive/assemblies/:assemblyId/lab-report` | Stable snapshot of the current assembly's Lab Protocol. |
| `GET` | `/v1/beehive/lab-report` | Current Lab-Protocol snapshot (plain alias). |
| `GET` | `/v1/models` | Registered models + `status` (`loaded`/`unloaded`). |
| `POST` | `/v1/chat/completions` | Route a request to the matching in-process model by `model` id. |

### 2.1 `POST /v1/beehive/assemblies`

Body:

```json
{ "models": ["omn", "brow", "abl"] }
```

`models` is an array of model **ids** previously registered via
`--beehive-model id=path`. Each named, not-yet-loaded model is loaded in-process
into the shared dedup shelf; already-loaded models are left resident. Unknown ids
are logged and skipped (never fatal). The response is the full Lab-Protocol JSON.

Response `200` (Lab-Protocol document, described below). This is the **single
endpoint for multi-model load in one run + report**.

### 2.2 `GET /v1/beehive/assemblies/:assemblyId/lab-report`

Returns the same Lab-Protocol document. `:assemblyId` is httplib param syntax
(matches any id; the process holds one assembly per run, so the id is accepted
for contract-compatibility and the current report is returned). `assembly_id` in
the body names the actual run.

### 2.3 `GET /v1/beehive/lab-report`

Plain alias of the same snapshot when the caller does not bind an id.

---

## 3. Lab-Protocol JSON schema (`schema_version: "1.0"`)

```json
{
  "schema_version": "1.0",
  "assembly_id": "beeasm-<epoch_ms>-<seq>",
  "models": [
    { "id": "brow", "path": "D:\\...\\lfm3-vl-2b-browseragent-1f00-Q8_0.gguf", "status": "loaded" }
  ],
  "optimization": {
    "measurements": {
      "resident_weight_bytes": 5233893632,
      "resident_weight_gib": 4.8744,
      "total_aliases": 853,
      "shape_groups": 11,
      "semantic_aliases": 0,
      "physical_vram_bytes": 0,
      "physical_vram_gib": 0.0
    },
    "tiers": {
      "tier1_byte_dedup":   { "unique_layers": 726, "aliases": 853 },
      "tier2_shape_groups": { "shape_owners": 11 },
      "tier3_dtype_faithful": { "fingerprinted_layers": 726, "semantic_aliases": 0 }
    },
    "manifest": {
      "manifest_layers": 726,
      "storage_types": { "0": 265, "8": 461 },
      "fingerprinted": 726,
      "sample_layers": [
        { "name": "blk.0.attn_norm.weight", "dims": "2048,1,1,1", "type_id": 0,
          "fp": "e7bd380f58328d8b", "sha256": "a539fc09..." }
      ]
    }
  },
  "timeline": [
    { "t_ms": 0,   "event": "assembly.start",         "detail": "beehive single-process begin" },
    { "t_ms": 50,  "event": "model.loaded",           "model": "abl" },
    { "t_ms": 100, "event": "dedup.shelf.finalized",  "unique_layers": 726, "aliases": 853, "resident_weight_bytes": 5233893632 }
  ],
  "presentation": {
    "theme": "wukong-tetris",
    "cinematic_duration_ms": 18000,
    "procedural_interpolation": true,
    "tagline": "3 models. 1 hive. Zero ketchup."
  }
}
```

### 3.1 Field reference

| Field | Type | Meaning | Source |
|---|---|---|---|
| `schema_version` | string | Contract version (`"1.0"`). | static |
| `assembly_id` | string | `beeasm-<epoch_ms>-<seq>` unique per run. | generation |
| `models[]` | array | Registered models; `id`, `path`, `status`. | registry |
| `optimization.measurements.resident_weight_bytes` | int | VRAM bytes materialized in the dedup shelf (weights only). | `llama_dedup_get_stats()` |
| `optimization.measurements.total_aliases` | int | Extra models pointing at already-resident layers (Tier-1 + shape). | registry |
| `optimization.measurements.shape_groups` | int | Distinct `(shape,type)` allocation groups across the shelf (Tier-2 infra). | registry |
| `optimization.measurements.semantic_aliases` | int | Same-type semantic reuses (Tier-3). 0 = none (VL twins are byte-identical, caught by Tier-1). | registry |
| `optimization.measurements.physical_vram_bytes` | int | **`0` unless the harness fills it.** The process knows resident weights; measured board delta is injected by the renderer/bench side. | injected |
| `optimization.tiers.*` | object | Per-tier reality. `dtype_faithful.fingerprinted_layers` = how many layers carry a native-domain fingerprint. | registry + manifest |
| `optimization.manifest.storage_types` | map<int,count> | GGML type id → layer count persisted (e.g. `0`=F32 ×265, `8`=Q8_0 ×461). | `beehive-cache/manifest.json` |
| `optimization.manifest.fingerprinted` | int | Layers with a non-zero native-domain fingerprint. | manifest |
| `optimization.manifest.sample_layers[]` | array | Up to 12 real layer entries with `name`, `dims`, `type_id`, `fp` (hex), `sha256`. | manifest |
| `timeline[]` | array | Load events: `assembly.start`, `model.loaded`, `dedup.shelf.finalized`. | process |
| `presentation.*` | object | Renderer hints only (theme, duration, interpolation flag). Never measured data. | static |

### 3.2 `storage_types` legend (relevant ids)

`0`=F32, `1`=F16, `8`=Q8_0, `30`=BF16, `34``/`35`=TQ1_0/TQ2_0, `24/25/26/27`=I8/I16/I32/I64, others per `GGMLQuantizationType`.

---

## 4. Wukong Tetris — animation intent

This is the **Showmanship Engine** intent, implemented by NAI Browser from the
report's Truth-Engine data. All visual midpoint states are procedural.

### 4.1 Setup

The renderer receives the `manifest.sample_layers` + per-tier counts + timeline
as the "skeleton" of the animation. For each unique layer:

- Each layer is a small **Tetris-style block** on an invisible 3D grid.
- A block's **color / shade** is derived from its `type_id` (F32 vs Q8_0 give
  distinct visual families) and its **dedup role**:
  - **unique** (materialized once) — full saturation.
  - **alias** (shared with an earlier model at the same address) — tinted toward
    the cube core / dimmed to signal "shared, not re-allocated".
- Block **shape** reflects `dims` aspect (e.g. long FFN matrices vs small norms).

### 4.2 Choreography (18 s, `cinematic_duration_ms`)

1. **0–4 s — Arrival.** Each model is a cloud of distinct blocks ("3 models =
   3 swarms"). Blocks carry the model id.
2. **4–10 s — Dedup.** Blocks that resolved to the **same address** (Tier-1 byte
   match, or Tier-2 same-shape group) visually merge — two blocks become one,
   which re-materializes at a single point and the twin fades. This is the core
   Wukong "folding tetris into a cube" moment.
3. **10–14 s — Assembly.** The surviving unique blocks fly into a single compact
   cube, layer by layer, matching the `timeline` order (`model.loaded` →
   `dedup.shelf.finalized`).
4. **14–18 s — Seal.** The cube completes, `assembly_id` + `tagline` render, and
   the animation parks (skippable; replays from saved JSON).

### 4.3 Rules (correctness)

- **Only TRUE events are anchored.** Every merge, collapse, and assembly step is
  driven by real data (a real alias count, a real shape group, a real fingerprint
  tie). Midpoint positions are interpolated, but a merge only happens where the
  report says addresses are shared.
- **No fake IR in the report.** The `timeline` emits real, low-frequency events.
  The renderer may subdivide the 18 s window for smoothness, but never invents a
  measurement that the JSON doesn't support.
- **Non-blocking.** Serving starts the moment `POST` returns; the animation is a
  client-side replay, safe to skip (`ESC`) or scrub.

### 4.4 The tagline

After the last Tetris block seats into the cube:

> **3 models. 1 hive. Zero ketchup.**

---

## 5. Example flow

```http
POST /v1/beehive/assemblies
Content-Type: application/json

{ "models": ["omn", "brow", "abl"] }
```

→ `200` Lab-Protocol JSON. NAI Browser:

1. Parses `optimization` + `timeline` → wires the Wukong-Tetris cinematic (skeleton
   from `manifest.sample_layers`, merges from alias/tier counts).
2. Starts serving chat requests via `POST /v1/chat/completions` (model id in body),
   **concurrently across models**, while the animation replays the assembly.
3. Offers "Replay" from the saved `assembly_id` via
   `GET /v1/beehive/assemblies/{id}/lab-report`.

---

## 6. Notes

- Endpoint paths use httplib `:param` syntax (`:assemblyId`), matching existing
  routes such as `/slots/:id_slot`.
- `physical_vram_bytes` intentionally starts at 0 from the server; the lab harness
  should substitute `memory.used[up] − memory.used[down]` (this NVIDIA instance
  does not expose per-process VRAM). This keeps the process-side report 100 %
  honest about what the process itself knows.
- See `scripts/beehive_lab_protocol.md` for the measured Tier 1–3 + infra numbers
  this contract reports back to you.

---

## 7. Temporal Forensic AV analysis (`/v1/beehive/temporal`)

Exhaustive audio+video analysis for LFM3-Omni over long (multi-128K) sources:
N fresh-context windows over two-frame temporal groups, each emitting a native
`emit_temporal_evidence` tool call, then a clean-context final synthesis.

Spec + prompts: `lfm3_omni_temporal_forensic_handoff_v2.zip` (navýš v2 forbids
`fps=` resampling in the exhaustive path; source FPS is metadata, decoded frame
PTS is the clock).

### 7.1 Model-facing audio format (VERIFIED)
`383,584 B / 4 = 95,896 f32 mono samples @ 16 kHz = 5.9935 s` for the illit
6 s clip — mono, 16 kHz, pcm f32 is exactly what the LFM3-Omni audio projector
expects (`mtmd_get_audio_sample_rate` = 16000).

### 7.2 Audio continuity rule (CRITICAL — do not regress)
- **Stereo 48 kHz is the source truth** in the browser/ring transport (archival /
  other consumers). The LFM3 branch must get **mono 16 kHz**.
- **One continuous resampler over the whole soundtrack**, not per-video-chunk.
  Per-chunk resample creates phase/filter discontinuity and sample-count drift at
  chunk boundaries. Downmix→mono + resample→16k must run once over the full
  timeline; **chunking happens after** the continuous audio transform.
- Model-facing path: `-ac 1 -ar <rate> -f f32le` in a **single** ffmpeg run over
  the whole clip (mtmd `load_soundtrack()` does this *once per request*), then the
  per-group slices are **contiguous half-open intervals** `[B[g], B[g+1])` whose
  concatenation equals the full soundtrack — zero drift, zero gaps (see
  `soundtrack_slice()` in `tools/mtmd/mtmd-helper.cpp`).
- **N-window orchestration must NOT re-run ffmpeg per window.** Windows are
  PTS-cut slices from one continuous decoded/resampled PCM timeline, not
  independently-resampled sub-clips.

### 7.3 Verified build state (beellama)
- P0-G: native `emit_temporal_evidence` tool calling works with the **flattened**
  schema (the deep nested schema crashed the LFM3 AutoParser; drop
  `subjects`/`source_refs`/`additionalProperties` on nested items).
- P0-D: continuous PTS-sliced soundtrack reaches the model's ears — LFM3-Omni
  reported beats/lyric and AV-synced choreography on the illit clip.
- `/v1/beehive/temporal` window planner live (two-frame groups, stride+overlap,
  group-aligned boundaries); inference loop + synthesis still to be wired (it
  must use ONE persistent decode session per the rule in 7.2).
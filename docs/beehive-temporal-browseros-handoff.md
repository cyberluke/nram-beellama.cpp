# BrowserOS ↔ Beehive Temporal AV — Handoff (for the agent finalizing the BrowserOS side)

> Durable cross-repo handoff for the **exhaustive temporal forensic audio+video analysis**
> feature. Produced by the beellama-side agent so the BrowserOS finalization agent can
> pick up without re-deriving the protocol. Beellama side is DONE + verified (see §1–§4);
> BrowserOS side is scoped here (§5–§8) for YOU to implement + build + deploy in that repo.
>
> Canonical spec (the whole feature): `lfm3_omni_temporal_forensic_handoff_v2.zip` (in
> `C:\Users\lukes.COREI9\Downloads\`), README + files 01–09. The beellama implementation
> follows v2 (no FPS resampling in exhaustive path; source FPS is metadata, decoded PTS is clock).

---

## 0. TL;DR

- Beellama now serves `POST /v1/beehive/temporal` (behive fork): two-frame-group window
  planner, native `emit_temporal_evidence` tool calling, and a continuous
  PTS-sliced mono-16 kHz soundtrack that LFM3-Omni **hears** (verified on the illit MKV).
- BrowserOS must: (a) make the ring capture **AV together** (video + its audio track, the
  ring stays **stereo-48k source-truth**), and (b) wire the temporal analysis tool to call
  `/v1/beehive/temporal` with the clip fed **in-memory** (no disk / no base64).
- Hard rule (see §4.2): the LFM3 branch gets **mono 16 kHz** from **one continuous
  resampler**; the ring/browser NEVER downmixes or resamples per chunk.

---

## 1. Beellama verified state (DONE — do not redo)

Working binary: `C:\Users\lukes.COREI9\AppData\Local\Temp\kilo\beellama-build\bin\llama-server.exe`
(beehive build; the canonical deploy source dir `build-win-ext-full` is read-only — copy
the runtime files OUT of the build-bin, never into the ext-full project).

| Piece | Status | Evidence |
|---|---|---|
| Native LFM3 tool calling | DONE | `emit_temporal_evidence` via `tools[]` + `tool_choice=required`; emits schema-constrained calls |
| Flattened tool schema | REQUIRED | deep schema (subjects[]/source_refs[]/additionalProperties) **crashes** the LFM3 AutoParser; flattened works |
| Continuous soundtrack to ears (P0-D) | DONE | `95,896` f32 mono @ 16k = `5.9935 s` for illit; model reported beats/lyric + AV-synced choreography |
| `/v1/beehive/temporal` endpoint | DONE (functional) | plans 2-frame-group windows, runs **N fresh-context evidence calls**, collects evidence, runs **final synthesis** |

Key beellama files touched: `tools/mtmd/mtmd-helper.cpp` (load_soundtrack gate fix +
`soundtrack_slice` contiguous PTS slicing + detail-budget solver), `tools/server/server.cpp`
(route), `tools/server/server-temporal.{h,cpp}` (new: planner + evidence loop + synthesis),
`tools/server/server-common.{h,cpp}` (`media_register_peek` for re-ingestion),
`tools/mtmd/mtmd.h/.cpp` (`mtmd_get_video_frame_tokens`).

---

## 2. The `emit_temporal_evidence` tool contract (flattened — USE THIS)

Flattened schema (the one that works — in the beehima test request too):

```json
{
  "type": "function",
  "function": {
    "name": "emit_temporal_evidence",
    "description": "Commit evidence from the current absolute audiovisual window for later cross-window synthesis",
    "parameters": {
      "type": "object",
      "properties": {
        "window_start_ms": {"type": "integer", "description": "Absolute source window start in ms"},
        "window_end_ms":   {"type": "integer", "description": "Absolute source window end in ms"},
        "observations": {
          "type": "array",
          "description": "Chronological audiovisual evidence records",
          "items": {
            "type": "object",
            "properties": {
              "start_ms": {"type": "integer"},
              "end_ms":   {"type": "integer"},
              "kind": {"type": "string"},
              "observation": {"type": "string"},
              "interpretation": {"type": "string"},
              "uncertainty": {"type": "string"}
            },
            "required": ["start_ms", "end_ms", "kind", "observation"]
          }
        },
        "boundary_continuity": {"type": "array", "items": {"type": "string"}},
        "unresolved": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["window_start_ms", "window_end_ms", "observations", "boundary_continuity", "unresolved"]
    }
  }
}
```

> ⚠️ Do NOT use the deep schema from `07_EMIT_TEMPORAL_EVIDENCE_TOOL.json` verbatim — it
> crashes the fork. Flatten as above (drop `subjects[]`, `source_refs[]`,
> `additionalProperties:false`; keep the core fields).

---

## 3. `/v1/beehive/temporal` request/response contract

```
POST {beehiveBase}/v1/beehive/temporal
{
  "model": "omni",                 // loaded LFM3-Omni instance id
  "media_url": "media://<id>",     // AV clip already ingested in-process (see §6)
  "user_prompt": "<<the agent's analysis objective>>",
  "total_ms": 6000,                // source duration ms (from the ring) — REQUIRED for windowing
  "window_ms": 60000,              // window length ms (default 60000)
  "overlap_ms": 1000,              // deterministic overlap ms (default 1000)
  "src_fps": 30.0,                 // source FPS (metadata only, PTS is the clock)
  "max_generation": 3072,          // per-window evidence token budget (keep >= 3072; smaller drops tool calls)
  "safety": 1024,
  "execute": true                  // true = run N evidence calls + synthesis inline; false = plan only
}
```
Behavior (verified): with `execute:true` it returns `{ status, windows[], evidence[], tool_call_failures, synthesis, … }`.
- `evidence[]` = one entry per successful window, each holding the parsed
  `emit_temporal_evidence` arguments (`window_start_ms`,`observations[]`,…).
- `tool_call_failures` = windows where the model didn't emit a parseable tool call after the
  **one deterministic retry** (the loop is failure-isolated — a failed window is counted and
  skipped, never breaks synthesis).
- **1 deterministic retry (v added):** for every window whose first evidence call misses the
  tool call, the endpoint runs **one fresh-context retry with a LARGER generation budget**
  (budget1 = min(max_generation, 3072); budget2 = min(max(4096, 2×max_generation), 8192)).
  The retry is the same media / timestamps / tool contract — the evidence shape is unchanged.
  It is NOT a silent fallback. Fresh media id per attempt (the register is single-consumption).
- **Coverage observability (new response fields):** `window_stats[]` (per window: `ok`,
  `used_retry`, and `attempts[]` each with `{attempt, ok, finish_reason, prompt_tokens,
  generation_tokens, budget, parser_state}`), plus aggregates `coverage_first_pass`,
  `coverage_with_retry`, `retry_saves`, `windows_total`, `windows_first_ok`. parser_state ∈
  `ok | repaired | no_tools_array | bad_tool_args_json | bad_response_json | …`.
- **Lenient evidence parser (new):** when the model emits a tool call whose `arguments` JSON
  fails strict parse, it is repaired (unescaped inner double-quotes inside narrative strings,
  e.g. `curved into "frame" shapes`) and re-parsed — `parser_state="repaired"`. Evidence content
  is unchanged; only the parsed form is accepted.
- **Measured (illit 6s, window_ms=4000/overlap=1000 → 3 windows, 3 trials):** first-pass ≈ 11%
  (1/9), with-retry ≈ 44% (4/9), retry_saves=3. Dominant failure: `finish_reason=length`,
  `parser=no_tools_array` — a 2B model burns its budget on verbose narration before reaching a
  tool call even at 6144 tokens. Parsing (`bad_tool_args_json`) is a rare secondary mode (now
  repaired). These are EMPIRICAL, not guesses; raise `window_ms` to shorten windows or the
  per-window budgets to push coverage higher.
- `synthesis` = the clean-context final answer over all evidence.
- Client should pass `total_ms` from the ring duration; without it the planner defaults to a single window.

### 3.1 Measured throughput (60fps, 30s source, window_ms=10000/overlap=2000 → 4 windows)
- **End-to-end: 231.2s for a 30s clip** (≈ 7.7× slower than realtime); elapsed-per-window ≈ 58s.
- **Dominant cost is DECODE-PREFILL, not generation.** Every window-attempt re-decodes the WHOLE
  clip to the same **384 frames @ 320×192** (`[mtmd-clamp] budget=126976 tpf=60 n_frames=384
  est=320x192`) and **re-prefills a constant ~13,955 tokens** — the decode budget clamp does NOT
  trim to the source-time window (all 4 logical windows re-prefill the full clip). 6 attempts
  (4 first + 2 retry) = 83,729 media-prefill tokens.
- **Per-attempt model time:** media prefill ≈ 6.1s @ ~2291 tok/s; evidence gen ≈ 640–6144 tokens
  @ ~86–99 tok/s (~85 tok/s at 2B Q8). Synthesis: trivial (~5.2s, 1165-token prompt).
- **Implication (matches the §4.2 continuity rule):** the efficient target is ONE persistent
  decode session → PTS-cut window slices, so each window prefills only its window's frames
  instead of the whole clip. The current path re-ingests the full clip per attempt
  (single-consumption design) — correct and failure-isolated, but decode-bound. Do NOT swap in
  per-chunk ffmpeg to "fix" this (breaks §4.2); build the persistent decode for the real win.
- **Coverage on this clip:** first-pass 50%, with-retry 50% (2 retries both failed; synthetic
  content gave shorter cleaner calls than illit). Loop-guard hit once (task 4482, token 1601).

Prompts the BrowserOS agent should load from the v2 zip:
`04_AGENT_STORE_SYSTEM_PROMPT.txt` (window system prompt), `06_FIRST_REQUEST_TEMPLATE.md`
(exact message/media/tool composition), `08_FINAL_SYNTHESIS_PROMPT.txt` (synthesis).

---

## 4. Audio & the ring — the rules that must not break

### 4.1 Formats
- **Ring / transport source truth: stereo, whatever the source sample rate (usually 48 kHz).**
  Keep it stereo in the browser ring AND when feeding the engine; it is the archival truth.
- **LFM3 branch (behive) expects mono 16 kHz f32 PCM**, sliced per two-frame group.

### 4.2 The continuity rule (CRITICAL)
- Downmix → mono + resample → 16 kHz must run **once over the full soundtrack**, as one
  continuous transform. **Never** resample/downmix per video chunk or per request slice —
  chunk boundaries would then carry phase/filter discontinuity + sample count drift.
- Chunking (the window slices) happens **after** the continuous mono-16k transform, as
  PTS-boundary half-open intervals.

---

## 5. BrowserOS — what to do (implement + build + deploy)

### 5.1 Make the ring capture audio + video together
Files (BrowserOS repo): `apps/app/lib/video-capture/session.ts`, `apps/app/lib/video-capture/ring.ts`,
`apps/app/lib/media-capture/tab-buffer-lru.ts`.

Current state: ring captures **video-only** WebM (MIME `video/webm;codecs=vp9|vp8|avc1`), no
audio track (audit: `session.ts` `start()`, `RING_MIME_ORDER`, `new MediaRecorder(stream,…)`).

To change:
1. `captureStream()` already returns the stream; **add the element's audio tracks** if absent
   (`stream.getAudioTracks()`), or merge the audio DSP stream. Reference: `MediaRecorder`
   with a stream carrying both tracks muxes AV into one WebM.
2. **MIME must include audio**: `video/webm;codecs=vp9,opus` (else `vp8,opus`). Add these
   to `RING_MIME_ORDER` **before** the video-only entries.
3. Keep the ring **stereo-48k**. Feed to beehive as-is (the engine downmixes). Do NOT
   downmix in the browser.
4. Verify one ring chunk now contains both a video and an audio track (MediaRecorder
   `ontrack`/`data` shape) so `exportLastSeconds()` returns an AV blob, not video-only.

### 5.2 Wire the temporal tool → `/v1/beehive/temporal` (in-memory) — DONE
Files: `apps/server/src/lib/beellama/beehive.ts` (client), `apps/server/src/api/routes/beehive.ts`,
`apps/app/lib/video-analysis/input-video-caller.ts`, `apps/server/src/api/services/mcp/browser-mcp-module.ts`
(tool registration), `apps/server/src/lib/video-temporal/handler.ts`,
`packages/browser-mcp/src/tools/video-temporal-analyze.ts` (tool contract),
`apps/server/src/api/routes/video-temporal-{events,snapshot}.ts`,
`apps/app/entrypoints/background/videoTemporalSubscriber.ts` (extension AV export).

**Implemented (typed green; server + browser-mcp tsc exit 0; `handler.test.ts` 3/3 pass):**
- `apps/app/lib/video-capture/session.ts` — ring captures audio+video (audio-capable MIME
  `video/webm;codecs=vp9,opus` first; `capturedAudio` flag; stereo-48k source-truth kept).
- `apps/server/src/lib/beellama/beehive.ts` — `beehiveIngestMedia()` (in-memory `media://` feed)
  and `beehiveTemporal()` (POST `/v1/beehive/temporal`), plus types.
- **Exhaustive E2E path (new):** tool arg `mode:'sheet'|'exhaustive'`; result adds `mode/synthesis/evidence`.
  `video-temporal-snapshot` accepts `?mode=av` (raw AV bytes upload); the SSE frame carries
  `mode`+`seconds`; `videoTemporalSubscriber` exports the AV ring blob via content-script
  `runtime.videoCapture.clip` and POSTs the bytes; the server handler resolves the beehive
  LFM3-Omni vision target (`resolveVisionOverride`→cehive baseUrl/model), ingests, runs
  `beehiveTemporal`, and returns the synthesis. Default `sheet` mode is untouched (falls back
  when no beehive target or no synthesis — never a silent 500).
- **Legacy/contact-sheet path retained as the default fallback.**
- Build/deploy of the BrowserOS app is the only remaining step (the **other agent's** scope).

For the build (other agent): bundle `apps/app` + `apps/server` + `packages/browser-mcp`, run the
video-temporal test (`bun test apps/server/src/lib/video-temporal/handler.test.ts`), and deploy.

### 5.3 Besplušování (resolution of the previously-reported tool-loop bug)
The "temporal tool re-invoking forever" you saw: the ring + the tool-loop stop-condition
(`stepCountIs`) were reported missing in the **deployed** server. Ensure the stop-condition
(do not re-invoke temporal sub-analysis pulses indefinitely; return after evidence +
synthesis) is present in the handler you deploy.

---

## 6. In-memory media ingest (behive) — the pipe/no-disk path
`POST {base}/v1/beehive/media` with the raw AV bytes (`application/octet-stream`) returns
`{media_url:"media://<id>", bytes, status:"ok"}`. It is single-consumption (consume once)
+ TTL-bounded — so re-ingest per request; never reuse an id. This is the named-pipe /
in-memory replacement for filesystem or base64. See the earlier `beehiveIngestMedia()`
client (already exists in `apps/server/src/lib/beellama/beehive.ts`).

---

## 7. Deploy notes (BrowserOS agent — finish build + deploy)

- Beellama built binary is on C: temp dir `C:\Users\lukes.COREI9\AppData\Local\Temp\kilo\beellama-build\bin`
  (llama-server.exe + DLLs). BrowserOS bundled `llama-server.exe` lives in
  `packages/browseros-agent/resources/binaries/beellama/` (and `resources/binaries/beellama` +
  `dist/prod/.../resources/binaries/beellama`). Copy the runtime files from the build-bin
  into those bundle locations, then run the BrowserOS build (tsc, namespace/bundling) and
  deploy — the build/deploy of the **BrowserOS** project is what You finalize.
- Beellama `build-win-ext-full` is read-only; never copy INTO it or move/delete it.
- Keep AAF delta only; preserve pins; do not commit unless explicitly asked.

---

## 8. Acceptance (run these to call it done)
- Ring chunk contains video + audio (stereo-48k).
- `/v1/beehive/temporal` returns a window plan for the clip.
- One window's evidence call returns parseable `emit_temporal_evidence` args that include
  **audio observations** (the model HEARS the music — e.g. beat/kick, lyrics), not just visuals.
- Synthesis (v2 prompt) merges overlapping-window duplicates and preserves timestamps.
- The tool-loop stop-condition fires (no infinite re-invocation).
- No filesystem or base64 path is used for the AV media on the beehive fast path.
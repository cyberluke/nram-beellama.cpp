// tools/kelvin-tui/src/presets.h
#pragma once

#include <string>
#include <vector>

// A fully-resolved llama-server launch configuration, tuned for a given
// hardware class. Encapsulates all the "don't bother me with technicalities"
// parameters for a model family.
struct ServerPreset {
    std::string id;
    std::string label;          // human-friendly
    std::string description;    // one short line

    int  ctx_size      = 163840;
    int  parallel      = 2;
    bool mtp           = true;
    int  mtp_draft_max = 2;
    int  gpu_layers    = 999;
    int  n_predict     = 32768;

    // KV cache quantization (VRAM saving).
    std::string cache_k = "q5_0";
    std::string cache_v = "q5_0";
    std::string spec_cache_k = "q8_0";
    std::string spec_cache_v = "q8_0";

    // Reasoning (tuned to not kill context - github issue workaround).
    std::string reasoning       = "on";
    std::string reasoning_effort = "low";
    int         reasoning_budget = 12288;
    std::string reasoning_budget_message = "Let's final";
    bool        preserve_reasoning       = false;

    // Sampling (Qwen3 thinking mode recommended).
    double temp    = 1.0;
    double top_p   = 0.95;
    int    top_k   = 20;
    double min_p   = 0.0;

    // Performance.
    int batch_size  = 2048;
    int ubatch_size = 128;
    int threads     = 24;
    bool cont_batching = true;
    bool cache_prompt  = true;
};

// The proven preset catalog (synthesized from the Qwen3.8-27B tuning script).
const std::vector<ServerPreset>& preset_catalog();

// VRAM-aware autopilot: pick the best preset for this machine/model.
// `want_parallel` = user wants 2 concurrent clients (true) or 1 (false).
// `vram_gb` = total GPU VRAM, or 0 to auto-detect.
// Rounds to a human label + resolution; the caller builds the spec.
const ServerPreset& autopilot_preset(bool want_parallel, int vram_hint_gb);

// Best-effort: query total VRAM GB via nvidia-smi. Returns 0 if unavailable.
int detect_vram_gb();
// tools/kelvin-tui/src/presets.cpp
#include "presets.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <windows.h>
#include <sstream>

// The tuned catalog. Values mirror the qwen38-27B RTX 4090 script.
static const std::vector<ServerPreset> kPresets = {
    {
        "160k", "Full 160k single", "Huge single-slot context, MTP on",
        163840, 1, true, 2, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "128k", "128k single", "Big single-slot context, MTP on",
        131072, 1, true, 2, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "96k-mtp-dual", "Dual 96k (2 apps)", "Two concurrent clients, ~96k each",
        163840, 2, true, 2, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "96k-mtp", "96k single", "Large single-slot context, MTP on",
        98304, 1, true, 2, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "64k-mtp", "64k fast", "Fast small context, MTP on",
        65536, 1, true, 2, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "parallel-2x64k", "2 workers 64k", "Two independent 64k clients, no MTP",
        131072, 2, false, 4, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
    {
        "parallel-2x48k", "2 workers 48k", "Two lighter 48k clients, no MTP",
        98304, 2, false, 4, 999, 32768,
        "q5_0","q5_0","q8_0","q8_0",
        "on","low",12288,"Let's final",false,
        1.0,0.95,20,0.0,
        2048,128,24,true,true,
    },
};
const std::vector<ServerPreset>& preset_catalog() { return kPresets; }

int detect_vram_gb() {
    // nvidia-smi is the most reliable cross-NVIDIA check.
    const char* cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits";
    FILE* f = _popen(cmd, "r");
    if (!f) return 0;
    char buf[64];
    int vram = 0;
    if (fgets(buf, sizeof(buf), f)) {
        vram = atoi(buf);
    }
    _pclose(f);
    return vram > 0 ? vram / 1024 : 0; // MiB -> GiB
}

const ServerPreset& autopilot_preset(bool want_parallel, int vram_hint_gb) {
    int vram = vram_hint_gb > 0 ? vram_hint_gb : detect_vram_gb();
    // Fallback: assume a solid discrete GPU (24 GB class) if undetected.
    if (vram <= 0) vram = 24;

    // A 24 GB card reports ~22-23 GB usable (framebuffer reservation), so
    // treat >= 20 as "big GPU". >= 24 GB can hold a big model + big KV.
    if (want_parallel) {
        if (vram >= 20) return kPresets[2]; // 96k-mtp-dual
        if (vram >= 12) return kPresets[5]; // parallel-2x64k (no MTP)
        return kPresets[6];                 // parallel-2x48k
    }
    if (vram >= 20) return kPresets[3]; // 96k-mtp
    if (vram >= 12) return kPresets[4]; // 64k-mtp
    return kPresets[4];                 // 64k-mtp safest
}
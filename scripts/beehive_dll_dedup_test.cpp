// beehive_dll_dedup_test.cpp — load a model twice via the built llama.dll and
// prove the second load's identical layers are DEDUPED (aliases > 0, so VRAM is
// not doubled). Uses LoadLibrary + GetProcAddress so no llama.lib import is needed.
//   cl /std:c++17 beehive_dll_dedup_test.cpp /Fe:beehive_dll_dedup_test.exe
#include <windows.h>
#include <cstdio>
#include <string>

typedef struct llama_model * (*fn_load)(const char *, const void *);
typedef void (*fn_free)(struct llama_model *);
typedef struct llama_dedup_stats {
    size_t unique_layers, total_aliases, resident_bytes, shape_owners;
} llama_dedup_stats;
typedef llama_dedup_stats (*fn_stats)(void);

struct llama_params;
extern "C" {
typedef void * (*llama_model_params_default_t)(void);
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: beehive_dll_dedup_test model.gguf\n"); return 2; }
    HMODULE dll = LoadLibraryA("llama.dll");
    if (!dll) { printf("FAIL: cannot load llama.dll (err %lu)\n", (unsigned long)GetLastError()); return 1; }

    auto load  = (fn_load) GetProcAddress(dll, "llama_model_load_from_file");
    auto stats = (fn_stats) GetProcAddress(dll, "llama_dedup_stats");
    auto par   = (llama_model_params_default_t) GetProcAddress(dll, "llama_model_default_params");
    if (!load || !stats || !par) { printf("FAIL: missing export\n"); return 1; }

    const char * model = argv[1];
    llama_dedup_stats s0 = stats(); // base
    printf("base dedup: unique=%zu aliases=%zu\n", s0.unique_layers, s0.total_aliases);

    // Load #1 — every layer materializes (unique grows).
    struct llama_model * m1 = load(model, par());
    llama_dedup_stats s1 = stats();
    printf("after load1: unique=%zu aliases=%zu (aliases should stay ~0)\n",
        s1.unique_layers, s1.total_aliases);

    // Load #2 — the SAME weights. Every identical layer is deduped => aliases jump.
    struct llama_model * m2 = load(model, par());
    llama_dedup_stats s2 = stats();
    printf("after load2: unique=%zu aliases=%zu resident_bytes=%zu\n",
        s2.unique_layers, s2.total_aliases, s2.resident_bytes);
    printf("deduplicated aliases in second load: %zu\n", s2.total_aliases - s1.total_aliases);

    bool pass = (s2.unique_layers <= s1.unique_layers + 1) && (s2.total_aliases > s1.total_aliases);
    printf(pass ? "BEEHIVE DLL DEDUP TEST: PASS\n" : "BEEHIVE DLL DEDUP TEST: FAIL\n");
    return pass ? 0 : 1;
}
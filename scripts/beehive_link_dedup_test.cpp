// beehive_link_dedup_test.cpp — link src\llama.lib, load a model twice, prove
// identical layers are DEDUPED (aliases grow, unique stays ~1x).
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: beehive_link_dedup_test model.gguf\n"); return 2; }
    const char * model = argv[1];

    llama_backend_init(); // loads ggml backends (CUDA/etc.) from the DLL dir
    ggml_backend_load_all(); // fork (LLAMA_BACKEND_DL): enumerate loadable backend DLLs
    llama_dedup_stats b = llama_dedup_get_stats();
    printf("base unique=%zu aliases=%zu\n", b.unique_layers, b.total_aliases);

    llama_model_params lp = llama_model_default_params();
    struct llama_model * m1 = llama_model_load_from_file(model, lp);
    if (!m1) { printf("FAIL: model 1 did not load\n"); return 1; }
    llama_dedup_stats after1 = llama_dedup_get_stats();
    printf("after1 unique=%zu aliases=%zu\n", after1.unique_layers, after1.total_aliases);

    struct llama_model * m2 = llama_model_load_from_file(model, lp);
    if (!m2) { printf("FAIL: model 2 did not load\n"); return 1; }
    llama_dedup_stats after2 = llama_dedup_get_stats();
    printf("after2 unique=%zu aliases=%zu\n", after2.unique_layers, after2.total_aliases);
    printf("aliases added by 2nd load: %zu\n", after2.total_aliases - after1.total_aliases);

    bool pass = (after2.total_aliases > after1.total_aliases)
             && (after2.unique_layers <= after1.unique_layers + 1);
    printf(pass ? "BEEHIVE LINK DEDUP TEST: PASS\n" : "BEEHIVE LINK DEDUP TEST: FAIL\n");
    llama_free_model(m2);
    llama_free_model(m1);
    return pass ? 0 : 1;
}

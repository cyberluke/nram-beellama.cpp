// beehive_manifest_test.cpp — load a real model; prove durable layer blobs +
// sha256 manifest entries are written (docker-like delta), then reload and
// confirm count stays stable (unchanged layers not re-written).
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: beehive_manifest_test model.gguf\n"); return 2; }
    const char * model = argv[1];
    llama_backend_init();
    ggml_backend_load_all();

    llama_dedup_manifest_set_dir("beehive-cache-test");
    llama_dedup_manifest_load();
    size_t before = llama_dedup_manifest_count();
    printf("manifest entries before load: %zu\n", before);

    llama_model_params lp = llama_model_default_params();
    struct llama_model * m1 = llama_model_load_from_file(model, lp);
    if (!m1) { printf("FAIL: model did not load\n"); return 1; }
    size_t after1 = llama_dedup_manifest_count();
    printf("manifest entries after load:  %zu (delta=%zu)\n", after1, after1 - before);

    // Reload: layers unchanged -> manifest should NOT grow (dedup + manifest reuse).
    struct llama_model * m2 = llama_model_load_from_file(model, lp);
    if (!m2) { printf("FAIL: model 2 did not load\n"); return 1; }
    size_t after2 = llama_dedup_manifest_count();
    printf("manifest entries after 2nd:   %zu (delta2=%zu)\n", after2, after2 - after1);

    llama_free_model(m2); llama_free_model(m1);
    llama_dedup_manifest_flush();

    bool pass = (after2 > before) && (after2 == after1);
    printf(pass ? "BEEHIVE MANIFEST TEST: PASS\n" : "BEEHIVE MANIFEST TEST: FAIL\n");
    return pass ? 0 : 1;
}
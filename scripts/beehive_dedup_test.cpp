// beehive_dedup_test.cpp — standalone unit test of the dedup shelf (tier 1 + tier 2).
// Compiles llama-dedup.cpp directly (no llama.dll dependency), proving the dedup
// logic + ABI outside the (currently pre-broken) full llama lib build.
//   cl /std:c++17 /I ggml\include /I src beehive_dedup_test.cpp src\llama-dedup.cpp
#include "llama-dedup.h"
#include <cassert>
#include <cstdio>
#include <cstring>

static void build_key(llama_dedup_key & k, const char * name, ggml_type t, int64_t a, int64_t b, uint64_t h) {
    k.name = name; k.type = t; k.ne[0]=a; k.ne[1]=b; k.ne[2]=1; k.ne[3]=1; k.hash = h;
}

int main() {
    llama_dedup_registry reg;
    ggml_tensor tA{}, tB{}, tC{};
    tA.type = GGML_TYPE_F16; tB.type = GGML_TYPE_F16; tC.type = GGML_TYPE_Q4_K;
    tA.ne[0]=4096; tB.ne[0]=4096; tC.ne[0]=4096;
    tA.ne[1]=4096; tB.ne[1]=4096; tC.ne[1]=4096;

    // ---- Tier 1: exact-layer dedup ----
    llama_dedup_key k1, k2, k3;
    // A and B: same layer name + shape + type + SAME hash -> identical (dedup hit)
    build_key(k1, "blk.3.attn_q.weight", GGML_TYPE_F16, 4096, 4096, 0xABAB);
    build_key(k2, "blk.3.attn_q.weight", GGML_TYPE_F16, 4096, 4096, 0xABAB);
    // C: same name/shape but DIFFERENT hash (finetune changed the values)
    build_key(k3, "blk.3.attn_q.weight", GGML_TYPE_Q4_K, 4096, 4096, 0xCDCD);

    assert(reg.claim(k1, &tA) == true);   // A wins (creates)
    assert(reg.claim(k2, &tB) == false);  // B is a duplicate -> NOT created, alias
    assert(reg.claim(k3, &tC) == true);   // C differs (quant+hash) -> own entry

    // B's layer must resolve to A's tensor (single address)
    assert(reg.find(k2) == &tA);

    // ---- Tier 2: shape/kernel dedup ----
    llama_dedup_shape_key s1{ GGML_TYPE_F16, {4096,4096,1,1} };
    llama_dedup_shape_key s2{ GGML_TYPE_F16, {4096,4096,1,1} };
    llama_dedup_shape_key s3{ GGML_TYPE_Q4_K, {4096,4096,1,1} };
    assert(reg.claim_shape(s1, &tA) == true);
    assert(reg.claim_shape(s2, &tB) == false); // same shape+type -> alias
    assert(reg.find_shape(s2) == &tA);
    assert(reg.claim_shape(s3, &tC) == true);  // same shape, diff type -> own

    // ---- Stats ----
    auto s = reg.stats();
    printf("unique_layers=%zu total_aliases=%zu resident_bytes=%zu shape_owners=%zu\n",
        s.unique_layers, s.total_aliases, s.resident_bytes, s.shape_owners);
    assert(s.unique_layers == 2);   // A + C (B deduped)
    assert(s.total_aliases >= 1);   // B alias
    assert(s.shape_owners == 2);    // F16 shape + Q4_K shape

    // ---- xxh64 determinism ----
    const char * bytes = "the quick brown fox";
    uint64_t h1 = llama_dedup_registry::xxh64((const uint8_t*)bytes, strlen(bytes), 0);
    uint64_t h2 = llama_dedup_registry::xxh64((const uint8_t*)bytes, strlen(bytes), 0);
    assert(h1 == h2);
    printf("xxh64 stable: %016llx\n", (unsigned long long)h1);

    printf("BEEHIVE DEDUP TEST: PASS\n");
    return 0;
}
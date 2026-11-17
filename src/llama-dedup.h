// llama-dedup.h — the beehive kernel.
// "BEEhive": one process, N model contexts, ONE deduplicated weight plane.
//
// Purpose: experimental multi-modal engine. When LFM3-VL + LFM3-Omni + an
// LFM3 finetune + several Qwen models share numerically-identical layers, that
// layer is materialized ONCE in VRAM and every model graph that needs it is
// repinned to the SAME address. Nothing duplicate is ever resident.
//
// The dedup key is provenance+content: the GGUF tensor name (which encodes the
// architecture + layer identity, e.g. "blk.5.attn_q.weight"), its shape, its
// type, and a content hash of the source byte window. Two models collide only
// when their layer is truly identical, so Qwen3/Qwen3.5 or the LFM3 family
// share; a finetune's changed layer (different hash) becomes its own entry.
//
// This is the shared "shelf" the beehive's LLM/omni/audio/video/embedding/
// reranker experts all read from. It is experimental by design.

#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Layer-partition key
// ---------------------------------------------------------------------------

struct llama_dedup_key {
    // Canonical identity of one layer's weights.
    std::string name;   // GGUF tensor name ("blk.5.attn_q.weight")
    ggml_type   type;   // storage type (Q4_K, F16, ...)
    int64_t     ne[4];  // shape
    uint64_t    hash;   // xxh64 of the source byte window (data-ordered)

    auto operator<(const llama_dedup_key & o) const {
        if (name != o.name) return name < o.name;
        if (ne[0] != o.ne[0]) return ne[0] < o.ne[0];
        if (ne[1] != o.ne[1]) return ne[1] < o.ne[1];
        if (ne[2] != o.ne[2]) return ne[2] < o.ne[2];
        if (ne[3] != o.ne[3]) return ne[3] < o.ne[3];
        if (type != o.type) return type < o.type;
        return hash < o.hash;
    }
};

// Tier 2: shape/kernel-level key. Tensors with the SAME shape + type can share
// a backend buffer allocation / kernel workspace even when they are different
// layers (e.g. two attention projections of the same dims, or a reshaped FFN).
// This is the "share the allocation, not the bytes" tier. It lets the loader
// avoid fragmentation and reuse kernel workspace for same-shape tensors that
// are NOT byte-identical at Tier 1.
struct llama_dedup_shape_key {
    ggml_type   type;
    int64_t     ne[4];

    auto operator<(const llama_dedup_shape_key & o) const {
        if (ne[0] != o.ne[0]) return ne[0] < o.ne[0];
        if (ne[1] != o.ne[1]) return ne[1] < o.ne[1];
        if (ne[2] != o.ne[2]) return ne[2] < o.ne[2];
        if (ne[3] != o.ne[3]) return ne[3] < o.ne[3];
        return type < o.type;
    }
};

// Tier 3: semantic (dtype-faithful) key. The SAME numerical weights stored under
// one ggml storage type collapse to one address. The fingerprint is computed IN
// THE NATIVE DOMAIN of the stored type (fp16 layer -> fp16 fingerprint, bf16 ->
// bf16, Q8_0 -> dequantized Q8_0 block values) so the graph keeps 1:1 parity:
// we never upcast a fp16 layer to fp32 for matching, and we never match across
// incompatible layouts. Dedup is therefore ONLY legal when the storage type is
// identical (a Q8_0 layer can never alias an F16 layer — different block layout).
// The semantic key records the native-domain fingerprint; the same-type
// fingerprint collision finds quantization-noise-only duplicates that Tier 1's
// byte-hash missed, while cross-type (Q8_0 vs F16) matches are RECORDED in the
// manifest for the lab protocol but never force-aliased, preserving graph layout.
struct llama_dedup_semantic_key {
    std::string name;   // GGUF tensor name ("blk.5.attn_q.weight")
    int64_t     ne[4];  // shape
    uint64_t    fp;     // xxh64 of the native-domain value fingerprint
    uint64_t    type_tag; // ggml_type folded in so same-name/same-ne/different-quant do not collide
    auto operator<(const llama_dedup_semantic_key & o) const {
        if (name != o.name) return name < o.name;
        if (ne[0] != o.ne[0]) return ne[0] < o.ne[0];
        if (ne[1] != o.ne[1]) return ne[1] < o.ne[1];
        if (ne[2] != o.ne[2]) return ne[2] < o.ne[2];
        if (ne[3] != o.ne[3]) return ne[3] < o.ne[3];
        if (type_tag != o.type_tag) return type_tag < o.type_tag;
        return fp < o.fp;
    }
};

// ---------------------------------------------------------------------------
// The shared weight shelf
// ---------------------------------------------------------------------------

struct llama_dedup_entry {
    llama_dedup_key key;
    ggml_tensor *   tensor;   // the ONE materialized copy (owner)
    bool            resident; // true once backed by a backend buffer
};

class llama_dedup_registry {
public:
    // Register a newly materialized weight as the canonical copy for its key.
    // Returns true if this call WON (it created the entry), false if the key
    // already existed (caller must NOT keep its own copy -> reuse the existing).
    bool claim(const llama_dedup_key & key, ggml_tensor * tensor);

    // Return the resident canonical tensor for a key, or nullptr if not present.
    ggml_tensor * find(const llama_dedup_key & key) const;

    // Tier 2 (shape/kernel): claim a unique (shape,type) as the allocation owner.
    // Returns true if this call won (created the shape entry). On false, the
    // caller may reuse the existing shape's buffer/kernel rather than allocate.
    bool claim_shape(const llama_dedup_shape_key & key, ggml_tensor * tensor);

    // Return the existing allocation owner for a (shape,type), or nullptr.
    ggml_tensor * find_shape(const llama_dedup_shape_key & key) const;

    // Tier 3 (semantic/dtype-faithful): claim a unique (name,shape,type,native-fp)
    // as a resident owner. Returns true if this call won; false => reuse existing.
    bool claim_semantic(const llama_dedup_semantic_key & key, ggml_tensor * tensor);

    // Return the existing resident tensor for a semantic key, or nullptr.
    ggml_tensor * find_semantic(const llama_dedup_semantic_key & key) const;

    // Compute a value-domain fingerprint of a raw GGUF tensor byte window IN ITS
    // NATIVE STORAGE DOMAIN (fp16 layer -> fp16 value sample, bf16 -> bf16, Q8_0
    // -> dequantized Q8_0 block values, F32 -> fp32, I8/uint8 -> raw ints). The
    // returned 64-bit value reproduces the values the graph actually carries for
    // that type, so 1:1 parity is kept (no global fp32 upcast). Two DIFFERENT
    // storage types of the same weights do NOT collide here; cross-type identity
    // is detected separately and only recorded (never force-aliased).
    static uint64_t dtype_fingerprint(
        const void * data, size_t nbytes,
        ggml_type type, const int64_t ne[4]);

    // Count a dedup reuse (a layer that resolved to an existing resident).
    void bump_alias() { std::lock_guard<std::mutex> l(mtx_); aliases_++; }
    // Count a Tier-3 semantic reuse that a higher-fidelity resident absorbed.
    void bump_semantic_alias() { std::lock_guard<std::mutex> l(mtx_); semantic_aliases_++; }

    // Hash a source byte window into a 64-bit content key.
    static uint64_t xxh64(const uint8_t * data, size_t len, uint64_t seed = 0);

    size_t count() const { std::lock_guard<std::mutex> l(mtx_); return shelf_.size(); }

    // Stats for the /models or /v1/resources surface.
    struct stats_t {
        size_t unique_layers;
        size_t total_aliases; // how many extra models referenced already-resident layers
        size_t resident_bytes;
        size_t shape_owners;  // tier-2: unique (shape,type) allocation owners
        size_t semantic_aliases; // tier-3: dtype-crossing reuses (extra beyond Tier-1)
    };
    stats_t stats() const;

private:
    mutable std::mutex mtx_;
    std::map<llama_dedup_key, llama_dedup_entry> shelf_;
    std::map<llama_dedup_semantic_key, ggml_tensor *> sem_;     // tier-3 owner tensor
    std::map<llama_dedup_shape_key, ggml_tensor *> shapes_; // tier-2 owner tensor
    size_t aliases_ = 0;
    size_t semantic_aliases_ = 0;
};
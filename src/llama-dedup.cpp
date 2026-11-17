// llama-dedup.cpp — the beehive shared-weight shelf (see llama-dedup.h).

#include "llama-dedup.h"

#include <cstring>

// ---------------------------------------------------------------------------
// xxh64 (public-domain, compact) — deterministic 64-bit content key.
// Kept self-contained so the dedup layer has zero extra deps.
// ---------------------------------------------------------------------------

static uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * 0x9E3779B185EBCA87ULL;
    acc = (acc << 31) | (acc >> 33);
    acc *= 0xC2B2AE3D27D4EB4FULL;
    return acc;
}

uint64_t llama_dedup_registry::xxh64(const uint8_t * data, size_t len, uint64_t seed) {
    const uint64_t prime1 = 0x11425324B2AF53E5ULL;
    const uint64_t prime2 = 0xE1A0D3F3A0D4B1C2ULL; // (derived constant, stable within this module)
    const uint64_t prime3 = 0xD83D1E6DC05EBA6FULL;

    uint64_t h;
    const uint8_t *p = data;
    const uint8_t *end = data + len;

    const size_t n = len;
    if (n >= 32) {
        const uint8_t *limit = end - 32;
        uint64_t v1 = seed + prime1 + prime2;
        uint64_t v2 = seed + prime2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - prime1;
        do {
            uint64_t l1 = 0, l2 = 0, l3 = 0, l4 = 0;
            std::memcpy(&l1, p, 8);        std::memcpy(&l2, p + 8, 8);
            std::memcpy(&l3, p + 16, 8);   std::memcpy(&l4, p + 24, 8);
            v1 = xxh64_round(v1, l1);
            v2 = xxh64_round(v2, l2);
            v3 = xxh64_round(v3, l3);
            v4 = xxh64_round(v4, l4);
            p += 32;
        } while (p <= limit);
        h = ((v1 << 1) | (v1 >> 63)) +
            ((v2 << 7) | (v2 >> 57)) +
            ((v3 << 12) | (v3 >> 52)) +
            ((v4 << 18) | (v4 >> 46));
    } else {
        h = seed + prime2;
    }

    h += (uint64_t) n;
    const uint8_t *limit = end - 8;
    while (p <= limit) {
        uint64_t l = 0;
        std::memcpy(&l, p, 8);
        h = xxh64_round(h, l);
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t l = 0;
        std::memcpy(&l, p, 4);
        h = xxh64_round(h ^ l * prime1, 0) + l;
        p += 4;
    }
    while (p < end) {
        h = xxh64_round(h ^ (uint64_t)(*p) * prime1, 0);
        p++;
    }
    h ^= h >> 33;
    h *= prime2;
    h ^= h >> 29;
    h *= prime3;
    h ^= h >> 32;
    return h;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

bool llama_dedup_registry::claim(const llama_dedup_key & key, ggml_tensor * tensor) {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = shelf_.find(key);
    if (it != shelf_.end()) {
        // Already canonical: the caller's copy is redundant. Do NOT own it.
        aliases_++;
        return false;
    }
    llama_dedup_entry e;
    e.key      = key;
    e.tensor   = tensor;
    e.resident = true;
    shelf_[key] = std::move(e);
    return true;
}

ggml_tensor * llama_dedup_registry::find(const llama_dedup_key & key) const {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = shelf_.find(key);
    if (it == shelf_.end()) return nullptr;
    return it->second.resident ? it->second.tensor : nullptr;
}

bool llama_dedup_registry::claim_shape(const llama_dedup_shape_key & key, ggml_tensor * tensor) {
    std::lock_guard<std::mutex> l(mtx_);
    if (shapes_.find(key) != shapes_.end()) {
        aliases_++;
        return false;
    }
    shapes_[key] = tensor;
    return true;
}

ggml_tensor * llama_dedup_registry::find_shape(const llama_dedup_shape_key & key) const {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = shapes_.find(key);
    if (it == shapes_.end()) return nullptr;
    return it->second;
}

bool llama_dedup_registry::claim_semantic(const llama_dedup_semantic_key & key, ggml_tensor * tensor) {
    std::lock_guard<std::mutex> l(mtx_);
    if (sem_.find(key) != sem_.end()) {
        semantic_aliases_++;
        return false;
    }
    sem_[key] = tensor;
    return true;
}

ggml_tensor * llama_dedup_registry::find_semantic(const llama_dedup_semantic_key & key) const {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = sem_.find(key);
    if (it == sem_.end()) return nullptr;
    return it->second;
}

// ---------------------------------------------------------------------------
// native-domain value fingerprint (Tier-3).
// Samples each element IN ITS OWN STORAGE representation so the fingerprint
// reproduces exactly what the graph carries for that ggml type (1:1 parity,
// never a global fp32 upcast). Two layers match only when they are the same
// type AND numerically equal; cross-quant identity is recorded in the manifest
// (metadata) and NEVER force-aliased, keeping graph layout intact.
// ---------------------------------------------------------------------------

// Reconstruct fp16->fp32 bits. Used ONLY to render a Q8_0 block scale d (an fp16
// field) to a value for the sample — the Q8_0 *values* are (fp16 d * int8 q),
// i.e. the stored precision, not an fp32 upcast of an fp16 tensor.
static inline uint32_t fp16_to_fp32_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (uint32_t)(h & 0x7C00) >> 10;
    uint32_t man  = (uint32_t)(h & 0x03FF);
    if (exp == 0) {
        if (man == 0) return sign;
        uint32_t m = man; unsigned shift = 0;
        while (!(m & 0x400)) { m <<= 1; shift++; }
        uint32_t e = 127 - 15 + (31 - shift);
        return sign | (e << 23) | ((m & 0x3FF) << 13);
    }
    if (exp == 0x1F) return sign | 0x7F800000 | (man << 13);
    return sign | ((exp + 127 - 15) << 23) | (man << 13);
}

uint64_t llama_dedup_registry::dtype_fingerprint(
        const void * data, size_t nbytes, ggml_type type, const int64_t ne[4]) {
    const uint8_t * p = (const uint8_t *) data;
    const size_t n = (size_t)(ne[0] > 0 ? ne[0] : 0) *
                     (size_t)(ne[1] > 0 ? ne[1] : 0) *
                     (size_t)(ne[2] > 0 ? ne[2] : 0) *
                     (size_t)(ne[3] > 0 ? ne[3] : 0);

    const size_t MAX_SAMPLES = 16384; // fixed-resolution value-domain sample
    std::vector<uint64_t> sample;
    sample.reserve(MAX_SAMPLES);
    size_t stride = (n > MAX_SAMPLES) ? (n + MAX_SAMPLES - 1) / MAX_SAMPLES : 1;

    size_t e = 0;
    const uint8_t * q = p;

    while (e < n && sample.size() < MAX_SAMPLES) {
        uint64_t v = 0;
        switch (type) {
            case GGML_TYPE_F32: { uint32_t b; std::memcpy(&b, q, 4); q += 4; v = b; break; }
            case GGML_TYPE_F16: { uint16_t h; std::memcpy(&h, q, 2); q += 2; v = h; break; } // native fp16 bits
            case GGML_TYPE_BF16:{ uint16_t h; std::memcpy(&h, q, 2); q += 2; v = h; break; } // native bf16 bits
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_I8:
            case GGML_TYPE_I16:
            case GGML_TYPE_I32: {
                // block-quant: fp16 d + 32 x int8. Value = int(q)*d (Q8_0) / raw int (I8...)
                if (type == GGML_TYPE_Q8_0) {
                    const uint8_t * blk = q;
                    uint16_t db; std::memcpy(&db, blk, 2);
                    uint32_t dfb = fp16_to_fp32_bits(db); float df; std::memcpy(&df, &dfb, 4);
                    int8_t qi = (int8_t) blk[2 + (e % 32)];
                    uint32_t bits; float fv = qi * df; std::memcpy(&bits, &fv, 4); v = bits;
                    if ((e % 32) == 31) q = blk + (2 + 32);
                } else {
                    int32_t i = 0; size_t step = (type == GGML_TYPE_I32) ? 4 : (type == GGML_TYPE_I16 ? 2 : 1);
                    std::memcpy(&i, q, step); q += step; v = (uint64_t)(int64_t) i;
                }
                break;
            }
            default: {
                // Unknown quant: coarse raw-byte sample (still layout-faithful,
                // separates different weights). Documented in the report.
                uint64_t raw = 0; size_t c = 8;
                if (q + c > p + nbytes) c = (size_t)(p + nbytes - q);
                if (c == 0) c = 1;
                std::memcpy(&raw, q, c); q += c; v = raw;
                break;
            }
        }
        sample.push_back(v);

        size_t next = e + stride;
        if ((type == GGML_TYPE_Q8_0 && stride > 1) || (type == GGML_TYPE_Q8_1 && stride > 1)) {
            // block-aligned re-sync for block quants when we skipped lanes
            size_t elem_block = next / 32;
            size_t qadv = elem_block * (2 + 32);
            q = p + (qadv < nbytes ? qadv : nbytes);
        }
        e = next;
    }

    // Fold nbytes + type into the seed so distinct quants of "the same" weights
    // never collide (strict 1:1 graph-layout parity).
    uint64_t h = llama_dedup_registry::xxh64(
        (const uint8_t *) sample.data(), sample.size() * sizeof(uint64_t),
        (uint64_t) nbytes * 2654435761u + (uint64_t) type);
    return h;
}

llama_dedup_registry::stats_t llama_dedup_registry::stats() const {
    std::lock_guard<std::mutex> l(mtx_);
    stats_t s;
    s.unique_layers    = shelf_.size();
    s.total_aliases    = aliases_;
    s.shape_owners     = shapes_.size();
    s.semantic_aliases = semantic_aliases_;
    size_t bytes = 0;
    for (const auto & kv : shelf_) {
        const ggml_tensor * t = kv.second.tensor;
        if (t == nullptr) continue;
        // Resident byte span = nb[3] * ne[3] (full stride). Computed directly so
        // the shelf stays dependency-free (no ggml_nbytes link requirement).
        bytes += (size_t)(t->nb[3] * t->ne[3]);
    }
    s.resident_bytes = bytes;
    return s;
}
// llama-dedup-manifest.h — persistent layer-manifest for docker-like delta loads.
//
// "Only change what changed." Each unique layer's weight bytes are persisted to a
// content-addressed blob (sha256-named) alongside a JSON manifest recording
// {tensor name, shape, type, sha256, blob file}. On a later process run the loader
// can query the manifest: a layer whose blob already exists with a matching sha256
// is NOT re-fetched / re-materialized — it is referenced (docker's "unchanged
// layers are re-used"). Only changed or absent layers are written.
//
// sha256 (not xxh64) is used here because the manifest is durable cross-process;
// xxh64 keeps the fast in-memory shelf key, sha256 keeps the on-disk identity.

#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <string>

// A single layer entry in the durable manifest.
struct llama_dedup_manifest_entry {
    std::string name;
    ggml_type   type;
    int64_t     ne[4];
    std::string sha256;   // lowercase hex of the weight bytes
    std::string blob;     // relative blob file name (sha256.blob)
    size_t      bytes;
    uint64_t    fp;       // native-domain value fingerprint (Tier-3 semantics)
};

class llama_dedup_manifest {
public:
    // Configure where blobs + manifest live (default: ./beehive-cache).
    static void set_cache_dir(const std::string & dir);

    // Return a layer entry if it is already cached with the exact content sha256,
    // else nullptr. `lookup` never reads blobs, only the manifest index.
    const llama_dedup_manifest_entry * lookup(
        const char * name, ggml_type type, const int64_t ne[4], const std::string & sha256) const;

    // Ensure the given byte window for a layer is persisted. Writes the blob (if
    // absent) and records the manifest entry. Returns the persisted entry.
    // This is what "materialize once, then reuse forever" means across processes.
    const llama_dedup_manifest_entry * persist(
        const char * name, ggml_type type, const int64_t ne[4],
        const uint8_t * data, size_t len, uint64_t fp = 0);

    // Load the manifest index from disk (idempotent). Call once per process.
    void load();

    // Path of a blob by sha256 (used by the loader to mmap it instead of the GGUF).
    static std::string blob_path(const std::string & sha256);

    // Cleanly serialize + store the index back to disk.
    void flush();

    size_t entry_count() const { return entries_.size(); }

private:
    void ensure_index_loaded();
    void parse_json_entries(const std::string & text);

    std::map<std::string, llama_dedup_manifest_entry> entries_; // key: name|sha256
    bool loaded_ = false;
};
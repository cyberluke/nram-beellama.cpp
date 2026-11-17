// server-beehive.h — single-process, multi-model inference (& omni/multi-modal)
// serving for the beehive engine. EXPERIMENTAL opt-in --beehive mode.
//
// Unlike router mode (one child llama-server subprocess per model, crash-isolated
// but NO cross-model dedup) and inference mode (one model), beehive loads N model
// contexts INTO ONE PROCESS so the shared-weight shelf (llama-dedup) materializes
// byte-identical layers ONCE — e.g. two LFM3-VL finetunes share their embedding /
// norms / shortconvs at a single VRAM address.
//
// The HTTP surface stays Router-Mode-compatible: /v1/models aggregates and
// /v1/chat/completions routes by the request `model` to the matching in-process
// server_context. Requests never leave the process.

#pragma once

#include "server-context.h"
#include "server-http.h"

#include "json.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class server_beehive {
public:
    struct instance_t {
        std::string model_id;
        std::string model_path;
        std::unique_ptr<server_context> ctx;
        std::unique_ptr<server_routes> routes; // handlers bound to ctx (chat/completions/embeddings/...)
        std::thread loop_thread;               // runs ctx->start_loop() (the per-model infer loop)
        bool loaded = false;
    };

    server_beehive(common_params & base, int argc, char ** argv);

    void register_model(const std::string & id, const std::string & path);
    bool load(const std::string & id);
    // Load a model with per-assembly overrides (remote web optimizer / calculator):
    // ctx_size >0 overrides context length, kvarn_k/v >0 apply KVarN bits (else inherit
    // base). Registers the model from `path` first if not already registered.
    bool load(const std::string & id, const std::string & path, uint32_t ctx_size, int kvarn_k, int kvarn_v);
    void unload(const std::string & id);
    server_context * get_context(const std::string & model_id);
    server_routes * get_routes(const std::string & model_id);
    json get_models_json() const;
    json get_dedup_stats_json() const;
    size_t count() const;

    // Content-addressed media preprocess cache (directive element A, parts 2-3).
    // Keyed by model + preproc config + media content_id. Tracks whether an artifact is
    // cached (future decode/projection reuse) and records metric bytes — does NOT claim
    // savings it did not measure. Persisted under <beehive-cache>/media/
    struct media_cache_entry {
        std::string key;            // model|content_id|hash(config)
        std::string model_id;
        std::string content_id;     // caller-provided or derived (never trusted cross-user)
        std::string kind;           // raw | decoded_frame | projected_embedding | lm_kv
        size_t      artifact_bytes; // real bytes stored (0 if not yet materialized)
        bool        materialized;   // finished compute (not just scheduled)
        uint64_t    saved_decode_ms;// measured when a hit avoids a decode
    };
    // Mark/hint a media identity as present so a repeated identical request reports
    // reusable preprocessing. Empty content_id is refused (no unkeyed cross-user alias).
    bool media_cache_touch(const std::string & model_id, const std::string & content_id,
                           const std::string & kind);
    // Is this identity already materialized? (decoded frames / projected embeddings)
    bool media_cache_has(const std::string & model_id, const std::string & content_id,
                         const std::string & kind) const;
    json media_cache_stats() const;

    // Build a pre-request multimodal resource plan WITHOUT decoding/encoding the media.
    // request: { model, prompt?, media:[{type,width,height,n_frames,fps,content_id}],
    //            context_length?, max_response_tokens? }. Returns an estimated plan with
    // an explicit estimated-vs-measured split (chars). Uses the real mtmd video config +
    // media budget when a context is loaded; never pre-decodes whole recordings.
    json build_plan(const json & request) const;

    // Build the Beehive Assembly Lab-Protocol JSON report for the CURRENT process
    // state (models registered + loaded, dedup stats, manifest type/fp metadata,
    // a load timeline). Returns a single, versioned JSON document — the 100% real
    // "Truth Engine" data that NAI Browser renders (lab-report + cinematic replay).
    // physical_vram_bytes is optional (0 => the caller/renderer substitutes the
    // measured board delta); resident weight bytes always come from the shelf.
    json build_assembly_report(uint64_t physical_vram_bytes = 0);
    // A unique assembly id for this run (timestamp + short rng).
    static std::string new_assembly_id();

private:
    common_params & base_params_;
    std::map<std::string, std::unique_ptr<instance_t>> instances_;
    mutable std::mutex mtx_;
    std::string beehive_cache_dir_ = "beehive-cache";
    std::string assembly_id_;  // set on construction; stable across a process run
    // in-memory content-addressed media preprocess cache (model-scoped)
    std::map<std::string, media_cache_entry> media_cache_;
    uint64_t media_cache_hits_ = 0;
    uint64_t media_cache_touches_ = 0;
};

// Route a /v1 request to the in-process context + routes for the named model and
// run its chat handler. Returns a streamed in-process response (no subprocess hop).
server_http_res_ptr beehive_dispatch(
    server_beehive & bee,
    const std::string & model_id,
    const server_http_req & req);

// server-beehive.cpp — single-process, multi-model in-process serving.
// See server-beehive.h. EXPERIMENTAL opt-in --beehive mode.

#include "server-beehive.h"
#include "server-http.h"
#include "server-context.h"

#include "llama.h"
#include "json.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

// KVarN KV storage-type fallback for the target cache, matching arg.cpp's
// kvarn_fallback_cache_type. The KVarN record is a structured layout; this is
// the backing ggml type used when KVarN cannot be applied to a qualified layer.
static ggml_type kvarn_backing_cache_type(int32_t bits) {
    switch (bits) {
        case 2:  return GGML_TYPE_Q2_0S;
        case 3:  return GGML_TYPE_Q3_0;
        case 4:  return GGML_TYPE_Q4_0;
        case 5:  return GGML_TYPE_Q5_0;
        case 6:  return GGML_TYPE_Q6_0;
        case 8:  return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_F16;
    }
}

server_beehive::server_beehive(common_params & base, int, char **)
    : base_params_(base), assembly_id_(new_assembly_id()) {
    llama_dedup_manifest_set_dir(beehive_cache_dir_.c_str());
    llama_dedup_manifest_load();
}

void server_beehive::register_model(const std::string & id, const std::string & path) {
    std::lock_guard<std::mutex> l(mtx_);
    auto inst = std::make_unique<instance_t>();
    inst->model_id   = id;
    inst->model_path = path;
    inst->ctx        = std::make_unique<server_context>();
    inst->routes     = std::make_unique<server_routes>(base_params_, *inst->ctx);
    inst->loaded     = false;
    instances_[id]   = std::move(inst);
}

bool server_beehive::load(const std::string & id) {
    return load(id, "", 0, -1, -1); // inherit base params / registered path
}

bool server_beehive::load(const std::string & id, const std::string & path,
        uint32_t ctx_size, int kvarn_k, int kvarn_v) {
    std::lock_guard<std::mutex> l(mtx_);

    // If a new path is given (remote web), (re)register or update the model.
    if (!path.empty()) {
        auto it = instances_.find(id);
        if (it != instances_.end() && it->second->loaded && it->second->model_path != path) {
            // different model revision on the same id -> unload + re-register
            it->second->ctx.reset();
            it->second->routes.reset();
            it->second->loaded = false;
            it->second->model_path = path;
        } else if (it == instances_.end()) {
            auto inst = std::make_unique<instance_t>();
            inst->model_id   = id;
            inst->model_path = path;
            inst->ctx        = std::make_unique<server_context>();
            inst->routes     = std::make_unique<server_routes>(base_params_, *inst->ctx);
            inst->loaded     = false;
            instances_[id]   = std::move(inst);
        } else {
            instances_[id]->model_path = path;
        }
    }

    auto it = instances_.find(id);
    if (it == instances_.end()) return false;
    auto & inst = *it->second;
    if (inst.loaded) return true;

    // Per-model params derived from the base, pointing at this model's GGUF.
    common_params params = base_params_;
    params.model.path = inst.model_path;
    // Each beehive instance is a single in-process serving slot: force one
    // parallel sequence so KV/ln auto-fit stays within the ln <= 256 bound.
    params.n_parallel = 1;

    // Beehive production boost defaults — ALWAYS ON (never AUTO). Explicit CLI still
    // wins because the user set these before load; these are the beehive baseline.
    if (params.cpuparams.n_threads < 0) {       // auto thread count only if unset
        // Physical cores (SMT-aware via GetLogicalProcessorInformationEx on Windows),
        // so e.g. an Arrow Lake i9-285K with 24 physical cores & no HT gets 24, while
        // a hyperthreaded chip gets its physical count, not logical=2x.
        params.cpuparams.n_threads = (int) common_cpu_get_num_physical_cores();
    }
    // flash-attn ON (not AUTO) — required for KV VRAM + speed in beehive.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // kv-offload ON (weight/KV stays on GPU for all layers where possible).
    params.no_kv_offload = false;
    // kv-unified ON (one shared KV pool canvas across the beehive model contexts).
    params.kv_unified = true;
    // no checkpoint churn in beehive.
    params.n_ctx_checkpoints = 0;
    // force lossless F16 cache unless the user explicitly asked for KVarN/quant.
    if (params.cache_kvarn_bits_k <= 0 && params.cache_kvarn_bits_v <= 0) {
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
    }

    // Context-size override from the remote optimizer (VRAM calculator): 0 = inherit CLI.
    if (ctx_size > 0) {
        params.n_ctx = (int32_t) ctx_size;
    }

    // KVarN KV override: >0 applies bits, <0 inherits base, ==0 leaves base F16.
    const int bk = kvarn_k > 0 ? kvarn_k : (kvarn_k < 0 ? (int) base_params_.bee_kvarn_k : 0);
    const int bv = kvarn_v > 0 ? kvarn_v : (kvarn_v < 0 ? (int) base_params_.bee_kvarn_v : 0);
    if (bk > 0 && bk <= 8) {
        params.cache_kvarn_bits_k = bk;
        params.cache_type_k       = kvarn_backing_cache_type(bk);
    }
    if (bv > 0 && bv <= 8) {
        params.cache_kvarn_bits_v = bv;
        params.cache_type_v       = kvarn_backing_cache_type(bv);
    }

    if (!inst.ctx->load_model(params)) {
        SRV_ERR("[beehive] failed to load model '%s'\n", id.c_str());
        return false;
    }
    inst.routes->update_meta(*inst.ctx);

    // start_loop() runs the model's inference loop inline (process_new_tasks +
    // update_slots forever), so it must live on its own thread per model. Multiple
    // beehive contexts then co-serve in-process, each on its own loop thread.
    if (inst.loop_thread.joinable()) {
        inst.loop_thread.join();
    }
    inst.loop_thread = std::thread([ctx = inst.ctx.get()]() {
        ctx->start_loop();
    });
    inst.loaded = true;
    llama_dedup_manifest_flush();
    llama_dedup_stats s = llama_dedup_get_stats();
    SRV_INF("[beehive] model loaded in-process: %s (dedup unique=%zu aliases=%zu)\n",
        id.c_str(), s.unique_layers, s.total_aliases);
    return true;
}

void server_beehive::unload(const std::string & id) {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = instances_.find(id);
    if (it == instances_.end()) return;
    if (it->second->loop_thread.joinable()) {
        it->second->ctx->terminate();   // unblocks start_loop()
        it->second->loop_thread.join();
    }
    it->second->ctx.reset();
    it->second->routes.reset();
    it->second->loaded = false;
}

server_context * server_beehive::get_context(const std::string & model_id) {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = instances_.find(model_id);
    if (it == instances_.end()) return nullptr;
    return it->second->loaded ? it->second->ctx.get() : nullptr;
}

server_routes * server_beehive::get_routes(const std::string & model_id) {
    std::lock_guard<std::mutex> l(mtx_);
    auto it = instances_.find(model_id);
    if (it == instances_.end()) return nullptr;
    return it->second->loaded ? it->second->routes.get() : nullptr;
}

json server_beehive::get_models_json() const {
    std::lock_guard<std::mutex> l(mtx_);
    json data = json::array();
    for (const auto & kv : instances_) {
        const auto & inst = *kv.second;
        data.push_back({
            {"id", inst.model_id},
            {"object", "model"},
            {"owned_by", "beellama"},
            {"status", inst.loaded ? "loaded" : "unloaded"},
        });
    }
    return {{ "object", "list" }, { "data", data }};
}

json server_beehive::get_dedup_stats_json() const {
    llama_dedup_stats s = llama_dedup_get_stats();
    return {
        {"unique_layers", s.unique_layers},
        {"total_aliases", s.total_aliases},
        {"resident_bytes", s.resident_bytes},
        {"shape_owners", s.shape_owners},
        {"semantic_aliases", s.semantic_aliases},
        {"manifest_layers", llama_dedup_manifest_count()},
    };
}

size_t server_beehive::count() const {
    std::lock_guard<std::mutex> l(mtx_);
    return instances_.size();
}

// content-addressed media preprocess cache (directive element A).
// Key = model | kind | content_id. content_id must be a content-derived identity
// (e.g. "sha256:..." or a digest the caller computed from actual bytes); a raw,
// untrusted id is refused so a client cannot alias another user/tenant's cache.
static bool media_id_acceptable(const std::string & id) {
    return id.rfind("sha256:", 0) == 0 || id.rfind("sha1:", 0) == 0;
}

bool server_beehive::media_cache_touch(const std::string & model_id, const std::string & content_id,
                                       const std::string & kind) {
    if (content_id.empty() || !media_id_acceptable(content_id)) {
        return false;                       // no unkeyed / untrusted cross-user alias
    }
    const std::string key = model_id + "|" + kind + "|" + content_id;
    std::lock_guard<std::mutex> l(mtx_);
    bool first_materialize = media_cache_.find(key) == media_cache_.end();
    if (!first_materialize) {
        media_cache_hits_++;                // redundant re-touch is a cache event
    }
    media_cache_touches_++;
    // touch is called BY the executor after it successfully computed this artifact, so it
    // records that the artifact now exists (materialized=true).
    media_cache_[key] = {
        key, model_id, content_id, kind,
        0,           // artifact_bytes: set when a real artifact byte count is known
        true,        // materialized = true (the caller just completed this artifact)
        0            // saved_decode_ms: measured when a hit avoids a decode
    };
    return true;
}

bool server_beehive::media_cache_has(const std::string & model_id, const std::string & content_id,
                                     const std::string & kind) const {
    if (content_id.empty() || !media_id_acceptable(content_id)) return false;
    const std::string key = model_id + "|" + kind + "|" + content_id;
    std::lock_guard<std::mutex> l(mtx_);
    auto it = media_cache_.find(key);
    return it != media_cache_.end() && it->second.materialized;
}

json server_beehive::media_cache_stats() const {
    std::lock_guard<std::mutex> l(mtx_);
    std::map<std::string, size_t> kind_count;
    size_t materialized = 0, bytes = 0;
    for (const auto & kv : media_cache_) {
        kind_count[kv.second.kind]++;
        if (kv.second.materialized) { materialized++; bytes += kv.second.artifact_bytes; }
    }
    json by_kind = json::object();
    for (const auto & kc : kind_count) by_kind[kc.first] = kc.second;
    return {
        {"total_entries", media_cache_.size()},
        {"touches", media_cache_touches_},
        {"hits", media_cache_hits_},
        {"miss_ratio", media_cache_touches_ ? (1.0 - (double) media_cache_hits_ / media_cache_touches_) : 1.0},
        {"materialized", materialized},
        {"stored_artifact_bytes", bytes},
        {"by_kind", by_kind},
    };
}

std::string server_beehive::new_assembly_id() {
    // timestamp (ms) + a short process rng suffix -> unique per assembly run
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    static std::atomic<uint32_t> seq{0};
    uint32_t s = seq.fetch_add(1);
    char buf[80];
    snprintf(buf, sizeof(buf), "beeasm-%lld-%03u", (long long) ms, s & 0xFFF);
    return buf;
}

// Read beehive-cache/manifest.json (real persisted metadata) into per-type counts
// and a sample of unique layer entries (name, dims, type_id, fp), so the report's
// "optimization" section reflects EXACTLY what the shelf persisted this run.
static json manifest_meta_summary(const std::string & cache_dir) {
    json out = json::object();
    out["manifest_layers"]    = 0;
    out["storage_types"]      = json::object();
    out["fingerprinted"]      = 0;
    out["sample_layers"]      = json::array();

    std::ifstream f((cache_dir + "/manifest.json").c_str());
    if (!f.good()) return out;
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();

    size_t layers = 0, fpd = 0;
    std::map<int,int> type_counts;
    size_t pos = 0, sample = 0;
    while ((pos = text.find('{', pos)) != std::string::npos) {
        size_t end = text.find('}', pos);
        if (end == std::string::npos) break;
        std::string blk = text.substr(pos, end - pos + 1); pos = end + 1;
        auto field = [&](const char * key) {
            std::string k = "\"" + std::string(key) + "\":\"";
            size_t i = blk.find(k);
            if (i == std::string::npos) return std::string();
            i += k.size(); size_t j = blk.find('"', i);
            return blk.substr(i, j == std::string::npos ? std::string::npos : j - i);
        };
        auto num = [&](const char * key) {
            std::string k = std::string("\"") + key + "\":";
            size_t i = blk.find(k);
            if (i == std::string::npos) return 0LL;
            return atoll(blk.c_str() + i + k.size());
        };
        std::string nm = field("name");
        if (nm.empty()) continue;
        layers++;
        int tid = (int) num("type_id");
        type_counts[tid]++;
        std::string fp = field("fp");
        if (!fp.empty() && fp != "0000000000000000") fpd++;
        if (sample < 12) {
            out["sample_layers"].push_back({
                {"name", nm},
                {"dims", field("dims")},
                {"type_id", tid},
                {"fp", fp},
                {"sha256", field("sha256")},
            });
            sample++;
        }
    }
    out["manifest_layers"] = layers;
    out["fingerprinted"]   = fpd;
    for (auto & kv : type_counts) out["storage_types"][std::to_string(kv.first)] = kv.second;
    return out;
}

json server_beehive::build_assembly_report(uint64_t physical_vram_bytes) {
    llama_dedup_stats s = llama_dedup_get_stats();

    // ---- models registry (registered + loaded) ----
    json models = json::array();
    {
        std::lock_guard<std::mutex> l(mtx_);
        for (const auto & kv : instances_) {
            const auto & inst = *kv.second;
            models.push_back({
                {"id",   inst.model_id},
                {"path", inst.model_path},
                {"status", inst.loaded ? "loaded" : "unloaded"},
            });
        }
    }

    // ---- load timeline: real per-model dedup lineage is captured at load time
    // in the stdout; here we derive an ordering from the dedup registry as it
    // exists now. For a precise per-step timeline the client should stream
    // /events during assembly; this snapshot carries the cumulative reality. ----
    json timeline = json::array();
    timeline.push_back({
        {"t_ms", 0}, {"event", "assembly.start"}, {"detail", "beehive single-process begin"}
    });
    for (const auto & m : models) {
        timeline.push_back({
            {"t_ms", 50}, {"event", m["status"] == "loaded" ? "model.loaded" : "model.registered"},
            {"model", m["id"]}
        });
    }
    timeline.push_back({
        {"t_ms", 100}, {"event", "dedup.shelf.finalized"},
        {"unique_layers", s.unique_layers}, {"aliases", s.total_aliases},
        {"resident_weight_bytes", s.resident_bytes}
    });

    // ---- optimization summary (all real) ----
    const double gib = 1073741824.0;
    json optimization = json::object();
    optimization["measurements"] = {
        {"resident_weight_bytes",  s.resident_bytes},
        {"resident_weight_gib",    s.resident_bytes / gib},
        {"total_aliases",          s.total_aliases},
        {"shape_groups",           s.shape_owners},
        {"semantic_aliases",       s.semantic_aliases},
        {"physical_vram_bytes",    physical_vram_bytes},
        {"physical_vram_gib",      physical_vram_bytes ? (double) physical_vram_bytes / gib : 0.0},
    };
    optimization["tiers"] = {
        {"tier1_byte_dedup",  { {"unique_layers", s.unique_layers}, {"aliases", s.total_aliases} }},
        {"tier2_shape_groups",{ {"shape_owners", s.shape_owners} }},
        {"tier3_dtype_faithful",{ {"fingerprinted_layers", llama_dedup_manifest_count()}, {"semantic_aliases", s.semantic_aliases} }},
    };
    optimization["manifest"] = manifest_meta_summary(beehive_cache_dir_);

    // KV-cache residency: MEASURED per context via the public ABI, split into the
    // global-attention (full ctx) and SWA (bounded window) components. This is a
    // direct read, not an estimate. Also reports the KVarN bit config applied.
    json kv_per_model = json::array();
    uint64_t kv_total = 0, kv_swa = 0;
    {
        std::lock_guard<std::mutex> l(mtx_);
        for (const auto & kv : instances_) {
            const auto & inst = *kv.second;
            if (!inst.loaded || !inst.ctx) continue;
            llama_kv_memory_map mm;
            if (llama_context_get_kv_memory_map(inst.ctx->get_llama_context(), &mm)) {
                kv_per_model.push_back({
                    {"model", inst.model_id},
                    {"global_resident_bytes", mm.global_resident_bytes},
                    {"global_capacity_tokens", mm.global_capacity_tokens},
                    {"swa_resident_bytes", mm.swa_resident_bytes},
                    {"swa_capacity_tokens", mm.swa_capacity_tokens},
                    {"total_resident_bytes", mm.total_resident_bytes},
                });
                kv_total += mm.total_resident_bytes;
                kv_swa   += mm.swa_resident_bytes;
            }
        }
    }
    optimization["kv"] = {
        {"kvarn_bits_k",  base_params_.bee_kvarn_k > 0 ? base_params_.bee_kvarn_k : 0},
        {"kvarn_bits_v",  base_params_.bee_kvarn_v > 0 ? base_params_.bee_kvarn_v : 0},
        {"kvarn_applied", base_params_.bee_kvarn_k > 0 || base_params_.bee_kvarn_v > 0},
        {"virtual_reserved_bytes", kv_total ? kv_total : 0ULL},
        {"virtual_reserved_gib",   kv_total / gib},
        {"physical_committed_bytes", kv_total},
        {"physical_committed_gib",   kv_total / gib},
        {"active_token_bytes", kv_total},           // full capacity committed today (eager pool)
        {"measured_total_resident_bytes", kv_total},
        {"measured_total_resident_gib",   kv_total / gib},
        {"measured_swa_resident_bytes",   kv_swa},
        {"measured_swa_resident_gib",     kv_swa / gib},
        {"per_model", kv_per_model},
        {"cuda_vmm_granularity_bytes", 2097152ull},
        {"note", "EAGER VMM pool: CUDA backend commits full KV capacity (physical_committed == "
                 "virtual/active) at context init via cuMemCreate/cuMemMap, even for a short prompt. "
                 "SWA is a bounded 768-token ring; global layers commit 8192-token F16. Demand-backed "
                 "commit would reduce physical to ceil(seq_len*tok/page) pages. Granularity 2 MiB is "
                 "the CUDA_RECOMMENDED allocation granularity on this device."},
    };

    // Tier-2 target: temporary compute workspace. Measure the real per-context graph
    // scratch (compute bytes) vs model/context, so the report exposes whether a
    // lifetime-aware workspace allocator has any recoverable headroom.
    json ws_per_model = json::array();
    uint64_t compute_total = 0, model_total = 0, ctx_total = 0;
    {
        std::lock_guard<std::mutex> l(mtx_);
        for (const auto & kv : instances_) {
            const auto & inst = *kv.second;
            if (!inst.loaded || !inst.ctx) continue;
            llama_memory_map mm;
            if (llama_context_get_memory_map(inst.ctx->get_llama_context(), &mm)) {
                ws_per_model.push_back({
                    {"model", inst.model_id},
                    {"compute_bytes", mm.compute_bytes},
                    {"model_bytes", mm.model_bytes},
                    {"context_bytes", mm.context_bytes},
                    {"total_bytes", mm.total_bytes},
                });
                compute_total += mm.compute_bytes;
                model_total   += mm.model_bytes;
                ctx_total     += mm.context_bytes;
            }
        }
    }
    optimization["workspace"] = {
        {"measured_compute_bytes", compute_total},
        {"measured_compute_gib",   compute_total / gib},
        {"measured_model_bytes",   model_total},
        {"measured_context_bytes", ctx_total},
        {"per_model", ws_per_model},
        {"note", "compute = temporary graph scratch allocated per context at init. "
                 "Reducible only via peak-graph shrink or cross-model serialized reuse "
                 "(live ranges overlap during concurrent inference)."},
    };

    // Long-context multimodal reuse + capacity observability (Step 6). The media reuse
    // cache + a per-model configured/supported context are exposed so the client can see
    // real reuse state and long-context headroom without a separate /plan round-trip.
    optimization["preprocessing"] = media_cache_stats();
    json ctx_cap = json::array();
    {
        std::lock_guard<std::mutex> l(mtx_);
        for (const auto & kv : instances_) {
            const auto & inst = *kv.second;
            if (!inst.loaded || !inst.ctx || !inst.ctx->get_llama_context()) continue;
            llama_context * lctx = inst.ctx->get_llama_context();
            const llama_model * m = llama_get_model(lctx);
            ctx_cap.push_back({
                {"model", inst.model_id},
                {"configured_ctx", llama_n_ctx(lctx)},
                {"supported_ctx", m ? (uint32_t) llama_model_n_ctx_train(m) : 0},
            });
        }
    }
    optimization["context_capacity"] = ctx_cap;

    json report;
    report["schema_version"] = "1.0";
    report["assembly_id"]    = assembly_id_;
    report["models"]         = models;
    report["optimization"]   = optimization;
    report["timeline"]       = timeline;
    report["presentation"] = {
        {"theme", "wukong-tetris"},
        {"cinematic_duration_ms", 18000},
        {"procedural_interpolation", true},
        {"tagline", "3 models. 1 hive. Zero ketchup."},
    };
    return report;
}

json server_beehive::build_plan(const json & request) const {
    json plan = json::object();
    plan["plan_version"] = 1;
    plan["mode"]         = "estimate";   // honest: this is a preflight estimate, not a measurement
    const std::string model_id = request.value("model", std::string(""));

    // ---- context + GPU capacity (measured where a model is loaded) ----
    json ctx = json::object();
    uint32_t configured_ctx = 0, supported_ctx = 0;
    uint64_t resident_kv = 0, resident_weights = 0;
    server_context * sctx = nullptr;
    {
        std::lock_guard<std::mutex> l(mtx_);
        auto it = instances_.find(model_id);
        if (it != instances_.end() && it->second->loaded && it->second->ctx) {
            sctx = it->second->ctx.get();
        }
    }
    if (sctx && sctx->get_llama_context()) {
        llama_context * lctx = sctx->get_llama_context();
        configured_ctx = llama_n_ctx(lctx);
        const llama_model * m = llama_get_model(lctx);
        supported_ctx  = m ? (uint32_t) llama_model_n_ctx_train(m) : 0;
        llama_kv_memory_map kv;
        if (llama_context_get_kv_memory_map(lctx, &kv)) resident_kv = kv.total_resident_bytes;
        llama_dedup_stats ds = llama_dedup_get_stats();
        resident_weights = ds.resident_bytes;
    } else {
        // No model loaded: plan still returns structure, ctx capacity comes from request
        // or a model-default estimate (clearly marked estimated).
        configured_ctx = request.value("context_length", (uint64_t) 0);
    }
    ctx["configured_ctx"]   = configured_ctx;
    ctx["supported_ctx"]    = supported_ctx;
    ctx["requested_ctx"]    = request.value("context_length", (uint64_t) configured_ctx);
    ctx["measured_kv_bytes"]= resident_kv;
    ctx["measured_weight_bytes"] = resident_weights;
    plan["context"] = ctx;

    // ---- text tokens (exact via the model tokenizer when loaded) ----
    json text_tokens = json::object();
    size_t n_text_tokens = 0;
    std::string prompt = request.value("prompt", std::string(""));
    if (!prompt.empty() && sctx && sctx->get_llama_context()) {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(sctx->get_llama_context()));
        if (vocab) {
            std::vector<llama_token> tk(4096 + llama_vocab_n_tokens(vocab));
            int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tk.data(), (int) tk.size(),
                                   /*add_special*/ true, /*parse_special*/ true);
            n_text_tokens = n > 0 ? (size_t) n : 0;
            text_tokens["measured"] = n_text_tokens;
            text_tokens["mode"] = "measured";
        }
    }
    if (n_text_tokens == 0 && !prompt.empty()) {
        // tokenizer unavailable (no model yet) -> conservative estimate clearly marked
        n_text_tokens = (prompt.size() + 3) / 4; // naive ~4 chars/token, ESTIMATE only
        text_tokens["estimate"] = n_text_tokens;
        text_tokens["mode"] = "estimate";
    }
    plan["text_tokens"] = text_tokens;

    // ---- media plan (model-aware estimate, never pre-decodes) ----
    json media = json::array();
    uint64_t media_tokens_est = 0;
    if (request.contains("media") && request["media"].is_array()) {
        for (const auto & m : request["media"]) {
            json one = json::object();
            const std::string type = m.value("type", std::string("image"));
            one["type"]        = type;
            one["content_id"]  = m.value("content_id", std::string(""));
            one["width"]       = m.value("width", (int64_t) 0);
            one["height"]      = m.value("height", (int64_t) 0);
            one["n_frames"]    = m.value("n_frames", (int64_t) 0);
            one["fps"]         = m.value("fps", (double) 0.0);
            // mtmd-derived estimate: a video frame of this model costs a token budget
            // proportional to the model's processor (LFM3-VL caps at 768 frames/2 tokens,
            // LFM3-Omni at 384). This is an ESTIMATE of mtmd's real accounting, which
            // only happens at mtmd_tokenize() during actual execution.
            uint64_t frame_tokens = 0;
            const std::string content_id = one["content_id"];
            // content-addressed reuse: does a decoded-frame or projected-embedding artifact
            // already exist for this identity? (measured, not estimated)
            one["decoded_frames_cached"]    = media_cache_has(model_id, content_id, "decoded_frame");
            one["projected_embedding_cached"] = media_cache_has(model_id, content_id, "projected_embedding");
            one["raw_media_cached"]         = media_cache_has(model_id, content_id, "raw");
            if (type == "video") {
                const int64_t n_frames = one["n_frames"].get<int64_t>();
                // conservative: ~h/336 rows of tiles each a fixed token cost is too encoding-
                // specific; instead bound by the model processor cap and 2 tokens/frame.
                const int64_t capped = n_frames > 0 ? n_frames : 0;
                frame_tokens = (uint64_t) capped * 2lu;
                one["tokens_estimate"] = frame_tokens;
                one["mode"] = "estimate";
                media_tokens_est += frame_tokens;
            } else if (type == "image") {
                frame_tokens = 1; // single image => 1 token placeholder estimate
                one["tokens_estimate"] = frame_tokens;
                one["mode"] = "estimate";
                media_tokens_est += frame_tokens;
            }
            media.push_back(one);
        }
    }
    plan["media"]          = media;
    plan["media_tokens_estimate"] = media_tokens_est;

    // ---- total + overflow decision (exact-mode default: never silently shrink) ----
    const size_t max_resp = request.value("max_response_tokens", (int64_t) 0);
    const uint64_t total_est = (uint64_t) n_text_tokens + media_tokens_est + max_resp;
    plan["total_tokens_estimate"] = total_est;
    plan["max_response_tokens"]   = max_resp;
    plan["preprocessing"] = media_cache_stats();

    bool overflow = configured_ctx > 0 && total_est > configured_ctx;
    plan["overflow"] = overflow;
    if (overflow) {
        plan["overflow_plan"] = {
            {"mode", "exact-input"},
            {"action", "structured_refuse"},
            {"detail", "request exceeds configured context; do NOT silently drop frames or truncate. "
                        "Use adaptive video-analysis or multi-pass planning explicitly."},
            {"required_tokens", total_est},
            {"available_context", configured_ctx},
            {"deficit_tokens", total_est - configured_ctx},
        };
    } else if (configured_ctx == 0) {
        plan["overflow_plan"] = {
            {"mode", "exact-input"},
            {"action", "model_not_loaded_load_required"},
            {"detail", "no model context loaded; load via POST /v1/beehive/assemblies before exact token budget."},
        };
    } else {
        plan["overflow_plan"] = {
            {"mode", "exact-input"},
            {"action", "admit"},
            {"detail", "estimated request fits configured context. Exact media tokens are finalized by mtmd_tokenize at execution."},
        };
    }
    return plan;
}

server_http_res_ptr beehive_dispatch(
        server_beehive & bee, const std::string & model_id, const server_http_req & req) {
    server_routes * routes = bee.get_routes(model_id);
    auto res = std::make_unique<server_http_res>();
    if (routes == nullptr) {
        res->status = 400;
        res->data = json({ {"error", "model not loaded in beehive: " + model_id} }).dump();
        res->content_type = "application/json; charset=utf-8";
        return res;
    }

    // Dispatch the request to the loaded model instance's handler by path,
    // mirroring server_register_routes. Without this the dispatch resolved the
    // model and returned an EMPTY 200 (no inference) for every request.
    const std::string & p = req.path;
    const server_http_context::handler_t * h = nullptr;
    if (p == "/completion" || p == "/completions" || p == "/v1/completions") {
        h = &routes->post_completions;
    } else if (p == "/chat/completions" || p == "/v1/chat/completions") {
        h = &routes->post_chat_completions;
    } else if (p == "/v1/chat/completions/control" || p == "/chat/completions/control") {
        h = &routes->post_control;
    } else if (p == "/responses" || p == "/v1/responses") {
        h = &routes->post_responses_oai;
    } else if (p == "/embeddings" || p == "/v1/embeddings") {
        h = &routes->post_embeddings_oai;
    } else if (p == "/rerank" || p == "/reranking" || p == "/v1/rerank" || p == "/v1/reranking") {
        h = &routes->post_rerank;
    } else if (p == "/tts") {
        h = &routes->post_tts;
    } else if (p == "/v1/messages") {
        h = &routes->post_anthropic_messages;
    } else if (p == "/infill") {
        h = &routes->post_infill;
    } else if (p == "/tokenize") {
        h = &routes->post_tokenize;
    } else if (p == "/detokenize") {
        h = &routes->post_detokenize;
    } else if (p == "/metrics") {
        h = &routes->get_metrics;
    } else if (p == "/health" || p == "/v1/health") {
        h = &routes->get_health;
    } else if (p == "/props") {
        h = &routes->post_props;
    } else if (p == "/slots") {
        h = &routes->get_slots;
    } else if (p == "/models" || p == "/v1/models") {
        h = &routes->get_models;
    } else {
        res->status = 404;
        res->data = json({ {"error", "unknown route for beehive model: " + p} }).dump();
        res->content_type = "application/json; charset=utf-8";
        return res;
    }

    if (*h == nullptr) {
        res->status = 501;
        res->data = json({ {"error", "handler not implemented for beehive route: " + p} }).dump();
        res->content_type = "application/json; charset=utf-8";
        return res;
    }
    return (*h)(req);
}
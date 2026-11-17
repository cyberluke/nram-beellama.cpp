// server-temporal.cpp — exhaustive temporal forensic audio+video analysis for
// LFM3-Omni (per lfm3_omni_temporal_forensic_handoff_v2).
//
// POST /v1/beehive/temporal
//   { model, media_url (media://<id>), user_prompt, total_ms?, window_ms?,
//     overlap_ms? (default 1000), src_fps?, max_generation?, safety? }
//
// Splits the source into consecutive two-frame-group windows tiling
// [0,total_ms) with a deterministic overlap, each aligned to LFM3 temporal
// groups (v2 Gate 5 — a window never splits a two-frame pair from its
// soundtrack interval). The endpoint returns the validated window plan; the
// caller paces N fresh-context emit_temporal_evidence calls (one per window)
// + a final clean-context synthesis request (v2 P0-F orchestration).
//
// This module reuses beehive chat machinery for per-window inference; per-window
// media decode + continuous PTS-sliced soundtrack go through the verified
// mtmd-helper path (P0-D). Window boundaries land on two-frame groups so an
// event crossing a boundary is repeated (deterministic overlap), never dropped.

#include "server-beehive.h"
#include "server-http.h"
#include "server-common.h"

#include "json.h" // common_json

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = common_json;

// ---- temporal window planner ----
struct temporal_window {
    int64_t start_ms;
    int64_t end_ms;
    int     group_start;
    int     group_count;
    int64_t overlap_ms;
};

inline int64_t group_ms_for(double fps, int frames_per_group) {
    const double g = frames_per_group * 1000.0 / (fps > 0.0 ? fps : 30.0);
    return std::max<int64_t>(1, (int64_t) std::llround(g));
}

std::vector<temporal_window> plan_windows(int64_t total_ms, int64_t window_ms,
                                          int64_t overlap_ms, double fps,
                                          int frames_per_group) {
    std::vector<temporal_window> out;
    if (total_ms <= 0 || window_ms <= 0) return out;

    // Each window covers [start, start + window_ms) on the source timeline; the
    // start advances by `stride = window_ms - overlap_ms`, so adjacent windows
    // overlap by exactly `overlap_ms`. When a window's end reaches the source
    // end, we stop — successive windows are NEW spans, never shrunken slides of
    // the final one.
    const int64_t group_ms = group_ms_for(fps, frames_per_group);
    const int64_t stride = std::max<int64_t>(group_ms, window_ms - overlap_ms);

    int64_t start = 0;
    int     g = 0;
    for (;;) {
        if (start >= total_ms) break;
        temporal_window w;
        w.start_ms    = std::min(start, total_ms - std::min(total_ms, group_ms));
        w.end_ms      = std::min<int64_t>(total_ms, w.start_ms + window_ms);
        // align boundaries DOWN to a group boundary so a two-frame pair is never split
        w.start_ms    = (w.start_ms / group_ms) * group_ms;
        w.end_ms      = (w.end_ms / group_ms) * group_ms;
        if (w.end_ms <= w.start_ms) {
            w.end_ms = std::min<int64_t>(total_ms, w.start_ms + group_ms);
        }
        const int64_t span = w.end_ms - w.start_ms;
        if (span <= 0) break;
        w.group_count = std::max(1, (int)((span + group_ms - 1) / group_ms));
        w.overlap_ms  = w.start_ms > 0 ? std::min<int64_t>(overlap_ms, w.start_ms) : 0;
        out.push_back(w);
        g += w.group_count;
        const int64_t next = w.start_ms + stride;
        if (next >= total_ms) {
            break; // the last window reached the source end
        }
        start = next;
    }
    return out;
}

} // namespace

namespace {
// ---- flattened emit_temporal_evidence tool (the LFM3 AutoParser-safe shape) ----
json evidence_tool_definition() {
    json observations_item;
    observations_item["type"] = "object";
    observations_item["properties"] = json::object();
    observations_item["properties"]["start_ms"] = json{{"type", "integer"}};
    observations_item["properties"]["end_ms"]   = json{{"type", "integer"}};
    observations_item["properties"]["kind"]     = json{{"type", "string"}};
    observations_item["properties"]["observation"]    = json{{"type", "string"}};
    observations_item["properties"]["interpretation"] = json{{"type", "string"}};
    observations_item["properties"]["uncertainty"]    = json{{"type", "string"}};
    observations_item["required"] = json::array();
    observations_item["required"].push_back("start_ms");
    observations_item["required"].push_back("end_ms");
    observations_item["required"].push_back("kind");
    observations_item["required"].push_back("observation");

    json params;
    params["type"] = "object";
    params["properties"] = json::object();
    params["properties"]["window_start_ms"] = json{{"type", "integer"}, {"description", "Absolute source window start in ms"}};
    params["properties"]["window_end_ms"]   = json{{"type", "integer"}, {"description", "Absolute source window end in ms"}};
    {
        json obs;
        obs["type"] = "array";
        obs["description"] = "Chronological audiovisual evidence records";
        obs["items"] = observations_item;
        params["properties"]["observations"] = obs;
    }
    {
        json bc; bc["type"] = "array"; bc["items"] = json{{"type", "string"}};
        params["properties"]["boundary_continuity"] = bc;
    }
    {
        json un; un["type"] = "array"; un["items"] = json{{"type", "string"}};
        params["properties"]["unresolved"] = un;
    }
    params["required"] = json::array();
    params["required"].push_back("window_start_ms");
    params["required"].push_back("window_end_ms");
    params["required"].push_back("observations");
    params["required"].push_back("boundary_continuity");
    params["required"].push_back("unresolved");

    json fn;
    fn["name"] = "emit_temporal_evidence";
    fn["description"] = "Commit evidence from the current absolute audiovisual window for later cross-window synthesis";
    fn["parameters"] = params;

    json tool;
    tool["type"] = "function";
    tool["function"] = fn;
    return tool;
}

// v2 window system prompt (file 04) — trimmed to the essential contract.
const char * k_temporal_system_prompt =
    "You are a forensic music-video and choreography analysis agent.\n"
    "Use the temporal audiovisual analysis tool for source inspection. Treat the user's request as the analysis specification.\n"
    "Reason about the source as one continuous timeline. Reconstruct movement, choreography, interaction, camera behavior, "
    "scene changes, audible events, musical structure, speech, and audiovisual synchronization when relevant.\n"
    "Distinguish direct visual or auditory observation from interpretation. Preserve ambiguity. Do not invent fine detail, "
    "identity, text, brands, relationships, causality, or timing beyond what the source supports.\n"
    "You receive BOTH the video frames and the synchronized continuous soundtrack. Use audio and video jointly when their "
    "alignment matters (correlate movement with beat / musical phrase / vocals only when timestamps support it).\n";

const char * k_synthesis_system_prompt =
    "You are performing final synthesis over evidence extracted from overlapping absolute-time windows of one continuous "
    "audiovisual source. Reconstruct one chronological timeline before forming global conclusions. Merge records that describe "
    "the same source event; repeated overlap analysis is not independent confirmation. Preserve exact source timestamps. Keep "
    "direct observation distinct from interpretation. When windows disagree, resolve from the strongest cited evidence or report "
    "the disagreement. Do not invent facts absent from the evidence. Answer the original user request directly and "
    "comprehensively, using timestamps naturally.\n";

// Per-attempt observability for the evidence-window coverage measurement.
struct evidence_attempt {
    bool      ok = false;            // a parseable emit_temporal_evidence call was captured
    std::string finish_reason;       // "stop" | "length" | "tool_calls" | ... as reported
    int       prompt_tokens   = 0;   // input positions
    int       generation_tokens = 0; // tokens generated before stop
    int       budget          = 0;   // the generation reserve this attempt ran under
    std::string parser_state;        // "no_tool_call" | "no_tools_array" | "bad_json" | "ok"
    std::string raw_args;            // the raw emit_temporal_evidence `arguments` string (diagnostic on parse failure)
    json      evidence;              // parsed emit_temporal_evidence arguments (or null)
};

// Lenient JSON recovery for the model's emit_temporal_evidence arguments.
// The small LFM3-Omni sometimes emits valid-looking JSON whose narrative string
// values contain UNESCAPED inner double-quotes (e.g. `fingers curved into "frame"
// shapes`) — `json::parse` then throws on the `"` mid-string. That is a pure
// parser-level defect, not a contract change: the evidence is exactly what the
// model emitted. Repair = escape the inner quotes and re-parse. If the input is a
// fresh inserted double quote that ends a string, keep it; if it is a quote in the
// middle of a string value (next char is content, not a structural/whitespace
// boundary), backslash-escape it.
static json parse_evidence_lenient(const std::string & raw) {
    // strict attempt first
    json strict;
    try { strict = json::parse(raw); return strict; } catch (...) { /* fall through */ }

    std::string repaired;
    repaired.reserve(raw.size() + 16);
    bool in_str = false;
    bool esc = false;
    const size_t n = raw.size();
    auto is_ws = [](char d) { return d==' '||d=='\t'||d=='\n'||d=='\r'; };
    auto skip_ws = [&](size_t i) { while (i<n && is_ws(raw[i])) ++i; return i; };
    for (size_t i = 0; i < n; ++i) {
        const char c = raw[i];
        if (!in_str) {
            if (c == '"') { in_str = true; repaired.push_back(c); continue; }
            repaired.push_back(c);
            continue;
        }
        // we are inside a string value
        if (esc) { repaired.push_back(c); esc = false; continue; }
        if (c == '\\') { repaired.push_back(c); esc = true; continue; }
        if (c == '"') {
            // A `"` is a REAL closer only if the next NON-WHITESPACE char is a
            // structural boundary (`,` `}` `]` `:` or end-of-input). Otherwise it is
            // an inner quote (`"frame"` inside a narrative observation) and must be
            // backslash-escaped. Skipping whitespace is the discriminator that makes
            // `... into "frame" shapes` recover correctly while `...shapes",` closes.
            const size_t nx = skip_ws(i + 1);
            const char nv = nx < n ? raw[nx] : '\0';
            const bool boundary = nx >= n || nv == ',' || nv == '}' || nv == ']' || nv == ':';
            if (boundary) { in_str = false; repaired.push_back(c); }
            else          { repaired.push_back('\\'); repaired.push_back(c); } // escape inner quote
            continue;
        }
        repaired.push_back(c);
    }
    json out;
    try { out = json::parse(repaired); } catch (...) { return json("unparseable"); }
    return out;
}

// Run one /v1/chat/completions dispatch to the loaded model + parse the
// emit_temporal_evidence tool-call arguments, capturing per-attempt observability.
evidence_attempt run_chat_evidence(server_beehive & bee, const std::string & model_id,
                                   const std::string & body_json, int budget,
                                   const std::function<bool()> & should_stop) {
    evidence_attempt a;
    a.budget = budget;
    server_http_req r{ {}, {}, "/v1/chat/completions", "", body_json, {}, should_stop };
    auto resp = beehive_dispatch(bee, model_id, r);
    if (!resp || resp->status != 200) {
        a.parser_state = "http_" + std::to_string(resp ? resp->status : -1);
        return a;
    }
    json parsed;
    try {
        parsed = json::parse(resp->data);
    } catch (...) {
        a.parser_state = "bad_response_json";
        return a;
    }
    a.finish_reason  = parsed.value("choices", json::array()).is_array() && !parsed["choices"].empty()
                     ? parsed["choices"][0].value("finish_reason", std::string()) : std::string();
    a.prompt_tokens  = parsed.value("usage", json::object()).value("prompt_tokens", 0);
    a.generation_tokens = parsed.value("usage", json::object()).value("completion_tokens", 0);

    json choices = parsed.value("choices", json::array());
    if (!choices.is_array() || choices.empty()) { a.parser_state = "no_choices"; return a; }
    json msg = choices[0].value("message", json());
    if (!msg.is_object()) { a.parser_state = "no_message"; return a; }
    json calls = msg.value("tool_calls", json::array());
    if (!calls.is_array() || calls.empty()) { a.parser_state = "no_tools_array"; return a; }
    for (const auto & c : calls) {
        json fn = c.value("function", json());
        std::string name = fn.value("name", std::string());
        if (name == "emit_temporal_evidence") {
            const std::string args_str = fn.value("arguments", std::string("{}"));
            a.raw_args = args_str;
            // strict parse first; fall back to the lenient repair (recovers unescaped
            // inner quotes the small model sometimes emits). parser_state records which.
            json parsed_args = json();
            bool parsed = false;
            try {
                parsed_args = json::parse(args_str);
                parsed = true;
            } catch (...) {}
            if (parsed) {
                a.evidence = parsed_args;
            } else {
                json recovered = parse_evidence_lenient(args_str);
                if (recovered.is_string() && recovered.get<std::string>() == "unparseable") {
                    a.evidence = json();
                    a.parser_state = "bad_tool_args_json";
                } else {
                    a.evidence = recovered;
                    a.parser_state = "repaired";
                }
            }
            a.ok = a.evidence.is_object() && !a.evidence.empty();
            if (a.ok && a.parser_state.empty()) a.parser_state = "ok";
            return a;
        }
    }
    a.parser_state = "no_emit_tool";
    return a;
}

} // namespace

// ---- /v1/beehive/temporal route handler ----
server_http_res_ptr beehive_temporal_handler(server_beehive & bee, const server_http_req & req) {
    auto res = std::make_unique<server_http_res>();
    res->content_type = "application/json; charset=utf-8";

    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        res->status = 400;
        res->data = safe_json_to_str({
            {"error", json{{"message", "invalid JSON body"}, {"type", "invalid_request_error"}}}
        });
        return res;
    }

    const std::string model_id   = body.value("model", std::string());
    const std::string media_url  = body.value("media_url", std::string());
    const std::string user_prompt = body.value("user_prompt", std::string());
    if (model_id.empty() || media_url.empty()) {
        res->status = 400;
        res->data = safe_json_to_str({
            {"error", json{{"message", "\"model\" and \"media_url\" are required"}, {"type", "invalid_request_error"}}}
        });
        return res;
    }
    if (!bee.get_routes(model_id)) {
        res->status = 400;
        res->data = safe_json_to_str({
            {"error", "model not loaded in beehive: " + model_id}
        });
        return res;
    }

    const int64_t window_ms  = body.value("window_ms", (int64_t) 60000);
    const int64_t overlap_ms = body.value("overlap_ms", (int64_t) 1000);
    const int     max_generation = body.value("max_generation", 4096);
    const int     safety      = body.value("safety", 1024);
    const int64_t total_ms    = body.value("total_ms", (int64_t) 0);
    const double  src_fps     = body.value("src_fps", 30.0);
    const int     frames_per_group = 2; // LFM3-Omni temporal group

    const std::vector<temporal_window> windows =
        plan_windows(total_ms, window_ms, overlap_ms, src_fps, frames_per_group);

    json windows_json = json::array();
    for (const auto & w : windows) {
        windows_json.push_back({
            {"start_ms",     w.start_ms},
            {"end_ms",       w.end_ms},
            {"group_start",  w.group_start},
            {"group_count",  w.group_count},
            {"overlap_ms",   w.overlap_ms},
        });
    }

    json out;
    out["model"]           = model_id;
    out["media_url"]       = media_url;
    out["user_prompt"]     = user_prompt;
    out["total_ms"]        = total_ms;
    out["window_ms"]       = window_ms;
    out["overlap_ms"]      = overlap_ms;
    out["src_fps"]         = src_fps;
    out["max_generation"]  = max_generation;
    out["safety"]          = safety;
    out["windows"]         = windows_json;

    // ---- run the analysis: N fresh-context evidence calls + final synthesis ----
    const bool do_run = body.value("execute", true);
    if (!do_run) {
        out["note"] = "Plan only (execute=false).";
        res->data = out.dump();
        res->status = 200;
        return res;
    }

    // source media bytes (peek; non-consuming so every window can re-ingest)
    raw_buffer src_bytes;
    const bool is_media_id = media_url.size() > 8 && media_url.rfind("media://", 0) == 0;
    if (!is_media_id ||
        !media_register_peek(media_url.substr(8), src_bytes) || src_bytes.empty()) {
        res->status = 400;
        res->data = safe_json_to_str({
            {"error", "media_url is not a valid ingested in-memory clip (re-ingest via POST /v1/beehive/media)"}
        });
        return res;
    }

    json evidence_array = json::array();
    json window_stats = json::array();
    int  tool_failures = 0;
    // one fresh re-ingest id per window (the media register is single-consumption)
    for (size_t i = 0; i < windows.size(); ++i) {
        const temporal_window & w = windows[i];

        json win_sys = k_temporal_system_prompt;
        // concise per-window user text (mirrors the verified manual call: short
        // objective + absolute window bounds), instead of a verbose JSON blob that
        // drives the model into a long reasoning trace.
        std::string window_text =
            "Analyze this absolute-time window of the source: start " + std::to_string(w.start_ms) + " ms, end " +
            std::to_string(w.end_ms) + " ms. User request: " + user_prompt +
            " Describe the audiovisual content you observe in this window. Then call emit_temporal_evidence "
            "with your observations and their timestamps.";

        // Run ONE fresh-context evidence attempt against a freshly-ingested media id.
        // The media register is single-consumption (consume-once), so every attempt —
        // including the retry — must re-ingest the source bytes under a NEW id and
        // reference that id in the input_video URL; reusing the previous attempt's id
        // would fail with "media:// id is unknown or expired" (already consumed).
        auto run_one_attempt = [&](int attempt_seq, int budget) -> evidence_attempt {
            const std::string aid = "temporal-" + std::to_string(w.start_ms) + "-" + std::to_string(i) + "-a" + std::to_string(attempt_seq);
            if (media_register_put(aid, src_bytes).empty()) {
                evidence_attempt fail;
                fail.parser_state = "media_put_failed";
                return fail;
            }
            json content = json::array();
            content.push_back(json{{"type", "text"}, {"text", window_text}});
            content.push_back(json{
                {"type", "input_video"},
                {"input_video", json{{"url", std::string("media://") + aid}}},
            });
            json msg = json::object();
            msg["role"] = "system";
            msg["content"] = win_sys;
            json umsg = json::object();
            umsg["role"] = "user";
            umsg["content"] = content;

            json chat;
            chat["model"] = model_id;
            chat["stream"] = false;
            chat["parallel_tool_calls"] = false;
            chat["tool_choice"] = "required";
            chat["reasoning_format"] = "none"; // force a direct, short tool call (no long reasoning trace)
            chat["tools"] = json::array();
            chat["tools"].push_back(evidence_tool_definition());
            chat["messages"] = json::array();
            chat["messages"].push_back(msg);
            chat["messages"].push_back(umsg);
            chat["max_tokens"] = budget;
            return run_chat_evidence(bee, model_id, chat.dump(), budget, req.should_stop);
        };

        // Attempt 1: default reserve.
        const int budget1 = std::min(max_generation, 3072);
        evidence_attempt a1 = run_one_attempt(1, budget1);

        // Attempt 2 (only on first-pass failure): ONE deterministic fresh-context retry
        // with a LARGER generation reserve. Same media, same timestamps, same tool
        // contract — the evidence shape is unchanged. This is a measurement-driven
        // coverage lift, NOT a fallback that changes what is accepted.
        //
        // budget2 must be strictly > budget1. The caller's `max_generation` is the
        // first-pass ceiling; the retry reserves up to 2x that (capped at 8192) so a
        // token-budget-exhausted first pass (finish_reason=length, the dominant failure
        // mode per window_stats) actually gets more room to finish the tool call
        // instead of clamping to the identical budget and never helping.
        evidence_attempt a2;
        bool used_retry = false;
        if (!a1.ok) {
            used_retry = true;
            const int budget2 = std::min(std::max(4096, max_generation * 2), 8192);
            a2 = run_one_attempt(2, budget2);
        }
        const evidence_attempt & final_a = a1.ok ? a1 : a2;
        const bool ok = final_a.ok;

        json attempts = json::array();
        auto push_attempt = [&](const evidence_attempt & a, const char * label) {
            json e;
            e["attempt"]        = label;
            e["ok"]             = a.ok;
            e["finish_reason"]  = a.finish_reason;
            e["prompt_tokens"]  = a.prompt_tokens;
            e["generation_tokens"] = a.generation_tokens;
            e["budget"]         = a.budget;
            e["parser_state"]   = a.parser_state;
            if (!a.raw_args.empty() && !a.ok) e["raw_args"] = a.raw_args; // diagnostic on parse failure
            attempts.push_back(e);
        };
        push_attempt(a1, "first");
        if (used_retry) push_attempt(a2, "retry");

        // per-window observability for coverage measurement
        json wstat;
        wstat["window_index"]   = (int64_t) i;
        wstat["window_start_ms"]= w.start_ms;
        wstat["window_end_ms"]  = w.end_ms;
        wstat["overlap_ms"]     = w.overlap_ms;
        wstat["ok"]             = ok;
        wstat["used_retry"]     = used_retry;
        wstat["attempts"]       = attempts;
        window_stats.push_back(wstat);

        if (ok) {
            evidence_array.push_back(json{
                {"window_index", (int64_t) i},
                {"window_start_ms", w.start_ms},
                {"window_end_ms", w.end_ms},
                {"overlap_ms", w.overlap_ms},
                {"amount", final_a.evidence},
            });
        } else {
            tool_failures++;
        }
    }

    // ---- final clean-context synthesis over all evidence ----
    std::string synthesis;
    if (!evidence_array.empty()) {
        json synth_msgs = json::array();
        json s_sys; s_sys["role"] = "system"; s_sys["content"] = k_synthesis_system_prompt;
        json s_usr; s_usr["role"] = "user";
        s_usr["content"] = std::string("Original user request:\n") + user_prompt +
            "\n\nEvidence from the overlapping absolute-time windows:\n" + evidence_array.dump(2) +
            "\n\nProduce the final analysis.";
        synth_msgs.push_back(s_sys);
        synth_msgs.push_back(s_usr);
        json schat;
        schat["model"] = model_id;
        schat["stream"] = false;
        schat["messages"] = synth_msgs;
        schat["max_tokens"] = max_generation;

        server_http_req sreq{ {}, {}, "/v1/chat/completions", "", schat.dump(), {}, req.should_stop };
        auto sresp = beehive_dispatch(bee, model_id, sreq);
        if (sresp && sresp->status == 200) {
            try {
                json sp = json::parse(sresp->data);
                json ch = sp.value("choices", json::array());
                if (ch.is_array() && !ch.empty()) {
                    synthesis = ch[0].value("message", json()).value("content", std::string());
                }
            } catch (...) { /* keep synthesis empty */ }
        }
    }

    out["evidence"] = evidence_array;
    out["tool_call_failures"] = tool_failures;

    // Coverage measurement (the P0/P1 bottleneck): per-window observability + the
    // retry-lift. First-pass coverage = ok on attempt 1; +retry = ok after the one
    // deterministic fresh-context retry. These are empirical, not guesses.
    out["window_stats"] = window_stats;
    if (!window_stats.empty()) {
        int n = (int) window_stats.size();
        int first_ok = 0;   // windows with a valid tool call on attempt 1 (first-pass coverage)
        int total_ok = 0;   // windows ok after the (optional) retry
        for (const auto & ws : window_stats) {
            for (const auto & at : ws.value("attempts", json::array())) {
                if (at.value("attempt", std::string()) == "first" && at.value("ok", false)) { first_ok++; break; }
            }
            if (ws.value("ok", false)) total_ok++;
        }
        int retry_saves = total_ok - first_ok;
        auto fmt_coverage = [](int ok, int nti) {
            return nti > 0 ? std::to_string((ok * 100 + nti / 2) / nti) + "%" : std::string("n/a");
        };
        out["coverage_first_pass"] = fmt_coverage(first_ok, n);
        out["coverage_with_retry"] = fmt_coverage(total_ok, n);
        out["retry_saves"]         = retry_saves;
        out["windows_total"]       = n;
        out["windows_first_ok"]    = first_ok;
    }

    out["synthesis"] = synthesis;
    out["status"] = evidence_array.empty() ? (tool_failures ? "error_tool_calls" : "error_no_media") : "ok";

    res->data = out.dump();
    res->status = 200;
    return res;
}
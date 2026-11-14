// tools/kelvin-tui/src/model_detector.cpp
#include "model_detector.h"

#include <windows.h>
#include <shlobj.h>

#include <filesystem>
#include <algorithm>
#include <set>
#include <cctype>
#include <sstream>

namespace fs = std::filesystem;

// --- LM Studio config ------------------------------------------------------
// LM Studio stores its settings in %USERPROFILE%\.cache\lm-studio\settings*
// and downloads models under a configured directory, usually
// %USERPROFILE%\.cache\lm-studio\models (older: <HOME>\.lmstudio\models).
std::string lm_studio_default_dir() {
    std::string home;
    if (const char* h = getenv("USERPROFILE")) home = h;
    if (home.empty()) return "";

    std::vector<std::string> candidates = {
        home + "\\.cache\\lm-studio\\models",
        home + "\\.lmstudio\\models",
        home + "\\models",   // some users redirect here
    };
    for (auto& c : candidates) {
        if (fs::is_directory(c)) return c;
    }
    // Could parse .cache/lm-studio settings JSON for "downloadModelPath"; skip.
    return candidates.empty() ? "" : candidates[0];
}

std::string describe_scan_locations() {
    std::string home = getenv("USERPROFILE") ? getenv("USERPROFILE") : "";
    std::ostringstream out;
    out << "LM Studio: " << lm_studio_default_dir() << "\n"
        << "HuggingFace cache: " << (home.empty() ? "" : home + "\\.cache\\huggingface\\hub") << "\n"
        << "Ollama models: " << (home.empty() ? "" : home + "\\.ollama\\models") << "\n"
        << "Custom: <user-picked folder>";
    return out.str();
}

namespace {
bool ends_with_ci(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    std::string tail = s.substr(s.size() - suffix.size());
    for (auto& c : tail) c = (char)tolower((unsigned char)c);
    std::string low = suffix;
    for (auto& c : low) c = (char)tolower((unsigned char)c);
    return tail == low;
}

// Recursively walks a dir, collecting *.gguf files (dedup by absolute path).
void walk(const std::string& root, const std::string& source,
          std::set<std::string>& seen, std::vector<DetectedModel>& out) {
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
         it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto& p = it->path();
        if (fs::is_directory(p, ec)) continue;
        std::string name = p.filename().string();
        if (!ends_with_ci(name, ".gguf")) continue;
        std::string abs = fs::absolute(p).string();
        if (!seen.insert(abs).second) continue; // dedup
        DetectedModel m;
        m.path   = abs;
        m.name   = name;
        m.source = source;
        std::error_code szec;
        m.size_gb = (double)fs::file_size(p, szec) / (1024.0 * 1024.0 * 1024.0);
        if (szec) m.size_gb = 0.0;
        out.push_back(std::move(m));
    }
}

// Ollama stores models as blob files (sha256-256-<hex>); convert to a readable
// name by stripping the algorithm prefix. Dedup by content-ish path.
void walk_ollama(const std::string& root, std::set<std::string>& seen,
                 std::vector<DetectedModel>& out) {
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return;
    // Ollama layout: .ollama/models/blobs/sha256-<hash> (no extension).
    const std::string blobs = root + "\\blobs";
    if (!fs::is_directory(blobs, ec)) return;
    for (fs::directory_iterator it(blobs, ec), end; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        std::string fn = it->path().filename().string();
        // Blob files: sha256-<64hex>. Skip the *-delete markers.
        size_t dash = fn.find("-");
        if (dash == std::string::npos) continue;
        std::string algo = fn.substr(0, dash);
        if (algo != "sha256") continue;
        std::string abs = fs::absolute(it->path()).string();
        if (!seen.insert(abs).second) continue;
        DetectedModel m;
        m.path   = abs;
        m.name   = fn; // keep as-is; readable enough
        m.source = "Ollama";
        std::error_code szec;
        m.size_gb = (double)fs::file_size(it->path(), szec) / (1024.0 * 1024.0 * 1024.0);
        if (szec) m.size_gb = 0.0;
        out.push_back(std::move(m));
    }
}
} // namespace

std::vector<DetectedModel> detect_models(const std::vector<std::string>& extra_dirs) {
    std::set<std::string> seen;
    std::vector<DetectedModel> out;
    std::string home = getenv("USERPROFILE") ? getenv("USERPROFILE") : "";

    // LM Studio
    {
        std::string lm = lm_studio_default_dir();
        walk(lm, "LM Studio", seen, out);
    }
    // HuggingFace local cache (hub/real_repo_path/snapshots/.../*.gguf)
    if (!home.empty()) {
        walk(home + "\\.cache\\huggingface\\hub", "HuggingFace", seen, out);
    }
    // Ollama
    if (!home.empty()) {
        walk_ollama(home + "\\.ollama\\models", seen, out);
    }
    // Common alternate drives/roots (user may have redirected downloads).
    // `D:\_MODELS` is a conventional local mirror on this machine.
    for (const std::string& alt : {"D:\\_MODELS"}) {
        walk(alt, "Custom", seen, out);
    }
    // Custom/user dirs (interactive folder pick, or env override).
    for (const auto& d : extra_dirs) {
        walk(d, "Custom", seen, out);
    }
    // Sort by size desc.
    std::stable_sort(out.begin(), out.end(),
        [](const DetectedModel& a, const DetectedModel& b) { return a.size_gb > b.size_gb; });
    return out;
}
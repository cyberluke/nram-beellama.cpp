// tools/kelvin-tui/src/model_detector.h
#pragma once

#include <string>
#include <vector>

struct DetectedModel {
    std::string path;       // absolute path to the .gguf
    std::string name;       // human-friendly (filename without .gguf)
    std::string source;     // "LM Studio", "HuggingFace", "Ollama", "Custom"
    double      size_gb = 0.0;
};

// Scans well-known model directories (LM Studio default, user folder,
// HuggingFace local cache, Ollama) plus an optional extra path.
std::vector<DetectedModel> detect_models(const std::vector<std::string>& extra_dirs);

// Returns a human-readable list of scan locations actually probed.
std::string describe_scan_locations();

// Best-effort: fetch LM Studio's configured download dir from its config.
std::string lm_studio_default_dir();
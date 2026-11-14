// tools/kelvin-tui/src/settings.h
#pragma once

#include <string>

// Persisted user settings (model + preset) in a simple ini-like file laid
// next to the executable. Greets returning users without losing their pick.
struct UserSettings {
    std::string model_path;   // absolute GGUF path last chosen
    std::string preset_id;    // catalog preset id ("" = autopilot)
    std::string parallel;     // "0" | "1" - desired 2-client mode
    std::string lang;         // "en" | "cs" | "sk"
};

// Returns the path of the settings file (next to the exe: settings.ini).
std::string settings_path();

bool settings_load(UserSettings& out);
bool settings_save(const UserSettings& in);

// Returns the directory of the current executable.
std::string exe_dir();
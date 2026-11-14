// tools/kelvin-tui/src/settings.cpp
#include "settings.h"

#include <windows.h>
#include <shlobj.h>

#include <fstream>
#include <sstream>

std::string exe_dir() {
    char self[MAX_PATH];
    GetModuleFileNameA(nullptr, self, MAX_PATH);
    std::string p(self);
    size_t pos = p.find_last_of("\\/");
    return pos == std::string::npos ? p : p.substr(0, pos);
}

std::string settings_path() {
    return exe_dir() + "\\settings.ini";
}

bool settings_load(UserSettings& out) {
    std::ifstream fin(settings_path());
    if (!fin) return false;
    UserSettings s;
    std::string line;
    while (std::getline(fin, line)) {
        // strip trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "model") s.model_path = val;
        else if (key == "preset") s.preset_id = val;
        else if (key == "parallel") s.parallel = val;
        else if (key == "lang") s.lang = val;
    }
    out = s;
    return !s.model_path.empty();
}

bool settings_save(const UserSettings& in) {
    std::ofstream fout(settings_path());
    if (!fout) return false;
    fout << "model=" << in.model_path << "\n";
    fout << "preset=" << in.preset_id << "\n";
    fout << "parallel=" << (in.parallel.empty() ? "0" : in.parallel) << "\n";
    fout << "lang=" << (in.lang.empty() ? "en" : in.lang) << "\n";
    return true;
}
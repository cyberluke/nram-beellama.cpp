// tools/kelvin-tui/src/i18n.h
#pragma once

#include <string>
#include <vector>
#include <map>

enum class Lang {
    En,
    Cs,
    Sk,
};

// Human name of each language (in its own language).
inline const char* lang_name(Lang l) {
    switch (l) {
        case Lang::En: return "English";
        case Lang::Cs: return "Čeština";
        case Lang::Sk: return "Slovenčina";
    }
    return "English";
}

// Two-letter code used in settings.
inline const char* lang_code(Lang l) {
    switch (l) {
        case Lang::En: return "en";
        case Lang::Cs: return "cs";
        case Lang::Sk: return "sk";
    }
    return "en";
}

// Three-line ASCII flag for each language; color applied by caller.
inline const std::vector<std::string> lang_flag(Lang l) {
    switch (l) {
        case Lang::En:
            return {
                "\x1b[30m-\x1b[0m\x1b[47m-\x1b[0m\x1b[34m-",
                "\x1b[30m-\x1b[0m\x1b[47m-\x1b[0m\x1b[34m-",
                "\x1b[30m-\x1b[0m\x1b[47m-\x1b[0m\x1b[34m-",
            };
        case Lang::Cs:
            return {
                "\x1b[47m                                  \x1b[0m",
                "\x1b[41m                                  \x1b[0m",
                "\x1b[44m                                  \x1b[0m",
            };
        case Lang::Sk:
            return {
                "\x1b[47m                                  \x1b[0m",
                "\x1b[34m                                  \x1b[0m",
                "\x1b[41m                                  \x1b[0m",
            };
    }
    return {};
}

// A table of all UI strings, keyed by an id and resolved per language.
class I18n {
public:
    explicit I18n(Lang l = Lang::En) : lang_(l) {}
    Lang lang() const { return lang_; }
    void set(Lang l) { lang_ = l; }

    // Resolve a key. Falls back to English if missing.
    std::string t(const std::string& key) const;

    Lang lang_ = Lang::En;
};

// Detect the Windows UI language; maps to one we support, else English.
Lang detect_lang();
// tools/kelvin-tui/src/i18n.cpp
#include "i18n.h"

#include <windows.h>
#include <winnls.h>

#include <cstring>

// String catalog. Missing keys fall back to English.
static const std::map<std::string, std::map<Lang, const char*>> kStrings = {
    {"app.title",        {{Lang::En,"The Lab"},        {Lang::Cs,"Laboratoř"},        {Lang::Sk,"Laboratórium"}}},
    {"app.tagline",      {{Lang::En,"Kelvin the Software Chemist"}, {Lang::Cs,"Kelvin, softwarový chemik"}, {Lang::Sk,"Kelvin, softvérový chemik"}}},
    {"app.subtagline",   {{Lang::En,"purifying dependency trees to 99.1%"}, {Lang::Cs,"čistíme stromy závislostí na 99,1 %"}, {Lang::Sk,"čistíme stromy závislostí na 99,1 %"}}},

    {"choose_lang",      {{Lang::En,"The Lab - choose language"}, {Lang::Cs,"Laboratoř - vyberte jazyk"}, {Lang::Sk,"Laboratórium - vyberte jazyk"}}},
    {"choose_model",     {{Lang::En,"The Lab - choose your crystal"}, {Lang::Cs,"Laboratoř - vyberte svůj krystal"}, {Lang::Sk,"Laboratórium - vyberte kryštál"}}},
    {"choose_preset",    {{Lang::En,"The Lab - tune the mix"}, {Lang::Cs,"Laboratoř - nastavte mix"}, {Lang::Sk,"Laboratórium - nastavte mix"}}},
    {"nav_hint",         {{Lang::En," arrows to move, Enter to pick, q to quit"}, {Lang::Cs," šipky pro pohyb, Enter pro výběr, q pro ukončení"}, {Lang::Sk," šípky pre pohyb, Enter pre výber, q pre ukončenie"}}},
    {"browse_folder",    {{Lang::En,"[+] Choose a model folder..."}, {Lang::Cs,"[+] Vyberte složku s modely..."}, {Lang::Sk,"[+] Vyberte priečinok s modelmi..."}}},
    {"auto_recommended", {{Lang::En,"recommended"}, {Lang::Cs,"doporučeno"}, {Lang::Sk,"odporúčané"}}},
    {"status_ready",     {{Lang::En,"ready"}, {Lang::Cs,"připraveno"}, {Lang::Sk,"pripravené"}}},
    {"status_thinking",  {{Lang::En,"thinking (streaming)..."}, {Lang::Cs,"přemýšlím (streamuji)..."}, {Lang::Sk,"premýšľam (streamujem)..."}}},
    {"status_error",     {{Lang::En,"error"}, {Lang::Cs,"chyba"}, {Lang::Sk,"chyba"}}},
    {"input_placeholder",{{Lang::En,"Message Kelvin the Software Chemist..."}, {Lang::Cs,"Napište zprávu Kelvinovi..."}, {Lang::Sk,"Napíšte správu Kelvinovi..."}}},
    {"status_hint",      {{Lang::En,"[Enter]=send  [Esc]=quit"}, {Lang::Cs,"[Enter]=odeslat  [Esc]=konec"}, {Lang::Sk,"[Enter]=odoslať  [Esc]=koniec"}}},
    {"waiting_message",  {{Lang::En,"(no messages yet)"}, {Lang::Cs,"(zatím žádné zprávy)"}, {Lang::Sk,"(zatiaľ žiadne správy)"}}},
    {"starting_server",  {{Lang::En,"starting llama-server..."}, {Lang::Cs,"spouštím llama-server..."}, {Lang::Sk,"spúšťam llama-server..."}}},
    {"lg_label",         {{Lang::En,"English"}, {Lang::Cs,"Čeština"}, {Lang::Sk,"Slovenčina"}}},
};

std::string I18n::t(const std::string& key) const {
    auto it = kStrings.find(key);
    if (it == kStrings.end()) return key;
    auto it2 = it->second.find(lang_);
    if (it2 != it->second.end()) return it2->second;
    // Fall back to English.
    auto it3 = it->second.find(Lang::En);
    return it3 != it->second.end() ? it3->second : key;
}

Lang detect_lang() {
    // GetUserDefaultUILanguage gives the Windows UI (or preferred UI) language.
    LANGID lang = GetUserDefaultUILanguage();
    WORD pri = PRIMARYLANGID(lang);
    WORD sub = SUBLANGID(lang);
    (void)sub;
    if (pri == LANG_CZECH) return Lang::Cs;
    if (pri == LANG_SLOVAK) return Lang::Sk;
    return Lang::En;
}
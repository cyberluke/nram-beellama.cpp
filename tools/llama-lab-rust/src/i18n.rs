/// Minimal i18n: English / Czech / Slovak catalogs with fallback to key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Cs,
    Sk,
}

impl Lang {
    pub fn name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Cs => "Čeština",
            Lang::Sk => "Slovenčina",
        }
    }

    pub fn from_index(i: usize) -> Lang {
        match i % 3 {
            0 => Lang::En,
            1 => Lang::Cs,
            _ => Lang::Sk,
        }
    }
}

use std::collections::HashMap;

type Table = HashMap<&'static str, &'static str>;

static ENTRIES: &[(&str, &str, &str, &str)] = &[
    ("app.title", "Llama — The Lab", "Llama — Laboratoř", "Llama — Laboratórium"),
    ("app.tagline", "Local LLM lab", "Lokální LLM laboratoř", "Lokálna LLM laboratórium"),
    ("app.subtagline", "inference crafted for your GPU", "inference šitá na míru vaší GPU", "inferencia šitá na mieru vašej GPU"),
    ("choose_lang", "choose language", "vyberte jazyk", "vyberte jazyk"),
    ("choose_backend", "choose compute backend", "vyberte výpočetní backend", "vyberte výpočtový backend"),
    ("hint_backend", "◀/▶ select · Enter continue · Backspace back", "◀/▶ výběr · Enter pokračovat · Backspace zpět", "◀/▶ výber · Enter pokračovať · Backspace späť"),
    ("mem_avail", "available", "k dispozici", "k dispozícii"),
    ("be_backend_of", "of", "z", "z"),
    ("be_cpu", "CPU", "CPU", "CPU"),
    ("be_cpu_r1", "Works anywhere, zero setup", "Funguje všude, nulové nastavení", "Funguje všade, nulové nastavenie"),
    ("be_cpu_r2", "Largest models via RAM", "Největší modely přes RAM", "Najväčšie modely cez RAM"),
    ("be_cpu_r3", "Slower tokens, no GPU needed", "Pomalejší tokeny, bez GPU", "Pomalšie tokeny, bez GPU"),
    ("be_cuda", "CUDA", "CUDA", "CUDA"),
    ("be_cuda_r1", "Fastest on NVIDIA GPUs", "Nejrychlejší na NVIDIA GPU", "Najrýchlejší na NVIDIA GPU"),
    ("be_cuda_r2", "Flash-attn + KVarN cache", "Flash-attn + KVarN cache", "Flash-attn + KVarN cache"),
    ("be_cuda_r3", "Best for big contexts", "Nejlepší pro velké kontexty", "Najlepší pre veľké kontexty"),
    ("be_vulkan", "Vulkan", "Vulkan", "Vulkan"),
    ("be_vulkan_r1", "Works on AMD/Intel/NVIDIA", "Funguje na AMD/Intel/NVIDIA", "Funguje na AMD/Intel/NVIDIA"),
    ("be_vulkan_r2", "Solid speed on any GPU", "Slušná rychlost na jakékoli GPU", "Slušná rýchlosť na akejkoľvek GPU"),
    ("be_vulkan_r3", "KVarN + Flash-attn, no CUDA needed", "KVarN + Flash-attn bez CUDA", "KVarN + Flash-attn bez CUDA"),
    ("gpu_diff_cuda", "why CUDA beats Vulkan on this card", "proč je CUDA na této kartě lepší než Vulkan", "prečo je CUDA na tejto karte lepšia ako Vulkan"),
    ("gpu_diff_vulkan", "why Vulkan beats CUDA here", "proč je zde Vulkan lepší než CUDA", "prečo je tu Vulkan lepší ako CUDA"),
    ("choose_model", "choose a model", "vyberte model", "vyberte model"),
    ("add_folder", "[+] Add custom model folder…", "[+] Přidat vlastní složku…", "[+] Pridať vlastný priečinok…"),
    ("fleet_mode", "Fleet mode", "Režim fleet", "Režim fleet"),
    ("status_ready", "ready", "připraveno", "pripravené"),
    ("status_thinking", "thinking (streaming)…", "přemýšlím (streamuji)…", "premýšľam (streamujem)…"),
    ("status_error", "error", "chyba", "chyba"),
    ("status_starting", "starting llama-server…", "spouštím llama-server…", "spúšťam llama-server…"),
    ("input_placeholder", "Message the local model…", "Napište zprávu…", "Napíšte správu…"),
    ("hint_models", "↑/k ↓/j pick · Enter start · + folder · P profile · F fleet", "↑/k ↓/j výběr · Enter start · + složka · P profil · F fleet", "↑/k ↓/j výber · Enter štart · + priečinok · P profil · F fleet"),
    ("hint_chat", "Enter send · Esc clear · Alt+T theme · Alt+N new · Alt+Q quit", "Enter odeslat · Esc smazat · Alt+T téma · Alt+N nový · Alt+Q konec", "Enter poslať · Esc vymazať · Alt+T téma · Alt+N nový · Alt+Q koniec"),
    ("no_messages", "No messages yet.", "Zatím žádné zprávy.", "Zatiaľ žiadne správy."),
    ("no_models", "No models found. Press + to add a folder.", "Žádné modely. Stiskněte + a přidejte složku.", "Žiadne modely. Stlačte + a pridajte priečinok."),
    ("server_not_started", "(server not started)", "(server nespusten)", "(server nespustený)"),
    ("server_log", "llama-server log", "protokol llama-server", "protokol llama-server"),
    ("chat", "Chat", "Chat", "Chat"),
    ("tut_fleet", "F = Fleet: počet paralelních slotů (klientů) sdílejících jeden KV fond", "F = Fleet: počet paralelných slotov (klientov) zdieľajúcich jeden KV fond", "F = Fleet: počet paralelných slotov (klientov) zdieľajúcich jeden KV fond"),
    // help overlay
    ("help_send", "send message", "odeslat zprávu", "poslať správu"),
    ("help_recall", "recall prompt history", "historie povědomí (↑/↓)", "história promptov"),
    ("help_scroll", "scroll llama console", "rolovat konzoli llama", "rolovať konzolu llama"),
    ("help_resize", "resize panes", "změnit velikost panelů", "zmeniť veľkosť panelov"),
    ("help_autoscroll", "resume console autoscroll", "obnovit auto-scroll konzole", "obnoviť auto-rokovanie konzoly"),
    ("help_theme", "toggle theme", "přepnout téma", "prepnúť tému"),
    ("help_this", "this help", "tato nápověda", "táto pomoc"),
    ("help_quit", "quit", "ukončit", "ukončiť"),
    ("help_commands", "/help /clear /theme /copy /quit", "/help /clear /theme /copy /quit", "/help /clear /theme /copy /quit"),
    ("hint_lang", "↑ / k  ↓ / j  move · Enter pick · Esc quit", "↑ / k  ↓ / j  pohyb · Enter výběr · Esc konec", "↑ / k  ↓ / j  pohyb · Enter výber · Esc koniec"),
    ("hint_pick_more", "↑/k ↓/j · Enter · + folder · Esc quit", "↑/k ↓/j · Enter · + složka · Esc konec", "↑/k ↓/j · Enter · + priečinok · Esc koniec"),
    ("save_cancel", "Enter save · Esc cancel", "Enter uložit · Esc zrušit", "Enter uložiť · Esc zrušiť"),
    ("unknown_command", "unknown command:", "neznámý příkaz:", "neznámy príkaz:"),
    ("server_exited", "server exited (code", "server ukončen (kód", "server ukončený (kód"),
    ("vision_arrows", "Space toggles vision", "Mezerník přepíná VISION", "Medzerník prepína VISION"),
    ("vision_on", "This model supports VISION — Space toggles it ON", "Tento model podporuje VISION — Mezerník zapíná", "Tento model podporuje VISION — Medzerník zapína"),
    ("vision_off", "This model supports VISION — Space toggles it OFF", "Tento model podporuje VISION — Mezerník vypíná", "Tento model podporuje VISION — Medzerník vypína"),
    ("gpu_total", "GPU total", "GPU celkem", "GPU celkom"),
    ("back_hint", "Backspace/Esc = back", "Backspace/Esc = zpět", "Backspace/Esc = späť"),
    ("preset_safe", "Cruise Control", "Jízdní režim", "Jazdný režim"),
    ("preset_safe_desc", "Safe & steady", "Bezpečný a stabilní", "Bezpečný a stabilný"),
    ("preset_solo", "Solo Sprint", "Sólový sprint", "Sólový šprint"),
    ("preset_solo_desc", "Max single-user", "Max pro jednoho", "Max pre jedného"),
    ("preset_twin", "Twin Boost", "Dvojitý boost", "Dvojitý boost"),
    ("preset_twin_desc", "Two concurrent users", "Dva souběžní uživatelé", "Dvaja súbežní používatelia"),
    ("choose_card", "◀/▶ tweak · ↑/↓ focus · Space vision · Enter launch · Esc back", "◀/▶ nastavení · ↑/↓ fokus · Mezerník vize · Enter spustit · Esc zpět", "◀/▶ nastavenie · ↑/↓ fokus · Medzerník vízia · Enter spustiť · Esc späť"),
    ("qwen_hint", "Qwen 3: reasoning-effort only changes reasoning-token volume — tuning it fixes the Qwen-3 runaway-reasoning GitHub issue", "Qwen 3: reasoning-effort mění jen objem reasoning tokenů — nastavení řeší GitHub issue runaway-reasoning Qwen-3", "Qwen 3: reasoning-effort mení iba objem reasoning tokenov — nastavenie rieši GitHub issue runaway-reasoning Qwen-3"),
    ("help_back", "back to model picker (stop server)", "zpět na výběr modelu (zastavit server)", "späť na výber modelu (zastaviť server)"),
    ("help_new", "new model (stop server)", "nový model (zastavit server)", "nový model (zastaviť server)"),
    ("help_fn", "F1 help · F2 theme · F3 autoscroll · F4 new model · F5 back", "F1 nápověda · F2 téma · F3 autoscroll · F4 nový model · F5 zpět", "F1 pomoc · F2 téma · F3 autoscroll · F4 nový model · F5 späť"),
    ("help_esc", "close help / clear input", "zavřít nápovědu / smazat vstup", "zavrieť pomoc / vymazať vstup"),
    ("help_alt", "Alt+T theme · Alt+A autoscroll · Alt+N new model · Alt+H help · Alt+B back · Alt+Q quit", "Alt+T téma · Alt+A autoscroll · Alt+N nový model · Alt+H nápověda · Alt+B zpět · Alt+Q konec", "Alt+T téma · Alt+A autoscroll · Alt+N nový model · Alt+H pomoc · Alt+B späť · Alt+Q koniec"),
    ("tut_profile", "P = Profil: předvolba kvality / velikosti kontextu (Small/Balanced/Big)", "P = Profil: predvoľba kvality / veľkosti kontextu (Small/Balanced/Big)", "P = Profil: predvoľba kvality / veľkosti kontextu (Small/Balanced/Big)"),
    ("continue", "Continue", "Pokračovat", "Pokračovať"),
    ("launching", "launching llama-server", "spouštím llama-server", "spúšťam llama-server"),
    ("loading_model", "loading model", "načítám model", "načítavam model"),
    ("launch_hint", "Enter = launch", "Enter = spustit", "Enter = spustiť"),
    ("vision_prompt", "🖼  This model has vision (mmproj). Choose how to run it:", "🖼  Tento model má VISION (mmproj). Vyberte režim:", "🖼  Tento model má VISION (mmproj). Vyberte režim:"),
    ("with_vision", "With vision", "S VISION", "S VISION"),
    ("without_vision", "Without vision (larger context)", "Bez vize (větší kontext)", "Bez vízie (väcší kontext)"),
    ("profile_small", "Small / fast", "Malý / rychlý", "Malý / rýchly"),
    ("profile_balanced", "Balanced", "Vyvážený", "Vyvážený"),
    ("profile_big", "Big / 96k", "Velký / 96k", "Veľký / 96k"),
    // 10 added features: help, clear, history count, token count, theme, etc.
    ("help_title", "Help", "Nápověda", "Pomoc"),
    ("clear_chat", "Clear chat", "Vymazat chat", "Vymazať chat"),
    ("show_history", "History", "Historie", "História"),
    ("token_count", "tokens", "tokenů", "tokenov"),
    ("theme", "Theme", "Téma", "Téma"),
];

fn build() -> (Table, Table, Table) {
    let mut en = Table::new();
    let mut cs = Table::new();
    let mut sk = Table::new();
    for (k, e, c, s) in ENTRIES {
        en.insert(*k, *e);
        cs.insert(*k, *c);
        sk.insert(*k, *s);
    }
    (en, cs, sk)
}

pub struct I18n {
    lang: Lang,
    en: Table,
    cs: Table,
    sk: Table,
}

impl I18n {
    pub fn new(lang: Lang) -> Self {
        let (en, cs, sk) = build();
        Self { lang, en, cs, sk }
    }

    pub fn set_lang(&mut self, lang: Lang) {
        self.lang = lang;
    }

    pub fn lang(&self) -> Lang {
        self.lang
    }

    pub fn t(&self, key: &str) -> String {
        let table = match self.lang {
            Lang::En => &self.en,
            Lang::Cs => &self.cs,
            Lang::Sk => &self.sk,
        };
        table.get(key).map(|s| s.to_string()).unwrap_or_else(|| key.to_string())
    }
}

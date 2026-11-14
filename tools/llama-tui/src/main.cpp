// tools/kelvin-tui/src/main.cpp
//
// "The Lab" - Kelvin the Software Chemist's dependency-distilling TUI.
//
// FTXUI: only ONE interactive App loop per process. The whole wizard + chat
// runs in a single Loop over a Container::Tab of components; advancing uses
// MenuOption.on_enter, and no Loop() is ever re-entered. This is the crash-free
// architecture (sequential ScreenInteractive::Loop() calls crash after #2).

#include "version.h"
#include "model_detector.h"
#include "file_picker.h"
#include "server_proc.h"
#include "chat_client.h"
#include "presets.h"
#include "settings.h"
#include "i18n.h"
#include "tui_ui.h"

#include <windows.h>
#include <combaseapi.h>

#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <string>
#include <vector>
#include <memory>

#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>

static I18n g_i18n;
static std::string ts(const char* key) { return g_i18n.t(key); }
static std::string i18n_hint_str() { return "   " + ts("nav_hint"); }

static std::string exe_dir_string() {
    char self[MAX_PATH];
    GetModuleFileNameA(nullptr, self, MAX_PATH);
    std::string p(self);
    size_t pos = p.find_last_of("\\/");
    return pos == std::string::npos ? p : p.substr(0, pos);
}

static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ep) {
    FILE* f = fopen((exe_dir_string() + "\\crash.log").c_str(), "a");
    if (f) {
        fprintf(f, "EXCEPTION code=0x%08X addr=%p at %lld\n",
                ep->ExceptionRecord->ExceptionCode, (void*)ep->ExceptionRecord->ExceptionAddress,
                (long long)time(nullptr));
        fclose(f);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

// Simple leveled file logger with timestamps. Writes to kelvin-tui.log.
static void klog(const char* level, const char* fmt, ...) {
    char msg[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    FILE* f = fopen((exe_dir_string() + "\\kelvin-tui.log").c_str(), "a");
    if (f) {
        time_t t = time(nullptr);
        struct tm* lt = localtime(&t);
        fprintf(f, "[%02d:%02d:%02d][%s] %s\n", lt->tm_hour, lt->tm_min, lt->tm_sec, level, msg);
        fclose(f);
    }
}
static void trace(const char* where) { klog("TRACE", "%s", where); }
static void L(const char* fmt, ...) {
    char msg[1024]; va_list ap; va_start(ap,fmt); vsnprintf(msg,sizeof(msg),fmt,ap); va_end(ap);
    klog("INFO", "%s", msg);
}
static void KLOG_ERR(const char* fmt, ...) {
    char msg[1024]; va_list ap; va_start(ap,fmt); vsnprintf(msg,sizeof(msg),fmt,ap); va_end(ap);
    klog("ERROR", "%s", msg);
}

static std::string drive_root(const std::string& p) {
    std::string drive = "?";
    if (p.size() >= 2 && p[1] == ':') drive = p.substr(0, 2);
    std::string rest = p.substr(2);
    std::string root;
    size_t slash = rest.find_first_of("\\/");
    if (slash != std::string::npos) root = rest.substr(0, slash);
    return drive + "\\" + root;
}

// Check a file exists (Windows).
static bool file_exists(const std::string& p) {
    DWORD attr = GetFileAttributesA(p.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

using namespace ftxui;

// Stage decorators: title + nav hint + the menu's rendered element.
static Element language_decorator(Component& menu, int sel) {
    (void)sel;
    auto flag = [](Lang l) -> Element {
        switch (l) {
            case Lang::En: return hbox({ text("▌▌▌▌▌")|color(Color::White), text("▌▌▌")|color(Color::Blue) });
            case Lang::Cs: return hbox({ text("▌▌▌▌▌")|color(Color::White), text("▌▌▌")|color(Color::Red), text("▌▌▌")|color(Color::Blue) });
            case Lang::Sk: return hbox({ text("▌▌▌▌▌")|color(Color::White), text("▌▌▌")|color(Color::Blue), text("▌▌▌")|color(Color::Red) });
        }
        return text("");
    };
    return vbox({
        text("The Lab - choose language")|bold|color(Color::Yellow),
        text(" "),
        hbox({ flag(Lang::En), text("  English"), filler() }),
        hbox({ flag(Lang::Cs), text("  Čeština"), filler() }),
        hbox({ flag(Lang::Sk), text("  Slovenčina"), filler() }),
        text(" "),
        text(ts("nav_hint")) | dim,
        menu->Render(),
    }) | border;
}

static Element model_decorator(Component& menu) {
    return vbox({
        text(ts("choose_model"))|bold|color(Color::Yellow),
        text(i18n_hint_str()) | dim,
        text(""),
        menu->Render(),
    }) | border;
}

static Element preset_decorator(Component& menu) {
    return vbox({
        text(ts("choose_preset"))|bold|color(Color::Yellow),
        text(i18n_hint_str()) | dim,
        text(""),
        menu->Render(),
    }) | border;
}

int main() {
    SetUnhandledExceptionFilter(crash_handler);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    trace("main: start");

    // TEMP diagnostic: backtrace on allocation failure.
    std::set_new_handler([] {
        static volatile LONG guard = 0;
        if (InterlockedIncrement(&guard) != 1) return;
        void* f2[40]; int n2 = CaptureStackBackTrace(2, 40, f2, nullptr);
        FILE* f = fopen((exe_dir_string() + "\\crash.log").c_str(), "a");
        if (f) {
            fprintf(f, "BAD_ALLOC2 @ %lld\n", (long long)time(nullptr));
            for (int i = 0; i < n2; i++) {
                MEMORY_BASIC_INFORMATION mbi{};
                char mod[MAX_PATH]="?";
                if (VirtualQuery(f2[i], &mbi, sizeof(mbi)) && mbi.Type==MEM_IMAGE) {
                    DWORD l=GetModuleFileNameA((HMODULE)mbi.AllocationBase,mod,MAX_PATH);
                    if(!l||l>=MAX_PATH)mod[0]=0;
                    fprintf(f,"  [%02d] rva=%p %s\n", i,
                            (void*)((size_t)f2[i]-(size_t)mbi.AllocationBase), mod[0]?mod:"?");
                }
            }
            fclose(f);
        }
    });

    auto screen = App::Fullscreen();

    Lang detected = detect_lang();
    UserSettings saved;
    if (settings_load(saved)) {
        if (saved.lang == "cs") detected = Lang::Cs;
        else if (saved.lang == "sk") detected = Lang::Sk;
    }
    bool autopick = getenv("KELVIN_TUI_AUTOPICK") != nullptr;

    std::vector<DetectedModel> models = detect_models({});
    if (models.empty()) {
        const char* m = getenv("KELVIN_TUI_MODEL");
        if (m) { DetectedModel d; d.path = m; d.name = m; d.source = "env"; models.push_back(d); }
    }
    std::string server_exe = exe_dir_string() + "\\llama-server.exe";

    int model_preselect = -1, preset_preselect = 0;
    if (!saved.model_path.empty())
        for (int i = 0; i < (int)models.size(); i++)
            if (models[i].path == saved.model_path) model_preselect = i;
    if (!saved.preset_id.empty()) {
        const auto& cat = preset_catalog();
        for (int i = 0; i < (int)cat.size(); i++)
            if (cat[i].id == saved.preset_id) preset_preselect = i + 1;
    }

    // Stage index shared by closures; the root Container::Tab switches on it.
    int stage = 0; // 0=lang,1=model,2=preset,3=chat
    int lang_sel = 0, model_sel = 0, preset_sel = 0;

    const Lang lang_order[3] = { Lang::En, Lang::Cs, Lang::Sk };
    for (int i = 0; i < 3; i++) if (lang_order[i] == detected) lang_sel = i;

    // --- Language menu ---
    std::vector<std::string> lang_names = { "English", lang_name(Lang::Cs), lang_name(Lang::Sk) };
    MenuOption lang_opt;
    lang_opt.on_enter = [&] { g_i18n.set(lang_order[lang_sel]); stage = 1; };
    auto lang_menu = Menu(&lang_names, &lang_sel, lang_opt);

    // --- Model menu ---
    std::vector<std::string> model_entries;
    for (const auto& m : models) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "[%.1f GB] %s  (%s  %s)",
                      m.size_gb, m.name.c_str(), m.source.c_str(), drive_root(m.path).c_str());
        model_entries.push_back(buf);
    }
    model_entries.push_back(ts("browse_folder"));
    MenuOption model_opt;
    model_opt.on_enter = [&] { stage = 2; };
    auto model_menu = Menu(&model_entries, &model_sel, model_opt);

    // --- Preset menu ---
    const ServerPreset& auto_preset = autopilot_preset(false, 0);
    std::vector<std::string> preset_entries;
    preset_entries.push_back("[Auto] " + auto_preset.label + " - " + auto_preset.description
                             + "  (" + ts("auto_recommended") + ")");
    const auto& cat = preset_catalog();
    for (const auto& p : cat) preset_entries.push_back(p.label + " - " + p.description);
    // Forward-declared; assigned below after the server/chat block is built.
    std::function<void(DetectedModel&, const ServerPreset&)> start_server;
    MenuOption preset_opt;
    preset_opt.on_enter = [&] {
        DetectedModel model;
        if (model_sel >= 0 && model_sel < (int)models.size()) model = models[model_sel];
        const ServerPreset* chosen = (preset_sel == 0) ? &auto_preset : &cat[preset_sel - 1];
        start_server(model, *chosen);
        stage = 3;
    };
    auto preset_menu = Menu(&preset_entries, &preset_sel, preset_opt);

    // --- Chat stage: lazy server launch on first entry + TuiUi chat ---
    auto chat_ui = std::make_shared<TuiUi>();
    chat_ui->set_i18n(&g_i18n);
    ChatClient chat_client(8888);
    std::vector<ChatMessage> history;
    chat_ui->set_on_send([&](const std::string& prompt) {
        chat_ui->set_status(ts("status_thinking"));
        ChatCallbacks cb;
        cb.on_delta = [&](const std::string& delta) { chat_ui->append_assistant(delta); };
        cb.on_done = [&](bool ok, const std::string& err) {
            if (ok) {
                const std::string& mn = model_sel < (int)models.size() ? models[model_sel].name : "?";
                chat_ui->set_status(ts("status_ready") + " - " + mn);
            } else {
                chat_ui->set_status(ts("status_error") + ": " + err);
            }
            history.push_back({"user", prompt});
        };
        chat_client.stream(history, prompt, cb);
    });

    bool server_started = false;
    bool server_requested = false;
    std::shared_ptr<ServerProcess> server_proc = std::make_shared<ServerProcess>();

    // Async callbacks from the server subprocess worker threads. These only set
    // up a task on the FTXUI loop thread (App::Post is thread-safe); all UI
    // state mutation happens on the loop thread, never inside Render().
    server_proc->on_log = [&](std::string line) {
        screen.Post([chat_ui, line = std::move(line)]() mutable {
            chat_ui->append_server_log(std::move(line));
            if (auto* a = App::Active()) a->RequestAnimationFrame();
        });
    };
    server_proc->on_health = [&](bool healthy) {
        screen.Post([&, healthy]() {
            if (healthy && server_requested && !server_started) {
                server_started = true;
                const std::string& mn = model_sel < (int)models.size() ? models[model_sel].name : "?";
                chat_ui->set_status(ts("status_ready") + " - " + mn);
                fprintf(stderr, "[The Lab] server ready\n");
            }
            if (auto* a = App::Active()) a->RequestAnimationFrame();
        });
    };

    start_server = [&](DetectedModel& model, const ServerPreset& preset) {
        if (server_requested) return;
        // Persist the selection IMMEDIATELY, regardless of server outcome.
        {
            UserSettings s;
            s.model_path = model.path;
            s.preset_id = (preset.id == auto_preset.id) ? "" : preset.id;
            s.parallel = "0";
            s.lang = lang_code(g_i18n.lang());
            settings_save(s);
        }
        if (model.path.empty()) {
            chat_ui->set_status(ts("status_error") + ": no model selected");
            return;
        }
        if (!file_exists(server_exe)) {
            chat_ui->set_status(ts("status_error") + ": llama-server.exe not found next to this app");
            return;
        }
        // Spawn llama-server; worker threads feed the UI async. Non-blocking.
        ServerSpec spec;
        spec.exe_path = server_exe;
        spec.model_path = model.path;
        spec.port = 8888;
        spec.log_path = exe_dir_string() + "\\llama-server.log";
        spec.preset = &preset;
        chat_ui->set_status(ts("status_thinking"));  // "starting llama-server..."
        if (!server_proc->start(spec)) {
            chat_ui->set_status(ts("status_error") + ": " + server_proc->last_error());
            return;
        }
        server_requested = true;
        fprintf(stderr, "[The Lab] llama-server started\n");
    };

    // Chat component (built lazily once). Server log/status arrive via the
    // App::Post tasks above, so Render() never touches cross-thread state.
    auto chat_component = chat_ui->build_component(screen);

    // Root: a Tab of the four stage components; 'stage' controls active tab.
    // Tab gives keyboard to the active child only.
    std::vector<Component> tabs = { lang_menu, model_menu, preset_menu, chat_component };
    auto root = Container::Tab(tabs, &stage);

    // TEMP headless auto-spawn (removed from release).
    if (getenv("KELVIN_TUI_FORCE_CHAT") != nullptr) {
        trace("main: force-chat auto-spawn");
        DetectedModel fm;
        if (!models.empty()) fm = models[0];
        fm.path = saved.model_path.empty() ? fm.path : saved.model_path;
        if (getenv("KELVIN_TUI_NO_SPAWN") == nullptr) {
            start_server(fm, auto_preset);
        }
        stage = 3;
    }

    // Wrap in a header/footer renderer.
    auto decorated = Renderer(root, [&] {
        switch (stage) {
            case 0: return language_decorator(lang_menu, lang_sel);
            case 1: return model_decorator(model_menu);
            case 2: return preset_decorator(preset_menu);
            default: return chat_component->Render();
        }
    });

    // Global quit + back navigation (server launch happens in preset on_enter).
    auto catcher = CatchEvent(decorated, [&](Event e) {
        if (e == Event::Character('q') && stage == 3) { server_proc->stop(); screen.Exit(); return true; }
        if (e == Event::Escape && stage > 0) { stage--; return true; }
        return false;
    });

    trace("main: loop");
    try {
        screen.Loop(catcher);
        trace("main: loop exited normally");
    } catch (const std::exception& e) {
        KLOG_ERR("FTXUI loop threw: %s", e.what());
        trace("main: loop threw C++ exception");
    } catch (...) {
        KLOG_ERR("FTXUI loop threw unknown exception");
        trace("main: loop threw unknown");
    }

    if (server_started) server_proc->stop();
    CoUninitialize();
    return 0;
}
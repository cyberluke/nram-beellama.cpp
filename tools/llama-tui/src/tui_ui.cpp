// tools/kelvin-tui/src/tui_ui.cpp
#include "tui_ui.h"

#include <ftxui/component/component_options.hpp>
#include <ftxui/screen/screen.hpp>

using namespace ftxui;

namespace {
Element MakeStatusBar(const std::string& status, const std::string& hint) {
    return hbox({
        text(" "),
        text(status) | color(Color::Green) | bold,
        filler(),
        text("  " + hint + "  ") | dim,
    }) | borderLight;
}

Element MakeSplash(const I18n* i18n) {
    // ASCII-art: "The Lab" - a bubbling chemist flask.
    std::string title = i18n ? i18n->t("app.title") : "The Lab";
    std::string chemist = i18n ? i18n->t("app.tagline") : "Kelvin the Software Chemist";
    std::string tagline = i18n ? i18n->t("app.subtagline") : "purifying dependency trees to 99.1%";
    std::string wait = i18n ? i18n->t("waiting_message") : "(no messages yet)";
    return vbox({
        text(""),
        text("      _  _     __"),
        text("    _| || |_  / /"),
        text("   |_  ..  _|/ / "),
        text("   |_      _|/  "),
        text("     |_||_|"),
        text(""),
        text("   " + title) | bold | color(Color::Yellow),
        text("   " + chemist) | color(Color::Cyan),
        text("   " + tagline) | dim,
        filler(),
        text("   " + wait) | dim,
    });
}
} // namespace

void TuiUi::set_status(const std::string& text) { status_ = text; }

void TuiUi::append_assistant(const std::string& delta) {
    if (delta.empty()) return;
    std::string line;
    for (char c : delta) {
        if (c == '\n') { lines_.push_back(line); line.clear(); }
        else line += c;
    }
    if (!line.empty()) lines_.push_back(line);
}

void TuiUi::append_server_log(std::string line) {
    // Bounded render buffer: only keep the most recent lines.
    server_log_.push_back(std::move(line));
    const size_t kKeep = 2000;
    if (server_log_.size() > kKeep)
        server_log_.erase(server_log_.begin(), server_log_.begin() + (server_log_.size() - kKeep));
}

ftxui::Component TuiUi::build_component(ScreenInteractive& screen) {
    std::string hint = i18n_ ? i18n_->t("status_hint") : "[Enter]=send  [Esc]=quit";
    std::string placeholder = i18n_ ? i18n_->t("input_placeholder") : "Message Kelvin the Software Chemist...";

    auto chat_viewport = Renderer([&] {
        std::vector<Element> els;
        els.push_back(MakeSplash(i18n_));
        if (lines_.empty()) {
            els.push_back(text(i18n_ ? i18n_->t("waiting_message") : "(no messages yet)") | dim);
        }
        for (const auto& l : lines_) {
            els.push_back(paragraph(l));
        }
        return vbox(std::move(els)) | frame | flex;
    });

    auto input = Input(&input_, &placeholder);
    input |= CatchEvent([&](Event e) {
        if (e == Event::Return) {
            if (!input_.empty()) {
                std::string msg = input_;
                input_ = "";
                lines_.push_back("You: " + msg);
                if (on_send_) on_send_(msg);
            }
            return true;
        }
        if (e == Event::Escape) { screen.Exit(); return true; }
        return false;
    });

    auto status_bar = Renderer([&] { return MakeStatusBar(status_, hint); });

    // Scrollable llama-server log pane. Drains the captured log provider each
    // frame so it stays live. NOTE: must be a vbox of individual text elements
    // (ftxui::text() with embedded \n crashes), and only ONE element flexes so
    // the panes don't fight over available height.
    auto log_pane = Renderer([&] {
        std::vector<Element> els;
        {
            size_t start = server_log_.size() > 200 ? server_log_.size() - 200 : 0;
            for (size_t i = start; i < server_log_.size(); i++) {
                // Sanitize: strip ANSI escapes and control chars (llama emits
                // color codes that ftxui::text can't render and would crash on).
                // Also cap each rendered line so a pathological (very long or
                // unterminated) llama output line can never blow up text().
                const std::string& raw = server_log_[i];
                std::string clean;
                const size_t cap = 2000;
                clean.reserve(raw.size() < cap ? raw.size() : cap);
                bool in_esc = false;
                for (char c : raw) {
                    if (clean.size() >= cap) break;
                    if (c == '\x1b') { in_esc = true; continue; }
                    if (in_esc) {
                        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) in_esc = false;
                        continue;
                    }
                    if ((unsigned char)c < 0x20 && c != '\t') continue;
                    clean += c;
                }
                els.push_back(text(clean.empty() ? " " : clean) | color(Color::GrayDark));
            }
        }
        if (els.empty()) els.push_back(text("(server starting...)") | color(Color::GrayDark) | dim);
        return vbox(std::move(els)) | frame;   // no flex: bounded pane
    });

    // NOTE: capture the shared_ptr<Component>s BY VALUE so they stay alive
    // after this function returns. Capturing them by reference (`[&]`) leaves
    // dangling references to the locals, causing a use-after-free when FTXUI
    // renders (crash in ComponentBase::OnRender / Render).
    auto chat = chat_viewport;
    auto lpan = log_pane;
    auto sbar = status_bar;
    auto inpt = input;

    return Renderer(input, [chat, lpan, sbar, inpt] {
        return vbox({
            chat->Render() | flex,              // chat takes surplus (flex)
            lpan->Render() | borderLight,       // llama log (bounded height)
            sbar->Render(),
            inpt->Render() | borderLight,       // input (bottom)
        });
    });
}
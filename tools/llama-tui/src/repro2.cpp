// Definitive repro: exact kelvin-tui chat staging under a Container::Tab
// with MakeSplash/MakeStatusBar/i18n + CatchEvent. No server, no threads.
#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
using namespace ftxui;

static Element MakeStatusBar(const std::string& status, const std::string& hint) {
    return hbox({
        text(" "),
        text(status) | color(Color::Green) | bold,
        filler(),
        text("  " + hint + "  ") | dim,
    }) | borderLight;
}

static Element MakeSplash() {
    std::string title = "The Lab";
    std::string chemist = "Kelvin the Software Chemist";
    std::string tagline = "purifying dependency trees to 99.1%";
    std::string wait = "(no messages yet)";
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

int main() {
    printf("repro2: start\n"); fflush(stdout);
    auto screen = App::Fullscreen();
    std::string input_;
    std::string status_ = "The Lab - starting...";
    std::vector<std::string> server_log_;
    std::vector<std::string> lines_;
    std::function<void(const std::string&)> on_send;

    auto chat_viewport = Renderer([&] {
        std::vector<Element> els;
        els.push_back(MakeSplash());
        if (lines_.empty()) els.push_back(text("(no messages yet)") | dim);
        for (const auto& l : lines_) els.push_back(paragraph(l));
        return vbox(std::move(els)) | frame | flex;
    });

    auto input = Input(&input_, "msg");
    input |= CatchEvent([&](Event e) {
        if (e == Event::Return) {
            if (!input_.empty()) { std::string m = input_; input_ = ""; lines_.push_back("You: " + m); if(on_send) on_send(m); }
            return true;
        }
        if (e == Event::Escape) { screen.Exit(); return true; }
        return false;
    });

    auto status_bar = Renderer([&] { return MakeStatusBar(status_, "[Enter]=send  [Esc]=quit"); });

    auto log_pane = Renderer([&] {
        std::vector<Element> els;
        size_t start = server_log_.size() > 200 ? server_log_.size() - 200 : 0;
        for (size_t i = start; i < server_log_.size(); i++) {
            std::string line = server_log_[i];
            std::string clean;
            const size_t cap = 2000; clean.reserve(line.size()<cap?line.size():cap);
            bool in_esc=false;
            for(char c:line){ if(clean.size()>=cap)break; if(c=='\x1b'){in_esc=true;continue;} if(in_esc){ if((c>='A'&&c<='Z')||(c>='a'&&c<='z'))in_esc=false; continue;} if((unsigned char)c<0x20 && c!='\t')continue; clean+=c; }
            els.push_back(text(clean.empty()?" ":clean) | color(Color::GrayDark));
        }
        if (els.empty()) els.push_back(text("(server starting...)") | color(Color::GrayDark) | dim);
        return vbox(std::move(els)) | frame;
    });

    auto chat = chat_viewport, lpan = log_pane, sbar = status_bar, inpt = input;
    auto chat_component = Renderer(input, [chat,lpan,sbar,inpt] {
        return vbox({ chat->Render()|flex, lpan->Render()|borderLight, sbar->Render(), inpt->Render()|borderLight });
    });

    int stage = 0;
    auto lang_menu = Renderer([&]{ return text("lang"); });
    auto model_menu = Renderer([&]{ return text("model"); });
    auto preset_menu = Renderer([&]{ return text("preset"); });
    std::vector<Component> tabs = { lang_menu, model_menu, preset_menu, chat_component };
    auto root = Container::Tab(tabs, &stage);
    stage = 3; // chat

    auto decorated = Renderer(root, [&] {
        switch (stage) {
            case 0: return text("lang");
            case 1: return text("model");
            case 2: return text("preset");
            default: return chat_component->Render();
        }
    });

    printf("repro2: loop\n"); fflush(stdout);
    try {
        screen.Loop(decorated);
        printf("repro2: loop returned\n"); fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        printf("repro2: THREW %s\n", e.what()); fflush(stdout);
        return 7;
    }
}
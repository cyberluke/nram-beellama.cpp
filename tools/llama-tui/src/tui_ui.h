// tools/kelvin-tui/src/tui_ui.h
#pragma once

#include "chat_client.h"
#include "i18n.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

// The FTXUI-based chat UI. Owns its rendering + event loop entry point.
class TuiUi {
public:
    using SendFn = std::function<void(const std::string& prompt)>;

    // Create the UI.
    explicit TuiUi() = default;

    // Set the send callback (caller wires streaming after construction).
    void set_on_send(SendFn on_send) { on_send_ = std::move(on_send); }

    // Append assistant text (streamed) to the chat scrollback.
    void append_assistant(const std::string& delta);

    // Append one server log line to the pane. MUST be called on the FTXUI
    // loop thread (e.g. from inside a task posted via App::Post()), so no
    // cross-thread access of UI state ever happens during Render().
    void append_server_log(std::string line);

    // Set the status-bar text (model, server state, token info).
    void set_status(const std::string& text);

    // Diagnostic: current status text.
    std::string debug_status() const { return status_; }

    // Inject localization (translated placeholder/splash/status-hint).
    void set_i18n(const I18n* i18n) { i18n_ = i18n; }

    // Build the chat Component (to be placed inside a shared screen/loop).
    ftxui::Component build_component(ftxui::ScreenInteractive& screen);

private:
    SendFn                       on_send_;
    std::vector<std::string>     server_log_;    // appended log lines for pane
    std::vector<std::string>     lines_;          // rendered chat text lines
    std::vector<ftxui::Element>  chat_lines_;     // reserved (unused)
    std::string                  input_;
    std::string                  status_ = "The Lab - starting...";
    bool                         live_request_ = false;
    bool                         quit_ = false;
    const I18n* i18n_ = nullptr;
};
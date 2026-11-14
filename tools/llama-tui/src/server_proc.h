// tools/kelvin-tui/src/server_proc.h
#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <functional>
#include <windows.h>

struct ServerPreset;

// Owns a llama-server subprocess spawned with no window, capturing its stdout/
// stderr through a pipe into an in-memory log buffer (and optionallly a file).
struct ServerSpec {
    std::string exe_path;      // absolute path to llama-server.exe
    std::string model_path;    // absolute path to the GGUF
    std::string log_path;      // also append output here ("" = none)
    int         port = 8888;
    int         gpu_layers = 999;
    int         ctx_size = 163840;
    int         parallel = 1;
    std::string extra_args;    // raw extra server args
    const ServerPreset* preset = nullptr;
};

// Async data flow (FTXUI loop thread): the reader/health worker threads do the
// I/O and hand completed results to the UI via these callbacks. The caller
// MUST marshal them onto the FTXUI loop thread (e.g. App::Post()). This keeps
// all rendering/mutation of UI state on the loop thread and avoids any
// cross-thread read during Render().
class ServerProcess {
public:
    using LogCallback  = std::function<void(std::string line)>;
    using HealthCallback = std::function<void(bool healthy)>;

    ~ServerProcess() { stop(); }

    // Spawn llama-server (CREATE_NO_WINDOW, inherited pipes). Returns false on
    // spawn failure. Non-blocking.
    bool start(const ServerSpec& spec);

    // Non-blocking: latest known /health state (updated by health worker).
    bool is_healthy();

    // Terminate child + reap worker threads.
    void stop();

    std::string last_error() const { return last_error_; }

    // Async sinks. Set before start(); invoked from worker threads with partial
    // log lines and health transitions. Wrap in App::Post to update the UI.
    LogCallback    on_log;
    HealthCallback on_health;

    // Latest captured whole log (bounded); for diagnostics / crash dumps only.
    std::string full_log() const;

private:
    std::string build_command_line(const ServerSpec& spec) const;
    static DWORD WINAPI reader_thread(LPVOID param);
    static DWORD WINAPI health_poll_thread(LPVOID param);

    PROCESS_INFORMATION proc_{};
    STARTUPINFOA        si_{};
    HANDLE              read_pipe_ = nullptr;   // parent end (we read)
    HANDLE              child_write_ = nullptr; // child end (inherited)
    HANDLE              job_ = nullptr;         // kill-on-close Job for the child

    // Background /health poller (started in start(), joined in stop()).
    HANDLE              health_thread_ = nullptr;
    std::atomic<bool>   health_stop_{false};
    std::atomic<bool>   healthy_{false};

    // Bounded in-memory log mirror (worker thread writes, mutex-guarded).
    mutable std::mutex log_mutex_;
    std::deque<std::string> log_lines_;
    std::string             log_path_;

    std::string last_error_;
    std::atomic<bool> thread_done_{false};
    HANDLE reader_handle_ = nullptr;
};
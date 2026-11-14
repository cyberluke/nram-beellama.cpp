// tools/kelvin-tui/src/server_proc.cpp
#include "server_proc.h"
#include "http_client.h"
#include "presets.h"

#include <cstdio>
#include <cstdarg>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>

// Tiny file logger (duplicate here to keep server_proc self-contained).
namespace {
void sp_log(const char* fmt, ...) {
    char msg[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string exe(path);
    std::string dir = exe.substr(0, exe.find_last_of("\\/"));
    FILE* f = fopen((dir + "\\kelvin-tui.log").c_str(), "a");
    if (f) { fprintf(f, "[SP] %s\n", msg); fclose(f); }
}
}

std::string ServerProcess::build_command_line(const ServerSpec& spec) const {
    const ServerPreset* p = spec.preset;
    std::ostringstream cmd;
    cmd << "\"" << spec.exe_path << "\""
        << " --model \"" << spec.model_path << "\""
        << " --host 127.0.0.1 --port " << spec.port
        << " --ctx-size " << (p ? p->ctx_size : spec.ctx_size)
        << " --n-gpu-layers " << (p ? p->gpu_layers : spec.gpu_layers)
        << " --parallel " << (p ? p->parallel : spec.parallel);

    if (p) {
        // KV + unified pool + flash attention (fixed ctx, no auto-fit).
        cmd << " --flash-attn on --fit off --kv-unified"
            << " --cache-type-k " << p->cache_k
            << " --cache-type-v " << p->cache_v
            << " --n-predict " << p->n_predict
            // Reasoning tuned so it does not kill the context (github issue).
            << " --reasoning " << p->reasoning
            << " --reasoning-effort " << p->reasoning_effort
            << " --reasoning-budget " << p->reasoning_budget
            << " --reasoning-budget-message \"Let's final\""
            << (p->preserve_reasoning ? " --reasoning-preserve" : " --no-reasoning-preserve")
            // Sampling.
            << " --temp " << p->temp
            << " --top-p " << p->top_p
            << " --top-k " << p->top_k
            << " --min-p " << p->min_p
            // Performance.
            << " --batch-size " << p->batch_size
            << " --ubatch-size " << p->ubatch_size
            << " --threads " << p->threads
            << (p->cont_batching ? " --cont-batching" : "")
            << " --metrics --slots";
        // Speculative / MTP.
        if (p->mtp) {
            cmd << " --spec-type draft-mtp"
                << " --spec-draft-n-max " << p->mtp_draft_max
                << " --spec-draft-type-k " << p->spec_cache_k
                << " --spec-draft-type-v " << p->spec_cache_v;
        } else {
            cmd << " --spec-draft-n-max 4";
        }
        // Prompt caching.
        if (p->cache_prompt) cmd << " --cache-prompt --cache-idle-slots";
    }
    if (!spec.extra_args.empty()) {
        cmd << " " << spec.extra_args;
    }
    return cmd.str();
}

bool ServerProcess::start(const ServerSpec& spec) {
    if (proc_.hProcess) {
        stop();
    }

    std::string cmd = build_command_line(spec);
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');
    sp_log("start: cmdline: %s", cmd.c_str());

    log_path_ = spec.log_path;

    // Clean Windows SDK: spawn with std handles connected to pipes we own;
    // CREATE_NO_WINDOW so no extra console disrupts FTXUI. A reader thread
    // drains the pipe into the log buffer (and optional file).
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE out_r = NULL, out_w = NULL;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
        last_error_ = "CreatePipe failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0); // parent read end not inherited

    ZeroMemory(&si_, sizeof(si_));
    si_.cb         = sizeof(si_);
    si_.hStdOutput = out_w;
    si_.hStdError  = out_w;
    si_.dwFlags    = STARTF_USESTDHANDLES;

    ZeroMemory(&proc_, sizeof(proc_));
    if (!CreateProcessA(
            nullptr, cmd_buf.data(), nullptr, nullptr, /*inheritHandles=*/TRUE,
            CREATE_NO_WINDOW, nullptr, nullptr, &si_, &proc_)) {
        last_error_ = "CreateProcessA failed (" + std::to_string(GetLastError()) + ")";
        CloseHandle(out_r);
        CloseHandle(out_w);
        return false;
    }
    CloseHandle(out_w);       // child has its own inherited copy
    read_pipe_   = out_r;
    thread_done_ = false;

    // OWNERSHIP: place llama-server in a Job with KILL_ON_JOB_CLOSE so it dies
    // automatically the moment kelvin-tui's job handle closes (normal exit OR
    // crash). This prevents orphaned llama-server instances piling up.
    job_ = CreateJobObjectA(nullptr, nullptr);
    if (job_) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof(li));
        AssignProcessToJobObject(job_, proc_.hProcess);
    }

    // Start a background /health poller so the FTXUI render thread never blocks
    // on a still-loading server.
    healthy_.store(false);
    health_stop_.store(false);
    health_thread_ = CreateThread(nullptr, 0, &ServerProcess::health_poll_thread, this, 0, nullptr);

    reader_handle_ = CreateThread(nullptr, 0, &ServerProcess::reader_thread, this, 0, nullptr);
    sp_log("start: spawned llama-server (pid=%lu) in job", proc_.dwProcessId);
    return true;
}

// Background poller: refresh the healthy_ flag every ~500ms. Bounded HTTP
// timeouts keep each probe short. Emits on_health on every transition so the
// UI can react lazily instead of polling.
DWORD WINAPI ServerProcess::health_poll_thread(LPVOID param) {
    ServerProcess* self = static_cast<ServerProcess*>(param);
    bool last = false;
    try {
        while (!self->health_stop_.load()) {
            if (!self->proc_.hProcess) break;
            HttpClient probe;
            std::string out;
            bool now = probe.get("http://127.0.0.1:" + std::to_string(8888) + "/health", out) &&
                       out.find("\"status\":\"ok\"") != std::string::npos;
            self->healthy_.store(now);
            if (now != last) {
                last = now;
                if (self->on_health) {
                    try { self->on_health(now); }
                    catch (...) { sp_log("on_health callback threw"); }
                }
            }
            for (int i = 0; i < 5 && !self->health_stop_.load(); i++)
                Sleep(100);
        }
    } catch (const std::exception& ex) {
        sp_log("health_poll_thread exception: %s", ex.what());
    } catch (...) {
        sp_log("health_poll_thread unknown exception");
    }
    return 0;
}

// Reader thread: drain llama output into the log buffer (and optional file).
DWORD WINAPI ServerProcess::reader_thread(LPVOID param) {
    ServerProcess* self = static_cast<ServerProcess*>(param);
    char buf[8192];
    DWORD n = 0;
    FILE* f = nullptr;
    if (!self->log_path_.empty()) { if (fopen_s(&f, self->log_path_.c_str(), "a+") != 0) f = nullptr; }

    std::string pending;
    try {
        while (self->read_pipe_ && ReadFile(self->read_pipe_, buf, sizeof(buf), &n, NULL) && n > 0) {
            if (f) { fwrite(buf, 1, n, f); fflush(f); }
            pending.append(buf, n);
            size_t nl;
            while ((nl = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                pending.erase(0, nl + 1);
                {
                    std::lock_guard<std::mutex> lock(self->log_mutex_);
                    self->log_lines_.push_back(std::move(line));
                    // Keep the in-memory log bounded (drop oldest).
                    while (self->log_lines_.size() > 2000)
                        self->log_lines_.pop_front();
                }
                if (self->on_log) {
                    try { self->on_log(self->log_lines_.back()); }
                    catch (...) { sp_log("on_log callback threw"); }
                }
            }
        }
    } catch (const std::exception& ex) {
        sp_log("reader_thread exception: %s", ex.what());
    } catch (...) {
        sp_log("reader_thread unknown exception");
    }
    if (!pending.empty()) {
        std::lock_guard<std::mutex> lock(self->log_mutex_);
        self->log_lines_.push_back(std::move(pending));
        if (self->on_log && !self->log_lines_.empty()) {
            try { self->on_log(self->log_lines_.back()); }
            catch (...) {}
        }
    }
    if (f) fclose(f);
    DWORD ex = 0;
    bool exited = self->proc_.hProcess &&
                  GetExitCodeProcess(self->proc_.hProcess, &ex) && ex != STILL_ACTIVE;
    sp_log("reader_thread finished (llama exited=%d code=%lu)", exited, ex);
    self->thread_done_ = true;
    return 0;
}

bool ServerProcess::is_healthy() {
    return healthy_.load();
}

void ServerProcess::stop() {
    // Stop the health poller first so it doesn't probe a dying child.
    health_stop_.store(true);
    if (health_thread_) {
        WaitForSingleObject(health_thread_, 2000);
        CloseHandle(health_thread_);
        health_thread_ = nullptr;
    }

    // Closing the Job handle kills the child (KILL_ON_JOB_CLOSE) if still alive.
    if (job_) { CloseHandle(job_); job_ = nullptr; }
    if (proc_.hProcess) {
        TerminateProcess(proc_.hProcess, 0);
        WaitForSingleObject(proc_.hProcess, 3000);
        CloseHandle(proc_.hProcess);
        CloseHandle(proc_.hThread);
        ZeroMemory(&proc_, sizeof(proc_));
    }
    // Closing the read pipe unblocks ReadFile in the reader thread.
    if (read_pipe_) { CloseHandle(read_pipe_); read_pipe_ = nullptr; }
    if (reader_handle_) {
        WaitForSingleObject(reader_handle_, 500);
        CloseHandle(reader_handle_);
        reader_handle_ = nullptr;
    }
}

std::string ServerProcess::full_log() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::string s;
    for (const auto& l : log_lines_) { s += l; s += '\n'; }
    return s;
}
// tools/kelvin-tui/src/http_client.h
#pragma once

#include <string>
#include <functional>

// Minimal WinHTTP-based HTTPS/HTTP client. No external deps.

// Callback signature for streaming bodies: receives body chunks as they arrive.
// Return false from the callback to abort the transfer.
typedef std::function<bool(const std::string&)> BodySink;

struct HttpResult {
    long   status = 0;
    std::string body;   // accumulated body (when not streaming to a sink)
    std::string error;
    bool ok() const { return status >= 200 && status < 300; }
};

class HttpClient {
public:
    // Simple GET; collects body into `out`.
    bool get(const std::string& url, std::string& out);

    // POST JSON; streams response body chunks to `sink` as they arrive (SSE).
    // Headers is newline-terminated extra headers (e.g. "Content-Type: application/json\r\n").
    bool post(const std::string& url, const std::string& body,
              const std::string& headers, const BodySink& sink, HttpResult* result);
};
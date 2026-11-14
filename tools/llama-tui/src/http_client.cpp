// tools/kelvin-tui/src/http_client.cpp
#include "http_client.h"

#include <windows.h>
#include <winhttp.h>

#include <sstream>

// Split "http://host:port/path" into scheme-free components for WinHTTP connect.
static bool parse_url(const std::string& url, std::string& host, int& port,
                      std::string& path, bool& is_https) {
    std::string u = url;
    is_https = u.rfind("https://", 0) == 0;
    if (is_https) u.erase(0, 8);
    else if (u.rfind("http://", 0) == 0) u.erase(0, 7);
    else return false;

    size_t slash = u.find('/');
    std::string authority = (slash == std::string::npos) ? u : u.substr(0, slash);
    path = (slash == std::string::npos) ? "/" : u.substr(slash);

    // Default port
    port = is_https ? 443 : 80;
    size_t colon = authority.rfind(':');
    if (colon != std::string::npos) {
        try { port = std::stoi(authority.substr(colon + 1)); } catch (...) { return false; }
        authority = authority.substr(0, colon);
    }
    host = authority;
    return !host.empty();
}

bool HttpClient::get(const std::string& url, std::string& out) {
    std::string host; int port; std::string path; bool https;
    if (!parse_url(url, host, port, path, https)) return false;

    HINTERNET session = WinHttpOpen(L"kelvin-tui/0.1",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return false;

    // Bounded timeouts: a non-responsive peer (e.g. llama-server still loading a
    // large model) must never block the caller indefinitely.
    DWORD t_resolve = 1000, t_connect = 1000, t_send = 1000, t_receive = 1000;
    WinHttpSetTimeouts(session, t_resolve, t_connect, t_send, t_receive);

    std::wstring whost(host.begin(), host.end());
    HINTERNET conn = WinHttpConnect(session, whost.c_str(), (INTERNET_PORT)port, 0);
    if (!conn) { WinHttpCloseHandle(session); return false; }

    std::wstring wpath(path.begin(), path.end());
    HINTERNET req = WinHttpOpenRequest(conn, L"GET", wpath.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        https ? WINHTTP_FLAG_SECURE : 0);
    if (!req) { WinHttpCloseHandle(conn); WinHttpCloseHandle(session); return false; }

    BOOL sent = WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!sent) {
        DWORD e = GetLastError();
        if (e == ERROR_WINHTTP_TIMEOUT) {
            WinHttpCloseHandle(req); WinHttpCloseHandle(conn); WinHttpCloseHandle(session);
            return false;
        }
    }
    BOOL recv = sent && WinHttpReceiveResponse(req, nullptr);
    if (sent && !recv && GetLastError() == ERROR_WINHTTP_TIMEOUT) {
        WinHttpCloseHandle(req); WinHttpCloseHandle(conn); WinHttpCloseHandle(session);
        return false;
    }
    bool ok = sent && recv;

    if (ok) {
        DWORD status = 0; DWORD status_len = sizeof(status);
        WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_len, WINHTTP_NO_HEADER_INDEX);
        if (status >= 400) ok = false;

        DWORD available = 0;
        while (ok && WinHttpQueryDataAvailable(req, &available) && available > 0) {
            std::string chunk(available, '\0');
            DWORD read = 0;
            if (!WinHttpReadData(req, &chunk[0], available, &read)) break;
            if (read > 0) { chunk.resize(read); out += chunk; }
        }
    }

    WinHttpCloseHandle(req);
    WinHttpCloseHandle(conn);
    WinHttpCloseHandle(session);
    return ok;
}

bool HttpClient::post(const std::string& url, const std::string& body,
                      const std::string& headers, const BodySink& sink, HttpResult* result) {
    std::string host; int port; std::string path; bool https;
    if (!parse_url(url, host, port, path, https)) return false;

    HINTERNET session = WinHttpOpen(L"kelvin-tui/0.1",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return false;

    // Bounded timeouts so a stalled server never blocks streaming indefinitely.
    DWORD t_resolve = 3000, t_connect = 3000, t_send = 3000, t_receive = 3000;
    WinHttpSetTimeouts(session, t_resolve, t_connect, t_send, t_receive);

    std::wstring whost(host.begin(), host.end());
    HINTERNET conn = WinHttpConnect(session, whost.c_str(), (INTERNET_PORT)port, 0);
    if (!conn) { WinHttpCloseHandle(session); return false; }

    std::wstring wpath(path.begin(), path.end());
    HINTERNET req = WinHttpOpenRequest(conn, L"POST", wpath.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        https ? WINHTTP_FLAG_SECURE : 0);
    if (!req) { WinHttpCloseHandle(conn); WinHttpCloseHandle(session); return false; }

    std::wstring wheaders(headers.begin(), headers.end());
    BOOL sent = WinHttpSendRequest(req, wheaders.c_str(), (DWORD)headers.size(),
        (LPVOID)body.data(), (DWORD)body.size(), (DWORD)body.size(), 0);
    BOOL recv = sent && WinHttpReceiveResponse(req, nullptr);
    bool ok = sent && recv;

    if (ok && result) {
        DWORD status = 0; DWORD status_len = sizeof(status);
        WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_len, WINHTTP_NO_HEADER_INDEX);
        result->status = (long)status;
        if (status >= 400) ok = false;
    }

    DWORD available = 0;
    while (ok && WinHttpQueryDataAvailable(req, &available) && available > 0) {
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!WinHttpReadData(req, &chunk[0], available, &read)) break;
        if (read > 0) {
            chunk.resize(read);
            if (result) result->body += chunk;
            if (sink && !sink(chunk)) break; // abort
        }
    }

    if (result && !ok) {
        if (!sent && GetLastError() == ERROR_WINHTTP_TIMEOUT) result->error = "timeout connecting to server";
        else if (sent && !recv && GetLastError() == ERROR_WINHTTP_TIMEOUT) result->error = "timeout waiting for response";
        else result->error = "HTTP request failed";
    }

    WinHttpCloseHandle(req);
    WinHttpCloseHandle(conn);
    WinHttpCloseHandle(session);
    return ok;
}
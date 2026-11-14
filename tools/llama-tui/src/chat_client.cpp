// tools/kelvin-tui/src/chat_client.cpp
#include "chat_client.h"
#include "http_client.h"

#include <sstream>
#include <cstdlib>
#include <cstdio>

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else out += c;
        }
    }
    return out;
}

void ChatClient::stream(const std::vector<ChatMessage>& history, const std::string& user_prompt,
                        const ChatCallbacks& cb) {
    std::ostringstream body;
    body << "{\"model\":\"local\",\"stream\":true,\"messages\":[";
    for (size_t i = 0; i < history.size(); i++) {
        if (i) body << ',';
        body << "{\"role\":\"" << (history[i].role == "assistant" ? "assistant" : "user")
             << "\",\"content\":\"" << json_escape(history[i].content) << "\"}";
    }
    body << "{\"role\":\"user\",\"content\":\"" << json_escape(user_prompt) << "\"}";
    body << "]}";

    std::string headers = "Content-Type: application/json\r\nAccept: text/event-stream\r\n";
    HttpResult result;

    bool ok = HttpClient().post("http://127.0.0.1:" + std::to_string(port_) + "/v1/chat/completions",
                                body.str(), headers, [&](const std::string& chunk) -> bool {
                                    // SSE payload; extract "data: ..." delta content.
                                    std::string s = chunk;
                                    size_t pos = 0;
                                    while ((pos = s.find("data: ", pos)) != std::string::npos) {
                                        size_t nl = s.find('\n', pos);
                                        std::string data = s.substr(pos + 6,
                                            nl == std::string::npos ? std::string::npos : nl - pos - 6);
                                        pos = nl == std::string::npos ? s.size() : nl + 1;
                                        if (data == "[DONE]") break;
                                        // Find "content":"..." inside the choice delta.
                                        size_t kc = data.find("\"content\":\"");
                                        if (kc != std::string::npos) {
                                            kc += 12;
                                            size_t end = data.find('"', kc);
                                            if (end != std::string::npos) {
                                                std::string tok = data.substr(kc, end - kc);
                                                // unescape common JSON escapes
                                                std::string out;
                                                for (size_t i = 0; i < tok.size(); i++) {
                                                    if (tok[i] == '\\' && i + 1 < tok.size()) {
                                                        char n = tok[i+1];
                                                        if (n == 'n') out += '\n';
                                                        else if (n == 't') out += '\t';
                                                        else if (n == 'r') out += '\r';
                                                        else if (n == '"') out += '"';
                                                        else if (n == '\\') out += '\\';
                                                        else out += n;
                                                        i++;
                                                    } else out += tok[i];
                                                }
                                                if (cb.on_delta) cb.on_delta(out);
                                            }
                                        }
                                    }
                                    return true;
                                }, &result);

    if (cb.on_done) cb.on_done(ok, ok ? "" : result.error);
}
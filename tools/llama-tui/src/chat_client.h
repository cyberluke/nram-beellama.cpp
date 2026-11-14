// tools/kelvin-tui/src/chat_client.h
#pragma once

#include <string>
#include <vector>
#include <functional>

struct ChatMessage {
    std::string role;    // "user" | "assistant"
    std::string content;
};

struct ChatCallbacks {
    // Invoked once per streamed content token/string chunk.
    std::function<void(const std::string&)> on_delta;
    // Invoked when the stream completes.
    std::function<void(bool success, const std::string& error)> on_done;
};

class ChatClient {
public:
    explicit ChatClient(int port = 8888) : port_(port) {}

    // Stream a chat completion. `history` is the full prior conversation.
    void stream(const std::vector<ChatMessage>& history, const std::string& user_prompt,
                const ChatCallbacks& cb);

private:
    int port_;
};
use futures_util::StreamExt;

/// A single chat message.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: String, // "user" | "assistant"
    pub content: String,
}

/// Stream a chat completion from llama-server's OpenAI-compatible endpoint.
/// `on_delta` is called as tokens arrive; returns Ok(()) on completion.
pub async fn stream_chat(
    port: u16,
    history: &[Message],
    prompt: &str,
    on_delta: impl Fn(&str) + Send + 'static,
) -> anyhow::Result<()> {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for m in history {
        messages.push(serde_json::json!({ "role": m.role.clone(), "content": m.content.clone() }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": prompt }));

    let body = serde_json::json!({
        "model": "local",
        "stream": true,
        "messages": messages,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        // SSE events are "data: {...}\n\n"; a chunk may contain several.
        while let Some(newline) = buffer.find('\n') {
            let line: String = buffer.drain(..=newline).collect();
            let line = line.trim();
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    return Ok(());
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = v.pointer("/choices/0/delta/content") {
                        if let Some(t) = delta.as_str() {
                            on_delta(t);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
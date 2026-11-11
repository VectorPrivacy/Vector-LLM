//! Minimal client for any OpenAI-compatible `/chat/completions` endpoint.
//!
//! Requests go through the Vector SDK's hardened HTTP client (the same one Vector uses
//! everywhere), so the bot's LLM traffic inherits its Tor failsafe and SSRF guards.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use vector_sdk::vector_core::net::build_http_client;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Pictures that go with the text, as `data:` URLs.
    pub images: Vec<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), images: Vec::new() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), images: Vec::new() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), images: Vec::new() }
    }

    /// The message as the API takes it: plain text, or text and images as parts.
    pub(crate) fn wire(&self) -> serde_json::Value {
        if self.images.is_empty() {
            return serde_json::json!({ "role": self.role.as_str(), "content": self.content });
        }
        let mut parts = vec![serde_json::json!({ "type": "text", "text": self.content })];
        parts.extend(self.images.iter().map(|url| serde_json::json!({ "type": "image_url", "image_url": { "url": url } })));
        serde_json::json!({ "role": self.role.as_str(), "content": parts })
    }
}

#[derive(Clone)]
pub struct LLMClient {
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: Option<u32>,
}

impl LLMClient {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
        model: String,
        temperature: f32,
        max_tokens: Option<u32>,
    ) -> Self {
        Self { base_url, api_key, model, temperature, max_tokens }
    }

    /// Send a chat completion and return the assistant's reply text.
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let payload = ChatRequest {
            model: &self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            messages: messages.iter().map(ChatMessage::wire).collect(),
        };
        // An empty answer is usually the provider's hiccup: the same request once more.
        match self.complete(&payload).await? {
            Ok(content) => Ok(content),
            Err(why) => {
                log::warn!("LLM gave no reply ({why}), asking again");
                self.complete(&payload).await?.map_err(|why| anyhow!("LLM gave no reply ({why})"))
            }
        }
    }

    /// One completion: the reply, or why there was none.
    async fn complete(&self, payload: &ChatRequest<'_>) -> Result<std::result::Result<String, String>> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let http = build_http_client(Duration::from_secs(120)).map_err(|e| anyhow!(e))?;
        let request = || {
            let req = http.post(url.as_str()).json(payload);
            match &self.api_key {
                Some(key) => req.bearer_auth(key),
                None => req,
            }
        };

        // A connection the server closed between requests fails at once: one more try, on a new one.
        let resp = match request().send().await {
            Ok(resp) => resp,
            // A refusal by the network's rules (Tor or I2P only) stays one.
            Err(e) if e.is_policy_refusal() => return Err(e.into()),
            Err(e) => {
                if e.is_transient() {
                    vector_sdk::vector_core::net::wait_after(&e).await.map_err(|why| anyhow!(why))?;
                } else {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                request().send().await?
            }
        };
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(anyhow!(
                "LLM HTTP {}: {}",
                status,
                String::from_utf8_lossy(&bytes)
            ));
        }

        let body: ChatResponse = serde_json::from_slice(&bytes)?;
        if let Some(u) = &body.usage {
            let cached = u.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens).unwrap_or(0);
            log::info!("🧾 {} tokens in ({cached} cached), {} out", u.prompt_tokens, u.completion_tokens);
        }
        Ok(body.reply())
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    messages: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetails>,
}

#[derive(Deserialize)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

impl ChatResponse {
    /// The first choice's text (chat format, else the legacy `text` some servers still emit), or
    /// why there is none.
    fn reply(self) -> std::result::Result<String, String> {
        let Some(choice) = self.choices.into_iter().next() else {
            return Err("no choices".into());
        };
        choice
            .message
            .and_then(|m| m.content)
            .or(choice.text)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("empty content, finish_reason {}", choice.finish_reason.as_deref().unwrap_or("none")))
    }
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<OwnedMessage>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OwnedMessage {
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(json: &str) -> std::result::Result<String, String> {
        serde_json::from_str::<ChatResponse>(json).unwrap().reply()
    }

    #[test]
    fn a_null_reply_is_no_reply_not_a_parse_error() {
        assert_eq!(reply(r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#), Ok("hi".into()));
        assert_eq!(
            reply(r#"{"choices":[{"finish_reason":"content_filter","message":{"role":"assistant","content":null}}]}"#),
            Err("empty content, finish_reason content_filter".into())
        );
        assert_eq!(reply(r#"{"choices":[{"text":"legacy"}]}"#), Ok("legacy".into()));
        assert_eq!(reply(r#"{"choices":[]}"#), Err("no choices".into()));
    }
}

use anyhow::{anyhow, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Clone)]
pub struct LLMClient {
    http: HttpClient,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
}

impl LLMClient {
    // Return a cloned client with a different model name
    pub fn with_model(&self, model: impl Into<String>) -> Self {
        let mut c = self.clone();
        c.model = model.into();
        c
    }
}

impl LLMClient {
    pub fn new(base_url: String, api_key: Option<String>, model: String, temperature: f32) -> Self {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build reqwest client");
        Self {
            http,
            base_url,
            api_key,
            model,
            temperature,
        }
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        // OpenAI-compatible /chat/completions
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.api_key {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", key)).unwrap(),
            );
        }

        let payload = ChatRequest {
            model: self.model.clone(),
            temperature: Some(self.temperature),
            messages: messages
                .into_iter()
                .map(|m| OpenAIMessage {
                    role: match m.role {
                        Role::System => "system".into(),
                        Role::User => "user".into(),
                        Role::Assistant => "assistant".into(),
                    },
                    content: m.content,
                })
                .collect(),
        };

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&payload)
            .send()
            .await?;

        // Capture status, then fully read the body once.
        let status = resp.status();
        let bytes = resp.bytes().await?;

        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes).to_string();
            return Err(anyhow!("LLM HTTP error: {} - {}", status, text));
        }

        // Parse JSON from the buffered bytes to avoid moving the response twice.
        let body: ChatResponse = serde_json::from_slice(&bytes)?;
        let choice = body.choices.get(0).ok_or_else(|| anyhow!("no choices in response"))?;

        // Primary: chat format
        if let Some(msg) = &choice.message {
            if !msg.content.is_empty() {
                return Ok(msg.content.clone());
            }
        }
        // Fallback: some servers still return "text" instead of "message"
        if let Some(text) = &choice.text {
            if !text.is_empty() {
                return Ok(text.clone());
            }
        }

        Err(anyhow!("empty assistant content"))
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    messages: Vec<OpenAIMessage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<OpenAIMessage>,
    // Some OpenAI-compatible servers may provide "text" (legacy)
    #[serde(default)]
    text: Option<String>,
}
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::Path};
use log::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub history_limit: usize,
    pub data_dir: String,
    pub typing_indicator: bool,
    pub vector_secret_key: Option<String>,
    pub system_prompt: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        // Base URL for OpenAI-compatible API
        let base_url = env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        // Optional API key (some local servers like llama.cpp may not require it)
        let api_key = env::var("LLM_API_KEY").ok().filter(|s| !s.is_empty());

        // Model name for the OpenAI-compatible server
        let model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        // Temperature (default 0.2)
        let temperature = env::var("LLM_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.2);

        // History limit (only count user/assistant turns, system is always kept)
        let history_limit = env::var("HISTORY_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16);

        // Data directory for storing per-chat memory JSON files
        let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string());

        // Whether to send typing indicators via Vector SDK (kind 30078)
        let typing_indicator = env::var("TYPING_INDICATOR")
            .ok()
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(true);

        // Optional Vector secret key (nsec bech32 or hex). If unset, we generate ephemeral keys.
        let vector_secret_key = env::var("VECTOR_SECRET_KEY").ok().filter(|s| !s.is_empty());

        // Default system prompt with robust .env fallback (in case dotenv didn't populate),
        // and append-only persistence to .env when missing to make it obvious/editable.
        let (system_prompt, sp_source) = match env::var("SYSTEM_PROMPT")
            .ok()
            .filter(|s| !s.trim().is_empty())
        {
            Some(v) => (normalize_env_value(&v), "env"),
            None => match read_env_file_var(".env", "SYSTEM_PROMPT") {
                Some(v) if !v.trim().is_empty() => (normalize_env_value(&v), "file"),
                _ => (
                    "You are a helpful assistant. Keep responses concise and factual.".to_string(),
                    "default",
                ),
            },
        };
        info!("SYSTEM_PROMPT source: {}", sp_source);
        if sp_source == "default" {
            // Append to .env only if missing; do not overwrite existing values.
            let _ = persist_env_var(".env", "SYSTEM_PROMPT", &system_prompt);
        }

        // Basic validation
        if base_url.trim().is_empty() {
            bail!("LLM_BASE_URL cannot be empty");
        }
        if model.trim().is_empty() {
            bail!("LLM_MODEL cannot be empty");
        }

        Ok(Self {
            base_url,
            api_key,
            model,
            temperature,
            history_limit,
            data_dir,
            typing_indicator,
            vector_secret_key,
            system_prompt,
        })
    }
}

// --- Local helpers for robust .env fallback parsing (single-line values) ---
fn read_env_file_var(path: &str, var: &str) -> Option<String> {
    let p = Path::new(path);
    if !p.exists() {
        return None;
    }
    let contents = fs::read_to_string(p).ok()?;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Support optional "export "
        let core = if let Some(rest) = trimmed.strip_prefix("export ") {
            rest.trim()
        } else {
            trimmed
        };
        if let Some((k, v)) = core.split_once('=') {
            if k.trim() == var {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

// Normalize typical .env values:
// - trims whitespace
// - strips surrounding quotes
// - removes inline comments
// - unescapes \n to real newlines (common for prompts)
fn persist_env_var(path: &str, var: &str, value: &str) -> anyhow::Result<()> {
    let p = Path::new(path);
    if p.exists() {
        let mut contents = fs::read_to_string(p)?;
        let exists = contents
            .lines()
            .any(|line| line.trim_start().starts_with(&format!("{var}=")));
        if exists {
            // Respect existing .env; never overwrite here.
            return Ok(());
        }
        if !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str(&format!("{var}={value}\n"));
        fs::write(p, contents)?;
    } else {
        let body = format!("# Auto-generated by Vector LLM Bot\n{var}={value}\n");
        fs::write(p, body)?;
    }
    Ok(())
}

fn normalize_env_value(s: &str) -> String {
    let mut t = s.trim().to_string();

    // Strip surrounding quotes if present
    if (t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')) {
        t = t[1..t.len().saturating_sub(1)].to_string();
    }

    // Remove inline comments if not quoted
    if let Some(hash) = t.find('#') {
        t = t[..hash].trim().to_string();
    }

    // Unescape common sequences
    t = t.replace(r"\n", "\n").replace(r"\t", "\t");

    t.trim().to_string()
}
mod config;
mod memory;
mod llm;

use dotenvy::dotenv;
use env_logger::Env;
use log::{debug, error, info, warn};
use std::{fs, path::Path, time::Duration};

use vector_sdk::nostr::{
    FromBech32, Keys, Kind, SecretKey, ToBech32, UnwrappedGift, RelayPoolNotification,
};
use vector_sdk::VectorBot;

use crate::config::Config;
use crate::llm::{LLMClient, ChatMessage, Role};
use crate::memory::ChatMemory;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cfg = Config::load()?;
    info!("Config loaded. base_url={}, model={}, history_limit={}", cfg.base_url, cfg.model, cfg.history_limit);
    info!("System prompt length: {}", cfg.system_prompt.len());

    // Build Nostr keys from env (supports nsec bech32 or hex)
    let keys = load_keys_from_env(&cfg)?;
    info!("Loaded Vector keys: {}", keys.public_key().to_bech32()?);

    // Build a simple Vector bot (uses SDK defaults for relays/subscriptions)
    let bot = VectorBot::new(
        keys.clone(),
        "Viktor",
        "Viktor",
        "I'm Viktor, just a dude.",
        "https://jskitty.cat/vector/img/app-logo.jpg",
        "https://jskitty.cat/vector/img/app-logo.jpg",
        "",
        "",
    ).await;

    // LLM client
    let llm = LLMClient::new(cfg.base_url.clone(), cfg.api_key.clone(), cfg.model.clone(), cfg.temperature);

    // Notification loop
    let mut notifications = bot.client.notifications();

    info!("Bot started. Waiting for messages...");
    loop {
        let notif = notifications.recv().await;
        match notif {
            Ok(RelayPoolNotification::Event { event, .. }) => {
                if event.kind == Kind::GiftWrap {
                    // Try to unwrap using Vector SDK's client.unwrap_gift_wrap()
                    match bot.client.unwrap_gift_wrap(&event).await {
                        Ok(UnwrappedGift { sender, rumor }) => {
                            // Filter for private direct messages only
                            if rumor.kind != Kind::PrivateDirectMessage {
                                continue;
                            }

                            let content = rumor.content;
                            debug!("Got DM from {}: {}", sender.to_bech32().unwrap_or_default(), content);

                            // Build per-chat memory
                            let chat_id = sender.to_bech32().unwrap_or_else(|_| "unknown".to_string());
                            let mut memory = ChatMemory::load_or_new(&cfg.data_dir, &chat_id, cfg.history_limit, cfg.system_prompt.clone())?;
                            debug!(
                                "Loaded/created memory for chat {}: system_prompt_len={}, history_limit={}",
                                chat_id,
                                memory.system_prompt.len(),
                                memory.history_limit
                            );

                            // Command: !clear -> wipe rolling message history for this chat (keep system prompt)
                            if content.trim().eq_ignore_ascii_case("!clear") {
                                memory.messages.clear();
                                memory.save(&cfg.data_dir, &chat_id)?;
                                let chat = bot.get_chat(sender).await;
                                let confirm = "Cleared your chat history (system prompt retained).";
                                if !chat.send_private_message(confirm).await {
                                    warn!("Failed to send clear confirmation to {}", chat_id);
                                }
                                continue;
                            }

                            // Command: !model [name|reset] -> set per-user model override (or clear)
                            if let Some(rest) = content.strip_prefix("!model") {
                                let arg = rest.trim();
                                let chat = bot.get_chat(sender).await;
                                if arg.is_empty() || arg == "?" || arg.eq_ignore_ascii_case("show") {
                                    let current = memory.model_override.clone().unwrap_or_else(|| cfg.model.clone());
                                    let msg = format!("Current model for this chat: {}", current);
                                    let _ = chat.send_private_message(&msg).await;
                                } else if arg.eq_ignore_ascii_case("reset") || arg.eq_ignore_ascii_case("clear") {
                                    memory.model_override = None;
                                    memory.save(&cfg.data_dir, &chat_id)?;
                                    let _ = chat.send_private_message("Per-chat model reset to default.").await;
                                } else {
                                    // take first whitespace-delimited token as model name
                                    let model_name = arg.split_whitespace().next().unwrap_or(arg).to_string();
                                    memory.model_override = Some(model_name.clone());
                                    memory.save(&cfg.data_dir, &chat_id)?;
                                    let _ = chat.send_private_message(&format!("Per-chat model set to '{}'.", model_name)).await;
                                }
                                continue;
                            }

                            // Add the user's message
                            memory.push(ChatMessage { role: Role::User, content: content.clone() });
                            memory.trim_to_limit();
                            memory.save(&cfg.data_dir, &chat_id)?;

                            // Build prompt for LLM (system + last N messages)
                            let messages = memory.messages_for_llm();
                            debug!(
                                "LLM call context: system_prompt_len={}, history_len={}",
                                messages.first().map(|m| m.content.len()).unwrap_or(0),
                                messages.len().saturating_sub(1)
                            );

                            // Determine model for this chat (per-user override)
                            let llm_for_chat = if let Some(m) = &memory.model_override {
                                llm.with_model(m.clone())
                            } else {
                                llm.clone()
                            };

                            // Optional: typing indicator
                            if cfg.typing_indicator {
                                // best-effort
                                let _ = bot.get_chat(sender).await.send_typing_indicator().await;
                            }

                            // Call LLM
                            match llm_for_chat.chat(messages).await {
                                Ok(reply_text) => {
                                    // Persist assistant message
                                    memory.push(ChatMessage { role: Role::Assistant, content: reply_text.clone() });
                                    memory.trim_to_limit();
                                    memory.save(&cfg.data_dir, &chat_id)?;

                                    // Send reply back via Vector SDK
                                    let chat = bot.get_chat(sender).await;
                                    if !chat.send_private_message(&reply_text).await {
                                        warn!("Failed to send reply to {}", chat_id);
                                    }
                                }
                                Err(e) => error!("LLM request failed: {}", e),
                            }
                        }
                        Err(e) => {
                            warn!("Failed to unwrap gift: {}", e);
                        }
                    }
                }
            }
            Ok(other) => {
                // Ignore other notifications for minimal bot
                debug!("Notification: {:?}", other);
            }
            Err(e) => {
                warn!("Notification channel error: {}. Reconnecting delay...", e);
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
}


/**
 Reads VECTOR_SECRET_KEY from env (bech32 nsec or hex).
 If missing, generates fresh keys and persists them to .env for subsequent runs.
*/
fn load_keys_from_env(cfg: &Config) -> anyhow::Result<Keys> {
    // Prefer reading the live env var directly (dotenv already loaded),
    // then Config-captured value, then fall back to parsing .env manually (robustness).
    let raw_env = std::env::var("VECTOR_SECRET_KEY").ok();
    let raw_cfg = cfg.vector_secret_key.clone();
    let raw_file = read_env_file_var(".env", "VECTOR_SECRET_KEY").ok();


    let raw = raw_env.or(raw_cfg).or(raw_file);

    if let Some(secret_raw) = raw {
        let secret = normalize_secret(&secret_raw);
        let len = secret.len();
        debug!("VECTOR_SECRET_KEY after normalization: len={}", len);

        // Try nsec bech32 first
        if secret.starts_with("nsec") {
            debug!("VECTOR_SECRET_KEY looks like bech32 (nsec...), attempting decode");
            match SecretKey::from_bech32(&secret) {
                Ok(sk) => {
                    info!("Loaded VECTOR_SECRET_KEY from bech32 (nsec..., len={})", len);
                    return Ok(Keys::new(sk));
                }
                Err(e) => {
                    warn!("VECTOR_SECRET_KEY present but invalid bech32 ({}). Refusing to overwrite; fix .env", e);
                    return Err(anyhow::anyhow!("Invalid VECTOR_SECRET_KEY bech32 in .env"));
                }
            }
        }

        // Fallback to hex (32-byte typical)
        if let Ok(bytes) = hex::decode(&secret) {
            debug!("VECTOR_SECRET_KEY parsed as hex ({} bytes), attempting SecretKey::from_slice", bytes.len());
            match SecretKey::from_slice(&bytes) {
                Ok(sk) => {
                    info!("Loaded VECTOR_SECRET_KEY from hex ({} bytes)", bytes.len());
                    return Ok(Keys::new(sk));
                }
                Err(e) => {
                    warn!("VECTOR_SECRET_KEY provided hex but failed to parse ({}). Refusing to overwrite; fix .env", e);
                    return Err(anyhow::anyhow!("Invalid VECTOR_SECRET_KEY hex in .env"));
                }
            }
        }

        if !secret.is_empty() {
            // Value present but not parseable; refuse to overwrite.
            warn!("VECTOR_SECRET_KEY present but not a valid nsec or hex. Refusing to overwrite; fix .env");
            return Err(anyhow::anyhow!("Invalid VECTOR_SECRET_KEY format in .env"));
        }
        // Empty after normalization -> treat as missing
        debug!("VECTOR_SECRET_KEY empty after normalization; treating as missing");
    }

    // Only generate and persist when the key is truly missing.
    let keys = Keys::generate();
    let nsec = keys.secret_key().to_bech32()?;
    persist_secret_to_env("VECTOR_SECRET_KEY", &nsec)?;
    info!("VECTOR_SECRET_KEY missing; generated new keys and saved to .env");
    Ok(keys)
}

fn read_env_file_var(path: &str, var: &str) -> anyhow::Result<String> {
    let p = Path::new(path);
    if !p.exists() {
        anyhow::bail!("no .env file");
    }
    let contents = fs::read_to_string(p)?;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Support optional "export " prefix
        let core = if let Some(rest) = trimmed.strip_prefix("export ") {
            rest.trim()
        } else {
            trimmed
        };
        if let Some((k, v)) = core.split_once('=') {
            if k.trim() == var {
                return Ok(v.trim().to_string());
            }
        }
    }
    anyhow::bail!("var not found")
}

fn normalize_secret(s: &str) -> String {
    // Start with trim of leading/trailing whitespace
    let mut t = s.trim().to_string();

    // Handle "export VAR=..." style (some shells or templates use this)
    if t.starts_with("export ") {
        t = t["export ".len()..].trim().to_string();
        if let Some(eq) = t.find('=') {
            t = t[eq + 1..].trim().to_string();
        }
    }

    // Strip surrounding single/double quotes if present
    if (t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')) {
        t = t[1..t.len().saturating_sub(1)].to_string();
    }

    // Remove inline comments (dotenv doesn't support inline comments unless quoted)
    if let Some(hash) = t.find('#') {
        t = t[..hash].trim().to_string();
    }

    // Keep only first non-whitespace token (secrets shouldn't contain spaces)
    if let Some(space) = t.find(char::is_whitespace) {
        t = t[..space].to_string();
    }

    t.trim().to_string()
}

// Append-only: add VAR=VALUE to .env if missing; never overwrite an existing line.
fn persist_secret_to_env(var: &str, value: &str) -> anyhow::Result<()> {
    let path = Path::new(".env");
    if path.exists() {
        let mut contents = fs::read_to_string(path)?;
        let exists = contents
            .lines()
            .any(|line| line.trim_start().starts_with(&format!("{var}=")));

        if exists {
            // Do not overwrite existing key; user likely set it intentionally.
            return Ok(());
        }

        // Append new line with the variable
        if !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str(&format!("{var}={value}\n"));
        fs::write(path, contents)?;
    } else {
        let body = format!("# Auto-generated by Vector LLM Bot\n{var}={value}\n");
        fs::write(path, body)?;
    }
    Ok(())
}
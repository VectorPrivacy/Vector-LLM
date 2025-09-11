use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::llm::{ChatMessage, Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMemory {
    pub system_prompt: String,
    pub history_limit: usize,
    #[serde(default)]
    pub model_override: Option<String>,
    pub messages: Vec<ChatMessage>, // only user/assistant turns; system is stored separately
}

impl ChatMemory {
    pub fn load_or_new<P: AsRef<Path>>(
        data_dir: P,
        chat_id: &str,
        history_limit: usize,
        system_prompt: String,
    ) -> Result<Self> {
        let path = memory_path(&data_dir, chat_id);
        fs::create_dir_all(path.parent().unwrap())
            .with_context(|| format!("creating data dir {}", path.display()))?;

        if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("reading memory file {}", path.display()))?;
            let mut mem: ChatMemory = serde_json::from_str(&raw)
                .with_context(|| format!("parsing memory file {}", path.display()))?;
            // Update policy from current runtime options
            mem.history_limit = history_limit;
            mem.system_prompt = system_prompt;
            mem.trim_to_limit();
            Ok(mem)
        } else {
            Ok(ChatMemory {
                system_prompt,
                history_limit,
                model_override: None,
                messages: Vec::new(),
            })
        }
    }

    pub fn save<P: AsRef<Path>>(&self, data_dir: P, chat_id: &str) -> Result<()> {
        let path = memory_path(&data_dir, chat_id);
        fs::create_dir_all(path.parent().unwrap())
            .with_context(|| format!("creating data dir {}", path.display()))?;
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json).with_context(|| format!("writing memory file {}", path.display()))?;
        Ok(())
    }

    pub fn push(&mut self, msg: ChatMessage) {
        // Only persist user/assistant turns; ignore stray system messages here
        if matches!(msg.role, Role::User | Role::Assistant) {
            self.messages.push(msg);
        }
    }

    // Keep only last N messages (N = history_limit), preserving recency
    pub fn trim_to_limit(&mut self) {
        if self.messages.len() > self.history_limit {
            let start = self.messages.len() - self.history_limit;
            self.messages = self.messages[start..].to_vec();
        }
    }

    // Format for OpenAI-compatible request: [system] + last N messages
    pub fn messages_for_llm(&self) -> Vec<ChatMessage> {
        let mut out = Vec::with_capacity(1 + self.messages.len());
        out.push(ChatMessage {
            role: Role::System,
            content: self.system_prompt.clone(),
        });
        out.extend(self.messages.iter().cloned());
        out
    }
}

fn memory_path<P: AsRef<Path>>(data_dir: P, chat_id: &str) -> PathBuf {
    // Vector npub IDs are already filesystem-safe; no sanitization required
    data_dir.as_ref().join(format!("{chat_id}.json"))
}

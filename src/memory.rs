//! Per-channel conversation memory, persisted as one JSON file per chat or channel.
//!
//! A channel is a DM (keyed by the peer's npub) or a Community channel (keyed by its channel
//! id) — the SDK gives both a uniform `chat_id`, so storage is uniform too. In group channels
//! each user turn records its author npub, which [`crate::prompt`] turns into a speaker label
//! plus an npub roster so the model can follow a multi-person conversation cheaply.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::llm::Role;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: Role,
    pub content: String,
    /// Author npub for a user turn in a group channel — drives speaker labels + the roster.
    /// Absent for assistant turns and for DMs (a single, implicit counterparty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Pictures that came with it, saved under `<data_dir>/images/`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// The Vector message it came from, while its attachments haven't been read yet: a message
    /// in a group that didn't address the bot is read only once someone asks about it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unread_from: Option<String>,
    /// The Vector message it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// When it was sent (unix ms); 0 for turns stored before this was recorded.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub at: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl StoredMessage {
    pub fn user(content: String, author: Option<String>) -> Self {
        Self { role: Role::User, content, author, images: Vec::new(), unread_from: None, id: None, at: 0 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelMemory {
    #[serde(default)]
    pub is_group: bool,
    #[serde(default)]
    pub messages: Vec<StoredMessage>,
}

impl ChannelMemory {
    /// Load this channel's memory, or start fresh. `is_group` reflects the live channel kind
    /// (it can change if a chat id is reused), so we always refresh it.
    pub fn load_or_new(data_dir: &str, chat_id: &str, is_group: bool) -> Result<Self> {
        let path = memory_path(data_dir, chat_id);
        if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let mut mem: ChannelMemory = serde_json::from_str(&raw)
                .with_context(|| format!("parsing {}", path.display()))?;
            mem.is_group = is_group;
            Ok(mem)
        } else {
            Ok(ChannelMemory { is_group, ..Default::default() })
        }
    }

    pub fn save(&self, data_dir: &str, chat_id: &str) -> Result<()> {
        let path = memory_path(data_dir, chat_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn push_user(&mut self, content: String, author: Option<String>) {
        self.messages.push(StoredMessage::user(content, author));
    }

    /// Add a turn, unless one from the same Vector message is already here: two paths (a reply
    /// pulling in what it answers, and that message's own arrival) can each bring it.
    pub fn push(&mut self, message: StoredMessage) {
        if message.id.is_some() && self.holds(message.id.as_deref()) {
            return;
        }
        self.messages.push(message);
    }

    /// Whether a turn from Vector message `id` is held.
    pub fn holds(&self, id: Option<&str>) -> bool {
        id.is_some() && self.messages.iter().any(|m| m.id.as_deref() == id)
    }

    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(StoredMessage { role: Role::Assistant, ..StoredMessage::user(content, None) });
    }

    /// Keep at most `limit` turns. Past it, the oldest quarter goes at once: the conversation's
    /// start then holds still for the turns that follow, which is what a provider's prompt cache keys on.
    pub fn trim(&mut self, limit: usize) {
        if self.messages.len() > limit {
            let cut = self.messages.len() - (limit - limit / 4);
            self.messages.drain(0..cut);
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Distinct author npubs across retained user turns, in first-seen order.
    pub fn distinct_authors(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for m in &self.messages {
            if let Some(a) = &m.author {
                if seen.insert(a.clone()) {
                    out.push(a.clone());
                }
            }
        }
        out
    }
}

fn memory_path(data_dir: &str, chat_id: &str) -> PathBuf {
    // npubs and 64-hex channel ids are already filesystem-safe; no sanitization needed.
    Path::new(data_dir).join("chats").join(format!("{chat_id}.json"))
}

/// A process-wide, race-safe cache over the per-channel memory files.
///
/// The SDK dispatches each inbound message in its own task, so two messages in the same channel
/// can be handled concurrently — a naive load/append/save would lose writes. This keeps each
/// channel's memory in RAM behind a mutex and persists on every mutation, so the file is only ever
/// written under the lock.
#[derive(Clone)]
pub struct MemoryStore {
    data_dir: String,
    cache: Arc<Mutex<HashMap<String, ChannelMemory>>>,
}

impl MemoryStore {
    pub fn new(data_dir: impl Into<String>) -> Self {
        Self { data_dir: data_dir.into(), cache: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Apply `f` to the channel's memory under the lock, persist, and return the updated snapshot.
    pub fn update<F>(&self, chat_id: &str, is_group: bool, f: F) -> ChannelMemory
    where
        F: FnOnce(&mut ChannelMemory),
    {
        let mut cache = self.cache.lock().unwrap();
        let mem = cache.entry(chat_id.to_string()).or_insert_with(|| load_or_default(&self.data_dir, chat_id, is_group));
        mem.is_group = is_group; // reflect the live channel kind
        f(mem);
        if let Err(e) = mem.save(&self.data_dir, chat_id) {
            log::warn!("memory save failed for {chat_id}: {e}");
        }
        mem.clone()
    }

    /// A read-only snapshot of the channel's memory (loaded from disk on first access).
    pub fn read(&self, chat_id: &str, is_group: bool) -> ChannelMemory {
        let mut cache = self.cache.lock().unwrap();
        let mem = cache.entry(chat_id.to_string()).or_insert_with(|| load_or_default(&self.data_dir, chat_id, is_group));
        mem.is_group = is_group;
        mem.clone()
    }
}

fn load_or_default(data_dir: &str, chat_id: &str, is_group: bool) -> ChannelMemory {
    ChannelMemory::load_or_new(data_dir, chat_id, is_group).unwrap_or_else(|e| {
        log::warn!("memory load failed for {chat_id} ({e}); starting fresh");
        ChannelMemory { is_group, ..Default::default() }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_two_paths_bring_is_kept_once() {
        let mut mem = ChannelMemory::default();
        let turn = |text: &str| StoredMessage { id: Some("abc".into()), ..StoredMessage::user(text.into(), None) };
        mem.push(turn("pulled in by the reply"));
        mem.push(turn("its own arrival"));
        mem.push_user("no id, always kept".into(), None);
        assert_eq!(mem.messages.len(), 2);
        assert!(mem.holds(Some("abc")) && !mem.holds(None));
    }

    #[test]
    fn a_full_memory_sheds_its_oldest_quarter_at_once() {
        let mut mem = ChannelMemory::default();
        for i in 0..51 {
            mem.push_user(i.to_string(), None);
            mem.trim(50);
        }
        assert_eq!(mem.messages.len(), 38);
        assert_eq!(mem.messages[0].content, "13");
        for i in 51..63 {
            mem.push_user(i.to_string(), None);
            mem.trim(50);
            assert_eq!(mem.messages[0].content, "13", "the start holds until the limit is passed again");
        }
    }
}

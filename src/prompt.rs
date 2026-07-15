//! Turns a channel's stored memory into the OpenAI-style message list sent to the LLM.
//!
//! - **DMs** are a plain two-party transcript: `[system] + user/assistant turns`.
//! - **Group channels** are multi-user, so each user turn is prefixed with the speaker's label
//!   and the system prompt gains a roster mapping every label to its npub. The label is the
//!   user's name when they have one (repeated cheaply per message) and a short npub otherwise —
//!   so the model can follow who said what without spending tokens on a full npub per line.

use std::collections::HashMap;

use crate::llm::{ChatMessage, Role};
use crate::memory::ChannelMemory;
use crate::names::{short_npub, NameResolver};

pub async fn build_messages(
    mem: &ChannelMemory,
    system_prompt: &str,
    names: &NameResolver,
) -> Vec<ChatMessage> {
    if !mem.is_group {
        let mut out = Vec::with_capacity(mem.messages.len() + 1);
        out.push(ChatMessage::system(system_prompt));
        for m in &mem.messages {
            out.push(ChatMessage { role: m.role, content: m.content.clone() });
        }
        return out;
    }

    // One label per distinct author, disambiguating any display-name collisions with an npub tail.
    let mut label_for: HashMap<String, String> = HashMap::new();
    let mut name_counts: HashMap<String, u32> = HashMap::new();
    let mut roster: Vec<(String, String)> = Vec::new(); // (label, npub)
    for npub in mem.distinct_authors() {
        let label = match names.name_of(&npub).await {
            Some(name) => {
                let count = name_counts.entry(name.clone()).or_insert(0);
                *count += 1;
                if *count > 1 {
                    format!("{name} ({})", short_npub(&npub))
                } else {
                    name
                }
            }
            None => short_npub(&npub),
        };
        label_for.insert(npub.clone(), label.clone());
        roster.push((label, npub));
    }

    let mut system = system_prompt.to_string();
    if !roster.is_empty() {
        system.push_str(
            "\n\nThis is a group chat with multiple participants. Every user message below is \
             prefixed with the speaker's name and a colon. You are the assistant — never prefix \
             your own replies, and address people by name when it helps. Participants:\n",
        );
        for (label, npub) in &roster {
            system.push_str(&format!("- {label} = {npub}\n"));
        }
    }

    let mut out = Vec::with_capacity(mem.messages.len() + 1);
    out.push(ChatMessage::system(system));
    for m in &mem.messages {
        match m.role {
            Role::Assistant => out.push(ChatMessage::assistant(m.content.clone())),
            _ => {
                let label = m
                    .author
                    .as_ref()
                    .and_then(|a| label_for.get(a))
                    .cloned()
                    .unwrap_or_else(|| "user".to_string());
                out.push(ChatMessage::user(format!("{label}: {}", m.content)));
            }
        }
    }
    out
}

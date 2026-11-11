//! Turns a channel's stored memory into the OpenAI-style message list sent to the LLM.
//!
//! - **DMs** are a plain two-party transcript: `[system] + user/assistant turns`.
//! - **Group channels** are multi-user, so each user turn is prefixed with the speaker's label
//!   and the system prompt gains a roster mapping every label to its npub. The label is the
//!   user's name when they have one (repeated cheaply per message) and a short npub otherwise —
//!   so the model can follow who said what without spending tokens on a full npub per line.

use std::collections::HashMap;
use std::path::Path;

use crate::llm::{ChatMessage, Role};
use crate::memory::{ChannelMemory, StoredMessage};
use crate::names::{short_npub, NameResolver};

/// Pictures the model sees at once: the newest in the conversation. Older ones stay in memory
/// as a note, so a long chat doesn't resend every picture each turn.
const IMAGES_IN_CONTEXT: usize = 4;

/// A user turn with its pictures: the newest [`IMAGES_IN_CONTEXT`] go along, older ones become a note.
fn with_images(content: String, m: &StoredMessage, shown: &mut usize, images_dir: &Path) -> ChatMessage {
    let mut msg = ChatMessage::user(content);
    for file in m.images.iter().rev() {
        let url = (*shown < IMAGES_IN_CONTEXT).then(|| crate::files::data_url(images_dir, file)).flatten();
        match url {
            Some(url) => {
                *shown += 1;
                msg.images.insert(0, url);
            }
            None => msg.content.push_str("\n[an earlier picture, no longer shown]"),
        }
    }
    msg
}

pub async fn build_messages(
    mem: &ChannelMemory,
    system_prompt: &str,
    names: &NameResolver,
    images_dir: &Path,
) -> Vec<ChatMessage> {
    let mut shown = 0;
    if !mem.is_group {
        // Newest first, so the newest pictures are the ones shown; then back in order.
        let mut turns: Vec<ChatMessage> = Vec::with_capacity(mem.messages.len());
        for m in mem.messages.iter().rev() {
            turns.push(match m.role {
                Role::User => with_images(m.content.clone(), m, &mut shown, images_dir),
                role => ChatMessage { role, ..ChatMessage::user(m.content.clone()) },
            });
        }
        turns.reverse();
        let mut out = Vec::with_capacity(turns.len() + 1);
        out.push(ChatMessage::system(system_prompt));
        out.extend(turns);
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

    let mut turns: Vec<ChatMessage> = Vec::with_capacity(mem.messages.len());
    for m in mem.messages.iter().rev() {
        turns.push(match m.role {
            Role::Assistant => ChatMessage::assistant(m.content.clone()),
            _ => {
                let label = m
                    .author
                    .as_ref()
                    .and_then(|a| label_for.get(a))
                    .cloned()
                    .unwrap_or_else(|| "user".to_string());
                with_images(format!("{label}: {}", m.content), m, &mut shown, images_dir)
            }
        });
    }
    turns.reverse();
    let mut out = Vec::with_capacity(turns.len() + 1);
    out.push(ChatMessage::system(system));
    out.extend(turns);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_newest_pictures_go_along() {
        let dir = std::env::temp_dir().join(format!("viktor-prompt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let turns: Vec<StoredMessage> = (0..6)
            .map(|i| {
                let file = format!("m{i}-0.png");
                std::fs::write(dir.join(&file), [i as u8; 4]).unwrap();
                StoredMessage { images: vec![file], ..StoredMessage::user(format!("picture {i}"), None) }
            })
            .collect();
        let mut shown = 0;
        let built: Vec<ChatMessage> = turns.iter().rev().map(|m| with_images(m.content.clone(), m, &mut shown, &dir)).collect();
        let with: Vec<bool> = built.iter().rev().map(|m| !m.images.is_empty()).collect();
        assert_eq!(with, [false, false, true, true, true, true], "the newest four");
        assert!(built.last().unwrap().content.ends_with("[an earlier picture, no longer shown]"));
        assert!(built[0].images[0].starts_with("data:image/png;base64,"));
        std::fs::remove_dir_all(&dir).ok();
    }
}

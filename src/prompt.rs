//! Turns a channel's stored memory into the OpenAI-style message list sent to the LLM.
//!
//! - **DMs** are a plain two-party transcript: `[system] + user/assistant turns`.
//! - **Group channels** are multi-user, so each user turn is prefixed with the speaker's label: their
//!   name when they have one, a short npub otherwise. A speaker's first turn also carries their full
//!   npub, so the model can tell people apart without a roster.
//!
//! A turn, once built, reads the same in every later request until memory is trimmed: providers
//! cache the longest unchanged start of a prompt, and that is most of every request.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::llm::{ChatMessage, Role};
use crate::memory::{ChannelMemory, StoredMessage};
use crate::names::{short_npub, NameResolver};

/// Pictures the model sees at once: the newest in the conversation. Older ones stay in memory
/// as a note, so a long chat doesn't resend every picture each turn.
const IMAGES_IN_CONTEXT: usize = 4;

const GROUP_PROMPT: &str = "\n\nThis is a group chat with multiple participants. Every user message below is \
    prefixed with the speaker's name and a colon; a speaker's first message also gives their npub in \
    square brackets. You are the assistant: never prefix your own replies, and address people by name \
    when it helps.";

/// How many of the window's pictures, oldest first, are shown as a note instead. They are hidden
/// [`IMAGES_IN_CONTEXT`] at a time, so between that many and twice that less one are shown, and a
/// new picture rewrites an earlier turn only once in that many.
fn hidden_pictures(mem: &ChannelMemory) -> usize {
    let total: usize = mem.messages.iter().map(|m| m.images.len()).sum();
    total.saturating_sub(IMAGES_IN_CONTEXT) / IMAGES_IN_CONTEXT * IMAGES_IN_CONTEXT
}

/// A user turn with its pictures; the first `hidden` of the window's pictures become a note.
fn with_images(content: String, m: &StoredMessage, seen: &mut usize, hidden: usize, images_dir: &Path) -> ChatMessage {
    let mut msg = ChatMessage::user(content);
    for file in &m.images {
        let url = (*seen >= hidden).then(|| crate::files::data_url(images_dir, file)).flatten();
        *seen += 1;
        match url {
            Some(url) => msg.images.push(url),
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
    let mut known = HashMap::new();
    if mem.is_group {
        for npub in mem.distinct_authors() {
            let name = names.name_of(&npub).await;
            known.insert(npub, name);
        }
    }
    assemble(mem, system_prompt, &known, images_dir)
}

/// The message list, given each group author's name (`None` for one without).
fn assemble(
    mem: &ChannelMemory,
    system_prompt: &str,
    names: &HashMap<String, Option<String>>,
    images_dir: &Path,
) -> Vec<ChatMessage> {
    let hidden = hidden_pictures(mem);
    let mut seen = 0;
    let mut out = Vec::with_capacity(mem.messages.len() + 1);
    if !mem.is_group {
        out.push(ChatMessage::system(system_prompt));
        for m in &mem.messages {
            out.push(match m.role {
                Role::User => with_images(m.content.clone(), m, &mut seen, hidden, images_dir),
                role => ChatMessage { role, ..ChatMessage::user(m.content.clone()) },
            });
        }
        return out;
    }

    // One label per distinct author, disambiguating any display-name collisions with an npub tail.
    let mut label_for: HashMap<String, String> = HashMap::new();
    let mut name_counts: HashMap<String, u32> = HashMap::new();
    for npub in mem.distinct_authors() {
        let label = match names.get(&npub).cloned().flatten() {
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
        label_for.insert(npub, label);
    }

    out.push(ChatMessage::system(format!("{system_prompt}{GROUP_PROMPT}")));
    let mut introduced: HashSet<&str> = HashSet::new();
    for m in &mem.messages {
        out.push(match m.role {
            Role::Assistant => ChatMessage::assistant(m.content.clone()),
            _ => {
                let speaker = match m.author.as_deref() {
                    Some(npub) => {
                        let label = label_for.get(npub).map(String::as_str).unwrap_or("user");
                        if introduced.insert(npub) {
                            format!("{label} [{npub}]")
                        } else {
                            label.to_string()
                        }
                    }
                    None => "user".to_string(),
                };
                with_images(format!("{speaker}: {}", m.content), m, &mut seen, hidden, images_dir)
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("viktor-prompt-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn picture_turn(dir: &Path, i: usize) -> StoredMessage {
        let file = format!("m{i}-0.png");
        std::fs::write(dir.join(&file), [i as u8; 4]).unwrap();
        StoredMessage { images: vec![file], ..StoredMessage::user(format!("picture {i}"), None) }
    }

    #[test]
    fn old_pictures_are_hidden_a_batch_at_a_time() {
        let dir = temp_dir("batch");
        let shown = |n: usize| {
            let mem = ChannelMemory { is_group: false, messages: (0..n).map(|i| picture_turn(&dir, i)).collect() };
            let built = assemble(&mem, "sys", &HashMap::new(), &dir);
            built[1..].iter().map(|m| !m.images.is_empty()).collect::<Vec<_>>()
        };
        assert_eq!(shown(4), [true; 4]);
        assert_eq!(shown(7), [true; 7], "up to twice the batch less one");
        assert_eq!(shown(8), [false, false, false, false, true, true, true, true]);
        let mem = ChannelMemory { is_group: false, messages: (0..8).map(|i| picture_turn(&dir, i)).collect() };
        let built = assemble(&mem, "sys", &HashMap::new(), &dir);
        assert!(built[1].content.ends_with("[an earlier picture, no longer shown]"));
        assert!(built[8].images[0].starts_with("data:image/png;base64,"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_speaker_is_introduced_once_and_named_after() {
        let names: HashMap<String, Option<String>> =
            [("npub1alice".to_string(), Some("Alice".to_string())), ("npub1bob".to_string(), None)].into();
        let mem = ChannelMemory {
            is_group: true,
            messages: vec![
                StoredMessage::user("hi".into(), Some("npub1alice".into())),
                StoredMessage::user("yo".into(), Some("npub1bob".into())),
                StoredMessage::user("again".into(), Some("npub1alice".into())),
            ],
        };
        let built = assemble(&mem, "sys", &names, Path::new("."));
        let said: Vec<&str> = built[1..].iter().map(|m| m.content.as_str()).collect();
        assert_eq!(said, ["Alice [npub1alice]: hi", "npub1bob [npub1bob]: yo", "Alice: again"]);
        assert_eq!(built[0].content, format!("sys{GROUP_PROMPT}"));
    }

    /// A busy group, one turn at a time: each prompt starts with the whole of the one before,
    /// except right after a trim or a batch of pictures going out of view.
    #[test]
    fn each_prompt_starts_with_the_one_before() {
        let dir = temp_dir("prefix");
        let names: HashMap<String, Option<String>> =
            (0..6).map(|i| (format!("npub1user{i}"), (i % 2 == 0).then(|| format!("User {i}")))).collect();
        let mut mem = ChannelMemory { is_group: true, ..Default::default() };
        let mut before: Vec<ChatMessage> = Vec::new();
        let (mut misses, mut expected) = (0, 0);
        for step in 0..400 {
            let first = mem.messages.first().map(|m| m.content.clone());
            let hidden = hidden_pictures(&mem);
            let mut turn = StoredMessage::user(format!("message {step}"), Some(format!("npub1user{}", (step * 7 / 3) % 6)));
            if step % 9 == 0 {
                turn.images = picture_turn(&dir, step).images;
            }
            mem.push(turn);
            if step % 4 == 0 {
                mem.push_assistant(format!("reply {step}"));
            }
            mem.trim(50);
            let now = assemble(&mem, "sys", &names, &dir);
            if now.len() < before.len() || now[..before.len()] != before[..] {
                misses += 1;
                let trimmed = mem.messages.first().map(|m| m.content.clone()) != first;
                assert!(trimmed || hidden_pictures(&mem) != hidden, "step {step} rewrote the prompt's start");
            }
            if mem.messages.first().map(|m| m.content.clone()) != first || hidden_pictures(&mem) != hidden {
                expected += 1;
            }
            before = now;
        }
        assert!(misses <= expected && misses < 60, "{misses} rewrites in 400 turns");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Writes the requests a channel's next dozen turns would make, for a live cache check:
    /// `PROBE_MEM=<memory json> PROBE_IMAGES=<dir> PROBE_OUT=<dir> cargo test -- --ignored probe`.
    #[test]
    #[ignore]
    fn probe_requests() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let full: ChannelMemory = serde_json::from_str(&std::fs::read_to_string(var("PROBE_MEM")).unwrap()).unwrap();
        let (images, out) = (std::path::PathBuf::from(var("PROBE_IMAGES")), std::path::PathBuf::from(var("PROBE_OUT")));
        std::fs::create_dir_all(&out).unwrap();
        let start = full.messages.len().saturating_sub(12);
        let mut mem = ChannelMemory { is_group: true, messages: full.messages[..start].to_vec() };
        for (n, m) in full.messages[start..].iter().enumerate() {
            mem.push(m.clone());
            if n % 2 == 1 {
                let wire: Vec<_> = assemble(&mem, "You are Viktor.", &HashMap::new(), &images).iter().map(ChatMessage::wire).collect();
                std::fs::write(out.join(format!("req-{n:02}.json")), serde_json::to_vec(&wire).unwrap()).unwrap();
            }
        }
    }
}

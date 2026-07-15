//! Public (firehose) mode: monitor the open Nostr `kind:1` stream and reply to relevant posts.
//!
//! This is the one feature that lives outside Vector's encrypted DM/Community model, so it drops
//! to the raw relay client (shared with the SDK via `vector_core::state::nostr_client`). The SDK's
//! `listen` loop ignores `kind:1`, and notifications are a broadcast, so this runs as an
//! independent consumer alongside it.
//!
//! Pipeline per post: direct-mention / whitelist / blacklist / rate-limit / keyword gates →
//! fetch NIP-10 thread context → LLM relevance check (skipped when we're addressed) →
//! generate a reply with proper User/Assistant roles → post it with NIP-10 threading tags.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use nostr_sdk::prelude::{
    Client, Event, EventBuilder, EventId, Filter, FromBech32, Kind, PublicKey,
    RelayPoolNotification, Tag, TagKind, ToBech32,
};

use crate::config::Config;
use crate::llm::{ChatMessage, LLMClient};
use crate::names::NameResolver;
use crate::util::{now_secs, sanitize_reply};

/// Consecutive irrelevant posts from an author before we stop spending LLM credits on them.
const BLACKLIST_THRESHOLD: u32 = 10;
/// How deep to walk a thread when reconstructing context.
const MAX_THREAD_DEPTH: usize = 25;
const EVENT_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

const RELEVANCE_CHECK_SUFFIX: &str = r#"

Given the above context about who you are, evaluate the following public post (and its thread context if provided).
Is this post relevant to you, your interests, or something you could meaningfully contribute to?
Consider: Is it asking a question you can answer? Is it discussing a topic you're knowledgeable about? Would your response add value?
Also: If the direct post you're responding to looks like a bot, do not reply.

IMPORTANT: Reply with ONLY "yes" or "no" - nothing else."#;

/// Per-author "consecutive irrelevant" counter, persisted to `<data_dir>/blacklist.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Blacklist {
    scores: HashMap<String, u32>,
}

impl Blacklist {
    fn path(data_dir: &str) -> std::path::PathBuf {
        std::path::Path::new(data_dir).join("blacklist.json")
    }

    fn load(data_dir: &str) -> Self {
        match std::fs::read_to_string(Self::path(data_dir)) {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
                warn!("Failed to parse blacklist.json ({e}); starting empty");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    fn save(&self, data_dir: &str) {
        if let Err(e) = std::fs::create_dir_all(data_dir)
            .and_then(|_| std::fs::write(Self::path(data_dir), serde_json::to_vec_pretty(self).unwrap_or_default()))
        {
            warn!("Failed to save blacklist.json: {e}");
        }
    }

    fn is_blacklisted(&self, npub: &str) -> bool {
        self.scores.get(npub).is_some_and(|&c| c >= BLACKLIST_THRESHOLD)
    }

    fn count(&self, npub: &str) -> u32 {
        self.scores.get(npub).copied().unwrap_or(0)
    }

    fn increment(&mut self, npub: &str) -> u32 {
        let c = self.scores.entry(npub.to_string()).or_insert(0);
        *c += 1;
        *c
    }

    fn reset(&mut self, npub: &str) {
        self.scores.remove(npub);
    }

    fn blacklisted_count(&self) -> usize {
        self.scores.values().filter(|&&c| c >= BLACKLIST_THRESHOLD).count()
    }
}

/// One post in a reconstructed thread, ordered oldest-first by the caller.
#[derive(Clone)]
struct ThreadPost {
    id_hex: String,
    author: PublicKey,
    content: String,
    created_at: u64,
    is_ours: bool,
}

/// Shared state for the public-mode loop.
pub struct PublicMode {
    client: Client,
    cfg: Config,
    llm: LLMClient,
    names: NameResolver,
    blacklist: RwLock<Blacklist>,
    last_process: AtomicU64,
    my_pubkey: PublicKey,
    my_npub: String,
    my_hex: String,
}

impl PublicMode {
    /// Build the loop state. `bot_npub` is the bech32 npub the bot signs as.
    pub fn new(
        client: Client,
        cfg: Config,
        llm: LLMClient,
        names: NameResolver,
        bot_npub: &str,
    ) -> Result<Self> {
        let my_pubkey = PublicKey::from_bech32(bot_npub)
            .map_err(|e| anyhow::anyhow!("bad bot npub: {e}"))?;
        let blacklist = Blacklist::load(&cfg.data_dir);
        let n = blacklist.blacklisted_count();
        if n > 0 {
            info!("📛 Loaded blacklist: {n} authors blacklisted");
        }
        Ok(Self {
            client,
            my_hex: my_pubkey.to_hex(),
            my_npub: bot_npub.to_string(),
            my_pubkey,
            cfg,
            llm,
            names,
            blacklist: RwLock::new(blacklist),
            last_process: AtomicU64::new(0),
        })
    }

    /// Subscribe to public notes and process them until the relay pool closes.
    pub async fn run(self: Arc<Self>) {
        let filter = Filter::new().kind(Kind::TextNote).limit(0); // only new posts
        if let Err(e) = self.client.subscribe(filter, None).await {
            error!("Public mode: failed to subscribe to kind:1 ({e}); disabled");
            return;
        }
        info!("📡 Public mode: subscribed to the kind:1 firehose");

        let mut notifications = self.client.notifications();
        loop {
            match notifications.recv().await {
                Ok(RelayPoolNotification::Event { event, .. }) => {
                    if event.kind == Kind::TextNote && event.pubkey != self.my_pubkey {
                        self.handle_note(*event).await;
                    }
                }
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    debug!("Public mode: lagged, skipped {n} notifications");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    warn!("Public mode: notification channel closed; stopping");
                    return;
                }
            }
        }
    }

    async fn handle_note(&self, event: Event) {
        let content = event.content.clone();
        let author = event.pubkey;
        let event_id_hex = event.id.to_hex();
        let author_npub = author.to_bech32().unwrap_or_default();
        let short_id = &event_id_hex[..8.min(event_id_hex.len())];

        let is_whitelisted = self.cfg.is_public_whitelisted(&author_npub);
        // Direct mentions (npub/hex in content, or a p-tag) bypass rate limit + blacklist.
        let directly_mentioned = self.check_direct_mention(&event);

        if !is_whitelisted && !directly_mentioned {
            if self.blacklist.read().await.is_blacklisted(&author_npub) {
                debug!("🚫 Skipping blacklisted author {author_npub}");
                return;
            }
            if self.cfg.public_rate_limit_secs > 0 {
                let elapsed = now_secs().saturating_sub(self.last_process.load(Ordering::Relaxed));
                if elapsed < self.cfg.public_rate_limit_secs {
                    debug!("Rate limited ({elapsed}s < {}s)", self.cfg.public_rate_limit_secs);
                    return;
                }
            }
        }

        let hashtags = hashtags(&event);
        if !directly_mentioned && !self.cfg.matches_keywords(&content, &hashtags) {
            debug!("🔑 Skipping {short_id}: no keyword match");
            return;
        }

        if is_whitelisted {
            info!("📋 Whitelisted author: {author_npub}");
        }
        if directly_mentioned {
            info!("📣 Directly mentioned (bypasses rate limit): {short_id}");
        }

        let thread = self.fetch_thread_context(&event).await;
        let mentioned = directly_mentioned || self.replies_to_us(&event, &thread);

        // Whitelisted users replying to someone *else* shouldn't drag us into the conversation.
        if is_whitelisted && is_reply(&event) && !mentioned {
            debug!("📋 Skipping whitelisted reply to a third party: {author_npub}");
            return;
        }

        let relevant = if mentioned {
            info!("📣 Addressed directly; auto-relevant for {short_id}");
            true
        } else {
            self.check_relevance(&event, &thread, &author_npub, is_whitelisted).await
        };
        if !relevant {
            return;
        }

        info!("✅ {short_id} deemed relevant; generating response");

        // Successful interaction clears any accumulated "no" score.
        if !is_whitelisted {
            let score = self.blacklist.read().await.count(&author_npub);
            if score > 0 {
                let mut bl = self.blacklist.write().await;
                bl.reset(&author_npub);
                bl.save(&self.cfg.data_dir);
                info!("🔄 Reset blacklist score for {author_npub} (was {score})");
            }
        }
        // Only "random" (non-addressed) posts advance the rate-limit clock.
        if !is_whitelisted && !mentioned {
            self.last_process.store(now_secs(), Ordering::Relaxed);
        }

        self.generate_and_post(&event, &thread, author).await;
    }

    /// Ask the LLM whether a non-addressed post is worth replying to, updating the blacklist.
    async fn check_relevance(
        &self,
        event: &Event,
        thread: &[ThreadPost],
        author_npub: &str,
        is_whitelisted: bool,
    ) -> bool {
        let context = format_thread_simple(thread, &event.content, &event.pubkey);
        let messages = vec![
            ChatMessage::system(format!(
                "{}\n{}\n\n{}",
                self.cfg.public_system_prompt, RELEVANCE_CHECK_SUFFIX, context
            )),
            ChatMessage::user("Should I respond to this post? Answer only 'yes' or 'no'."),
        ];

        let relevant = match self.llm.chat(messages).await {
            Ok(resp) => resp.trim().to_lowercase().starts_with("yes"),
            Err(e) => {
                warn!("Relevance check failed: {e}");
                return false;
            }
        };

        if !relevant && !is_whitelisted {
            let mut bl = self.blacklist.write().await;
            let score = bl.increment(author_npub);
            bl.save(&self.cfg.data_dir);
            if score >= BLACKLIST_THRESHOLD {
                info!("📛 Blacklisted {author_npub} after {score} consecutive 'no'");
            }
        }
        relevant
    }

    /// Build the response with thread context, then post it (or log it under dry-run).
    async fn generate_and_post(&self, event: &Event, thread: &[ThreadPost], author: PublicKey) {
        let messages = self.build_thread_messages(thread, &event.content, &author).await;
        let reply = match self.llm.chat(messages).await {
            Ok(r) => sanitize_reply(&r),
            Err(e) => {
                error!("Failed to generate public reply: {e}");
                return;
            }
        };
        info!("💬 Generated reply: {reply}");

        let (root_id_hex, root_author, reply_to) = thread_reply_info(thread, &event.id.to_hex(), author);

        // Notify every participant (minus us), plus the author.
        let mut participants: Vec<PublicKey> =
            thread.iter().filter(|p| !p.is_ours).map(|p| p.author).collect();
        if !participants.contains(&author) {
            participants.push(author);
        }

        if self.cfg.public_dry_run {
            info!("🔸 DRY RUN — would reply to {} (root {})", event.id.to_hex(), root_id_hex);
            info!("🔸 Content: {reply}");
            return;
        }

        // A human-paced typing delay, ~75ms/char, floor 2s.
        let delay = (reply.chars().count() as u64 * 75).max(2000);
        tokio::time::sleep(Duration::from_millis(delay)).await;

        let mut builder = EventBuilder::text_note(&reply)
            .tag(e_marker_tag(&root_id_hex, "root", &root_author));
        if let Some((parent_id_hex, parent_author)) = reply_to {
            builder = builder.tag(e_marker_tag(&parent_id_hex, "reply", &parent_author));
        }
        for pk in participants {
            builder = builder.tag(Tag::public_key(pk));
        }

        match self.client.send_event_builder(builder).await {
            Ok(output) => info!("✅ Posted public reply: {}", output.id()),
            Err(e) => error!("❌ Failed to post public reply: {e}"),
        }
    }

    /// LLM messages for the response: thread context as User/Assistant turns + the current post.
    async fn build_thread_messages(
        &self,
        thread: &[ThreadPost],
        current_content: &str,
        current_author: &PublicKey,
    ) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system(&self.cfg.public_system_prompt)];
        for post in thread {
            if post.is_ours {
                messages.push(ChatMessage::assistant(post.content.clone()));
            } else {
                let label = self.label_for(&post.author).await;
                messages.push(ChatMessage::user(format!("{label}: {}", post.content)));
            }
        }
        let label = self.label_for(current_author).await;
        messages.push(ChatMessage::user(format!("{label}: {current_content}")));
        messages
    }

    async fn label_for(&self, pubkey: &PublicKey) -> String {
        match pubkey.to_bech32() {
            Ok(npub) => self.names.label_of(&npub).await,
            Err(_) => pubkey.to_hex(),
        }
    }

    // --- mention detection ---

    fn check_direct_mention(&self, event: &Event) -> bool {
        if event.content.contains(&self.my_npub) || event.content.contains(&self.my_hex) {
            return true;
        }
        event.tags.iter().any(|t| {
            let v = t.clone().to_vec();
            v.len() >= 2 && v[0] == "p" && v[1] == self.my_hex
        })
    }

    /// Whether this post replies to one of our posts found in the thread context.
    fn replies_to_us(&self, event: &Event, thread: &[ThreadPost]) -> bool {
        for tag in event.tags.iter() {
            let v = tag.clone().to_vec();
            if v.len() >= 2 && v[0] == "e" && thread.iter().any(|p| p.is_ours && p.id_hex == v[1]) {
                return true;
            }
        }
        false
    }

    // --- thread reconstruction ---

    /// Walk NIP-10 `e` tags (root/reply markers, with legacy positional fallback) up to
    /// [`MAX_THREAD_DEPTH`], returning the posts oldest-first.
    async fn fetch_thread_context(&self, event: &Event) -> Vec<ThreadPost> {
        let mut seed = referenced_event_ids(event);
        if seed.is_empty() {
            return Vec::new();
        }

        let mut posts: Vec<ThreadPost> = Vec::new();
        let mut fetched: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut depth = 0;

        while !seed.is_empty() && depth < MAX_THREAD_DEPTH {
            let mut next = Vec::new();
            for id_hex in std::mem::take(&mut seed) {
                if !fetched.insert(id_hex.clone()) {
                    continue;
                }
                let fetched_event = match tokio::time::timeout(
                    EVENT_FETCH_TIMEOUT,
                    self.fetch_event(&id_hex),
                )
                .await
                {
                    Ok(Some(ev)) => ev,
                    _ => continue,
                };
                posts.push(ThreadPost {
                    id_hex: fetched_event.id.to_hex(),
                    author: fetched_event.pubkey,
                    content: fetched_event.content.clone(),
                    created_at: fetched_event.created_at.as_secs(),
                    is_ours: fetched_event.pubkey == self.my_pubkey,
                });
                for tag in fetched_event.tags.iter() {
                    let v = tag.clone().to_vec();
                    if v.len() >= 2 && v[0] == "e" && !fetched.contains(&v[1]) {
                        next.push(v[1].clone());
                    }
                }
            }
            seed = next;
            depth += 1;
        }

        posts.sort_by_key(|p| p.created_at);
        if !posts.is_empty() {
            info!("📜 Fetched {} posts of thread context", posts.len());
        }
        posts
    }

    async fn fetch_event(&self, id_hex: &str) -> Option<Event> {
        let id = EventId::from_hex(id_hex).ok()?;
        self.client
            .fetch_events(Filter::new().id(id), EVENT_FETCH_TIMEOUT)
            .await
            .ok()?
            .into_iter()
            .next()
    }
}

// --- free helpers ---

/// Extract the event ids to seed thread reconstruction: root + reply markers, else legacy
/// positional (first = root, last = direct parent).
fn referenced_event_ids(event: &Event) -> Vec<String> {
    let mut root = None;
    let mut reply = None;
    let mut positional = Vec::new();
    for tag in event.tags.iter() {
        let v = tag.clone().to_vec();
        if v.len() < 2 || v[0] != "e" {
            continue;
        }
        match v.get(3).map(String::as_str) {
            Some("root") => root = Some(v[1].clone()),
            Some("reply") => reply = Some(v[1].clone()),
            _ => positional.push(v[1].clone()),
        }
    }

    let mut out = Vec::new();
    if let Some(r) = root {
        out.push(r);
    }
    if let Some(r) = reply {
        if !out.contains(&r) {
            out.push(r);
        }
    }
    if out.is_empty() {
        if let Some(first) = positional.first() {
            out.push(first.clone());
        }
        if let Some(last) = positional.last() {
            if !out.contains(last) {
                out.push(last.clone());
            }
        }
    }
    out
}

fn hashtags(event: &Event) -> Vec<String> {
    event
        .tags
        .iter()
        .filter_map(|t| {
            let v = t.clone().to_vec();
            (v.len() >= 2 && v[0] == "t").then(|| v[1].clone())
        })
        .collect()
}

fn is_reply(event: &Event) -> bool {
    event.tags.iter().any(|t| {
        let v = t.clone().to_vec();
        v.len() >= 2 && v[0] == "e"
    })
}

/// `(root_id, root_author, Some((parent_id, parent_author)))` for NIP-10 reply tagging.
fn thread_reply_info(
    thread: &[ThreadPost],
    current_id_hex: &str,
    current_author: PublicKey,
) -> (String, PublicKey, Option<(String, PublicKey)>) {
    match thread.first() {
        // With context, root is the oldest post and we reply directly to the current post.
        Some(root) => (
            root.id_hex.clone(),
            root.author,
            Some((current_id_hex.to_string(), current_author)),
        ),
        // No context: the current post is itself the root.
        None => (current_id_hex.to_string(), current_author, None),
    }
}

/// Plain-text thread rendering used for the (cheaper) relevance check.
fn format_thread_simple(thread: &[ThreadPost], current: &str, current_author: &PublicKey) -> String {
    if thread.is_empty() {
        return format!("Post:\n\"{current}\"");
    }
    let mut s = String::from("Thread context (oldest to newest):\n\n");
    for (i, post) in thread.iter().enumerate() {
        let who = if post.is_ours {
            "You (the bot)".to_string()
        } else {
            short_bech32(&post.author)
        };
        s.push_str(&format!("[{}] {who} wrote:\n\"{}\"\n\n", i + 1, post.content));
    }
    s.push_str(&format!("[Current] {} wrote:\n\"{current}\"", short_bech32(current_author)));
    s
}

fn short_bech32(pk: &PublicKey) -> String {
    pk.to_bech32()
        .map(|s| format!("{}…", &s[..16.min(s.len())]))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// A NIP-10 markered `e` tag: `["e", <id>, "", <marker>, <author-hex>]`.
fn e_marker_tag(id_hex: &str, marker: &str, author: &PublicKey) -> Tag {
    Tag::custom(
        TagKind::custom("e"),
        vec![id_hex.to_string(), String::new(), marker.to_string(), author.to_hex()],
    )
}

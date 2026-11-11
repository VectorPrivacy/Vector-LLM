//! Vector LLM Bot — an OpenAI-compatible chatbot for Vector.
//!
//! Built on the Vector SDK, which unifies DMs and Community group channels behind one
//! `on_message` handler. The bot:
//!   - replies to every DM, keeping a rolling per-conversation history;
//!   - participates in group channels when mentioned or replied to, understanding who said
//!     what via per-speaker labels + an npub roster (see [`prompt`]);
//!   - optionally monitors the public Nostr `kind:1` firehose (see [`public`]).

mod config;
mod files;
mod llm;
mod memory;
mod names;
mod prompt;
#[cfg(feature = "public")]
mod public;
mod util;

use std::sync::Arc;

use anyhow::Result;
use env_logger::Env;
use log::{error, info, warn};

use vector_sdk::nostr::{FromBech32, PublicKey, SecretKey, ToBech32};
use vector_sdk::{BotEvent, IncomingMessage, VectorBot};

use crate::config::{Config, InvitePolicyKind};
use crate::llm::LLMClient;
use crate::memory::{MemoryStore, StoredMessage};
use crate::names::NameResolver;
use crate::util::sanitize_reply;

/// Shared, immutable-ish state handed to every message handler invocation.
struct Ctx {
    cfg: Config,
    llm: LLMClient,
    names: NameResolver,
    store: MemoryStore,
    bot_npub: String,
    bot_hex: String,
    /// Where pictures people send are kept for the model to look at.
    images_dir: std::path::PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    // `.env` is loaded inside `Config::load()`.

    let cfg = Config::load()?;
    info!(
        "Config: model={}, base_url={}, history_limit={}",
        cfg.model, cfg.base_url, cfg.history_limit
    );
    info!(
        "Modes: dm={}, group={}, public={}",
        cfg.dm_mode, cfg.group_mode, cfg.public_mode
    );
    if cfg.public_mode && cfg.public_dry_run {
        info!("🔸 Public DRY RUN: responses are generated but not posted");
    }
    if !cfg.dm_mode && !cfg.group_mode && !cfg.public_mode {
        warn!("⚠️  No modes enabled (dm/group/public all off); the bot will not respond to anything.");
    }

    // Build the bot: an explicit key (nsec / hex / mnemonic) wins; otherwise the SDK loads — or, on
    // first run, mints and persists — a stable identity under `<data_dir>/identity.nsec`.
    let mut builder = VectorBot::builder().data_dir(&cfg.data_dir);
    if let Some(key) = normalized_key(&cfg)? {
        builder = builder.key(key);
    }
    builder = match cfg.invite_policy {
        InvitePolicyKind::Public => builder.public(),
        InvitePolicyKind::Whitelist => builder.whitelist(cfg.invite_whitelist.clone()),
        InvitePolicyKind::Manual => builder,
    };
    let bot = builder.build().await?;
    info!("🤖 Online as {}", bot.npub());

    // Broadcast the bot's profile (kind:0) so clients show a name/avatar and a bot badge.
    if cfg.publish_profile {
        match bot.update_profile(&cfg.bot_name, &cfg.bot_avatar, &cfg.bot_banner, &cfg.bot_about).await {
            Ok(()) => info!("📇 Published profile as '{}'", cfg.bot_name),
            Err(e) => warn!("Failed to publish profile: {e}"),
        }
    }

    let bot_npub = bot.npub().to_string();
    let bot_hex = PublicKey::from_bech32(&bot_npub)?.to_hex();
    let llm = LLMClient::new(
        cfg.base_url.clone(),
        cfg.api_key.clone(),
        cfg.model.clone(),
        cfg.temperature,
        cfg.max_tokens,
    );
    let names = NameResolver::new(bot.clone());
    let store = MemoryStore::new(&cfg.data_dir);

    // Public (firehose) mode runs as an independent consumer on the shared relay client, alongside
    // the SDK's own DM/Community listen loop.
    #[cfg(feature = "public")]
        if cfg.public_mode {
            match vector_sdk::vector_core::state::nostr_client() {
                Some(client) => {
                    let pm = public::PublicMode::new(
                        client,
                        cfg.clone(),
                        llm.clone(),
                        names.clone(),
                        &bot_npub,
                    )?;
                    tokio::spawn(Arc::new(pm).run());
                }
                None => warn!("Public mode enabled but no relay client is available; skipping it"),
            }
        }

    // Asking for a mode this binary cannot provide must be loud. Silently
    // ignoring PUBLIC_MODE=true would look identical to "the firehose is quiet",
    // and you'd debug relays for an hour before suspecting the build.
    #[cfg(not(feature = "public"))]
    if cfg.public_mode {
        anyhow::bail!(
            "PUBLIC_MODE=true but this binary was built without the `public` feature.\n\
             Public mode is disabled by default since SDK 0.6: nostr-sdk 0.45 removed signing \n\
             from `Client`, so the firehose path cannot post replies without reworking how the \n\
             bot signs events.\n\
             Either set PUBLIC_MODE=false, or rebuild with `cargo build --release --features public` \n\
             once that rework is done."
        );
    }

    let images_dir = std::path::Path::new(&cfg.data_dir).join("images");
    let ctx = Arc::new(Ctx { cfg, llm, names, store, bot_npub, bot_hex, images_dir });

    // Viktor's one slash command. Declaring it publishes the kind-10304
    // manifest, so `/clear` shows in a client's `/` picker; ordinary
    // conversation (a mention, a reply, a DM) still flows through on_message
    // below and is what actually makes him talk.
    {
        let ctx = ctx.clone();
        bot.command("clear", "Forget this conversation and start fresh").run(move |c| {
            let ctx = ctx.clone();
            async move {
                ctx.store.update(&c.msg.chat_id, c.msg.is_group, |m| m.clear());
                let _ = c.reply("Conversation history cleared.").await;
            }
        });
    }

    info!("Connecting…");

    // `on_event` rather than `on_message`: a superset that also carries Ready and ChannelKeyed.
    // Slash commands are consumed upstream by the SDK on both paths, so /clear is unaffected.
    bot.on_event(move |bot, event| {
        let ctx = ctx.clone();
        async move {
            match event {
                BotEvent::Message(msg) => handle_message(bot, msg, ctx).await,
                // The live subscription is registered: from here we actually hear traffic, and
                // community state has been caught up, so this is the first honest moment to
                // report what we can see.
                BotEvent::Ready { communities } => {
                    info!("✅ Subscribed and listening ({communities} communities)");
                    report_channel_inventory(&bot).await;
                }
                BotEvent::ChannelKeyed { community_id, channel_id, backfilled } => {
                    handle_channel_keyed(bot, ctx, &community_id, &channel_id, backfilled).await
                }
                _ => {}
            }
        }
    })
    .await?;

    Ok(())
}

/// Log every community channel and how this bot stands with it.
///
/// An unreadable private channel is two different situations, and only one is a problem:
/// we were never granted access (normal, and most private channels are this), or we hold the
/// access role and the key hasn't landed (the state that otherwise looks like an idle channel).
/// The channel's access roster tells them apart, so only the second warns.
async fn report_channel_inventory(bot: &VectorBot) {
    for community in bot.communities().await {
        let channels = community.channels().await;
        info!(
            "🏘  community {} — {} channel(s), {} private",
            &community.id()[..8.min(community.id().len())],
            channels.len(),
            channels.iter().filter(|c| c.is_private()).count()
        );

        let mut awaiting_key = Vec::new();
        for ch in &channels {
            let kind = if !ch.is_private() {
                "public"
            } else if ch.is_readable() {
                "private, readable"
            } else if community
                .channel_members(ch.id())
                .iter()
                .any(|m| m.npub() == bot.npub())
            {
                awaiting_key.push(ch.name());
                "private, GRANTED — awaiting key"
            } else {
                "private, no access"
            };
            info!("   #{} ({kind}, epoch {})", ch.name(), ch.epoch());
        }

        if !awaiting_key.is_empty() {
            warn!(
                "🔑 access granted but no key yet for {} channel(s): {}",
                awaiting_key.len(),
                awaiting_key.join(", ")
            );
        }
    }
}

/// A private channel's key arrived, so it just became readable. `backfilled` messages predating the
/// grant were pulled into local state, and they are history rather than delivery: they never arrive
/// as live events. Seed the conversation memory from them so the first reply there isn't contextless.
async fn handle_channel_keyed(
    bot: VectorBot,
    ctx: Arc<Ctx>,
    community_id: &str,
    channel_id: &str,
    backfilled: usize,
) {
    let name = bot
        .community(community_id)
        .channels()
        .await
        .into_iter()
        .find(|c| c.id() == channel_id)
        .map(|c| c.name().to_string())
        .unwrap_or_default();
    info!("🔓 private channel #{name} ({channel_id}) is now readable, {backfilled} message(s) back-filled");

    let seeded = seed_memory_from_history(&bot, &ctx, channel_id).await;
    if seeded > 0 {
        info!("📖 seeded {seeded} message(s) of #{name} history into context");
    }

    if ctx.cfg.announce_channel_access {
        let _ = bot.channel(channel_id).send("Hey, I can see this channel now.").await;
    }
}

/// Fill an empty channel memory from back-filled history. Returns how many turns were seeded;
/// 0 if memory already had content.
async fn seed_memory_from_history(bot: &VectorBot, ctx: &Ctx, chat_id: &str) -> usize {
    if !ctx.store.read(chat_id, true).messages.is_empty() {
        return 0;
    }

    // Local, pull-only, newest-first-then-chronological: exactly the window we want to prime with.
    let limit = ctx.cfg.history_limit;
    let recent = bot.channel(chat_id).history(limit).await;
    if recent.is_empty() {
        return 0;
    }

    ctx.store.update(chat_id, true, |m| {
        for msg in &recent {
            let text = msg.content.trim();
            if text.is_empty() {
                continue;
            }
            if msg.mine {
                m.push_assistant(text.to_string());
            } else {
                m.push_user(text.to_string(), msg.npub.clone());
            }
        }
        m.trim(limit);
    });
    recent.len()
}

/// How recently a picture or file sent in a group, without addressing us, counts as what the next
/// message that does address us is about.
const RECENT_ATTACHMENT_MS: u64 = 10 * 60_000;

/// How long a file sent on its own waits for the words that usually follow it (a caption arrives as
/// its own message): those answer for both, rather than two replies.
const FILE_FOLLOW_UP: std::time::Duration = std::time::Duration::from_secs(5);

async fn handle_message(bot: VectorBot, msg: IncomingMessage, ctx: Arc<Ctx>) {
    info!(
        "📥 recv group={} mine={} chat={} files={} text={:?}",
        msg.is_group,
        msg.is_mine(),
        &msg.chat_id[..12.min(msg.chat_id.len())],
        msg.message.attachments.len(),
        msg.text().chars().take(50).collect::<String>()
    );
    if msg.is_mine() {
        return;
    }
    let raw = msg.text().trim();
    let attached = !msg.message.attachments.is_empty();
    if raw.is_empty() && !attached {
        return;
    }
    let is_group = msg.is_group;

    // Mode gating.
    if is_group && !ctx.cfg.group_mode {
        return;
    }
    if !is_group && !ctx.cfg.dm_mode {
        return;
    }

    let chat_id = msg.chat_id.clone();
    let author = msg.message.npub.clone();
    let limit = ctx.cfg.history_limit;
    // Already in memory: a reply that arrived first pulled it in as what it answers.
    if ctx.store.read(&chat_id, is_group).holds(Some(&msg.message.id)) {
        return;
    }

    // In a group channel we ingest *every* message for context, but only engage when addressed
    // (mentioned or replied to). An unaddressed message is recorded and we stop there; its files
    // are read only if someone asks us about them.
    if is_group && !is_addressed(&bot, &msg, &ctx.bot_npub, &ctx.bot_hex).await {
        info!("🔇 group msg not addressed to us — recorded for context, no reply");
        let mut turn = StoredMessage::user(with_file_names(raw, &msg.message), author);
        turn.at = msg.message.at;
        turn.id = Some(msg.message.id.clone());
        turn.unread_from = attached.then(|| msg.message.id.clone());
        ctx.store.update(&chat_id, is_group, |m| {
            m.push(turn);
            m.trim(limit);
        });
        return;
    }

    // Drop our own mention from group text so it doesn't pollute the stored context. A bare ping
    // ("@Viktor" with nothing else) strips to empty — still respond, falling back to the raw text
    // so the model sees it was addressed and answers from the conversation context.
    let mut text = if is_group {
        strip_self_mentions(raw, &ctx.bot_npub, &ctx.bot_hex)
    } else {
        raw.to_string()
    };
    if text.is_empty() && !attached {
        text = raw.to_string();
    }

    if ctx.cfg.typing_indicator {
        let _ = msg.channel().typing().await;
    }

    // What came with this message, and anything earlier it points at: the message it replies to,
    // and pictures or files sent just before without addressing us ("look at this, @Viktor").
    let read = files::read(&bot, &msg.message, &ctx.images_dir, ctx.cfg.vision).await;
    let mut turn = StoredMessage::user(join(&text, &read.text), author);
    turn.images = read.images;
    turn.at = msg.message.at;
    turn.id = Some(msg.message.id.clone());
    let earlier = read_earlier_attachments(&bot, &ctx, &msg, turn.author.as_deref()).await;

    // Record the user turn (race-safe) and take a trimmed snapshot for the prompt.
    let file_only = text.is_empty() && attached;
    let at = turn.at;
    let mut mem = ctx.store.update(&chat_id, is_group, |m| {
        for t in earlier {
            m.push(t);
        }
        m.push(turn);
        m.trim(limit);
    });
    if file_only {
        tokio::time::sleep(FILE_FOLLOW_UP).await;
        mem = ctx.store.read(&chat_id, is_group);
        if mem.messages.iter().any(|m| m.role == llm::Role::User && m.at > at) {
            info!("📎 file in {chat_id} followed by a message: that one answers for both");
            return;
        }
    }

    let messages = prompt::build_messages(&mem, &ctx.cfg.system_prompt, &ctx.names, &ctx.images_dir).await;

    info!("⚙️  engaging {chat_id} (group={is_group}) — calling LLM ({} ctx msgs)", messages.len());
    match ctx.llm.chat(messages).await {
        Ok(reply) => {
            let reply = sanitize_reply(&reply);
            if reply.is_empty() {
                warn!("⚠️  LLM returned an EMPTY reply for {chat_id}; nothing sent");
                return;
            }
            info!("💬 replying to {chat_id} ({} chars)", reply.chars().count());
            ctx.store.update(&chat_id, is_group, |m| {
                m.push_assistant(reply.clone());
                m.trim(limit);
            });
            // Group channels get a threaded reply (clarifies who we answer); DMs get a plain message.
            let sent = if is_group {
                msg.reply(&reply).await.is_ok()
            } else {
                msg.channel().send(&reply).await.is_ok()
            };
            if !sent {
                warn!("Failed to send reply to {chat_id}");
            }
        }
        Err(e) => error!("❌ LLM request failed for {chat_id}: {e}"),
    }
}

/// `text` and what its attachments came to, one after the other.
fn join(text: &str, read: &str) -> String {
    match (text.is_empty(), read.is_empty()) {
        (_, true) => text.to_string(),
        (true, false) => read.to_string(),
        (false, false) => format!("{text}\n\n{read}"),
    }
}

/// A message's text with the names of what it attached, for a turn whose files aren't read yet.
fn with_file_names(text: &str, msg: &vector_sdk::Message) -> String {
    if msg.attachments.is_empty() {
        return text.to_string();
    }
    let names: Vec<String> = msg
        .attachments
        .iter()
        .map(|a| if a.name.trim().is_empty() { format!("a .{} file", a.extension) } else { a.name.clone() })
        .collect();
    join(text, &format!("[attached: {}]", names.join(", ")))
}

/// Read the attachments this message points at that nobody has read yet: the message it replies
/// to, and recent unaddressed messages from the same sender that carried files. A turn memory holds
/// is read in place, where it was said; a reply target memory doesn't hold comes back as a turn of
/// its own, to go just before this message.
async fn read_earlier_attachments(bot: &VectorBot, ctx: &Ctx, msg: &IncomingMessage, sender: Option<&str>) -> Vec<StoredMessage> {
    let replied_to = msg.message.replied_to.as_str();
    let now = msg.message.at;
    let mem = ctx.store.read(&msg.chat_id, msg.is_group);
    let unread: Vec<String> = mem
        .messages
        .iter()
        .rev()
        .take(6)
        .filter_map(|m| m.unread_from.clone().filter(|id| id == replied_to || (m.at + RECENT_ATTACHMENT_MS >= now && m.author.as_deref() == sender)))
        .collect();
    let fetch_reply = !replied_to.is_empty() && !mem.holds(Some(replied_to));
    if unread.is_empty() && !fetch_reply {
        return Vec::new();
    }
    let history = bot.core().get_messages(&msg.chat_id, 200, 0).await;
    let find = |id: &str| history.iter().find(|m| m.id == id && !m.mine && !m.attachments.is_empty());
    for id in unread {
        let Some(earlier) = find(&id) else { continue };
        let read = files::read(bot, earlier, &ctx.images_dir, ctx.cfg.vision).await;
        ctx.store.update(&msg.chat_id, msg.is_group, |m| {
            let Some(t) = m.messages.iter_mut().find(|t| t.unread_from.as_deref() == Some(id.as_str())) else { return };
            t.content = join(t.content.trim_end(), &read.text);
            t.images.extend(read.images);
            t.unread_from = None;
        });
    }
    let mut before = Vec::new();
    if let Some(earlier) = fetch_reply.then(|| find(replied_to)).flatten() {
        let read = files::read(bot, earlier, &ctx.images_dir, ctx.cfg.vision).await;
        let mut t = StoredMessage::user(join(earlier.content.trim(), &read.text), earlier.npub.clone());
        t.images = read.images;
        t.at = earlier.at;
        t.id = Some(earlier.id.clone());
        before.push(t);
    }
    before
}

/// In a group channel, are we addressed? True if the message mentions our npub/hex in its text, or
/// replies to one of our own messages.
///
/// Reply detection can't rely on `replied_to_npub`: in a community the parent's stored author is an
/// epoch pseudonym (or unset), not our real npub. Instead we look the parent up in the channel's
/// message store and check its `mine` flag, which the core sets correctly regardless of pseudonyms.
async fn is_addressed(
    bot: &VectorBot,
    msg: &IncomingMessage,
    bot_npub: &str,
    bot_hex: &str,
) -> bool {
    // Cheap signals first: a mention in the text, or an already-resolved reply author.
    let text = msg.text();
    if text.contains(bot_npub) || text.contains(bot_hex) {
        return true;
    }
    if msg.message.replied_to_npub.as_deref() == Some(bot_npub) {
        return true;
    }

    // Reply to a message we authored — only query the store when this actually is a reply.
    let replied_to = &msg.message.replied_to;
    if replied_to.is_empty() {
        return false;
    }
    let to_own = bot
        .core()
        .get_messages(&msg.chat_id, 200, 0)
        .await
        .iter()
        .any(|m| m.mine && &m.id == replied_to);
    log::debug!(
        "group reply: replied_to={}… → to_own={to_own}",
        &replied_to[..8.min(replied_to.len())]
    );
    to_own
}

/// Strip our own `nostr:npub…` / `npub…` / hex mention tokens from text.
fn strip_self_mentions(text: &str, bot_npub: &str, bot_hex: &str) -> String {
    let mut out = text.to_string();
    for token in [
        format!("nostr:{bot_npub}"),
        bot_npub.to_string(),
        format!("nostr:{bot_hex}"),
        bot_hex.to_string(),
    ] {
        out = out.replace(&token, "");
    }
    out.trim().trim_start_matches('@').trim().to_string()
}

/// Resolve the configured secret into a form the builder accepts (`nsec…` or mnemonic), converting
/// a raw hex secret key to `nsec`. `None` means "let the SDK manage a persistent identity".
fn normalized_key(cfg: &Config) -> Result<Option<String>> {
    let Some(raw) = cfg.secret_key.as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };

    // nsec or a (space-separated) BIP-39 mnemonic — the builder handles both directly.
    if raw.starts_with("nsec1") || raw.contains(' ') {
        return Ok(Some(raw.to_string()));
    }

    // Otherwise treat it as a hex secret key and convert to nsec.
    let bytes = hex::decode(raw)
        .map_err(|_| anyhow::anyhow!("VECTOR_NSEC is not a valid nsec, mnemonic, or hex secret"))?;
    let sk = SecretKey::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("invalid hex secret key: {e}"))?;
    let nsec = sk.to_bech32().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Some(nsec))
}

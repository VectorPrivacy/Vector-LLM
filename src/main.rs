//! Vector LLM Bot — an OpenAI-compatible chatbot for Vector.
//!
//! Built on the Vector SDK, which unifies DMs and Community group channels behind one
//! `on_message` handler. The bot:
//!   - replies to every DM, keeping a rolling per-conversation history;
//!   - participates in group channels when mentioned or replied to, understanding who said
//!     what via per-speaker labels + an npub roster (see [`prompt`]);
//!   - optionally monitors the public Nostr `kind:1` firehose (see [`public`]).

mod config;
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
use vector_sdk::{IncomingMessage, VectorBot};

use crate::config::{Config, InvitePolicyKind};
use crate::llm::LLMClient;
use crate::memory::MemoryStore;
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
        if bot
            .update_profile(&cfg.bot_name, &cfg.bot_avatar, &cfg.bot_banner, &cfg.bot_about)
            .await
        {
            info!("📇 Published profile as '{}'", cfg.bot_name);
        } else {
            warn!("Failed to publish profile");
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

    let ctx = Arc::new(Ctx { cfg, llm, names, store, bot_npub, bot_hex });

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

    info!("Listening for messages…");

    // One handler for DMs and Community group channels alike. The SDK runs each invocation in its
    // own task, so a slow LLM call never blocks other conversations.
    bot.on_message(move |bot, msg| {
        let ctx = ctx.clone();
        async move {
            handle_message(bot, msg, ctx).await;
        }
    })
    .await?;

    Ok(())
}

async fn handle_message(bot: VectorBot, msg: IncomingMessage, ctx: Arc<Ctx>) {
    info!(
        "📥 recv group={} mine={} chat={} text={:?}",
        msg.is_group,
        msg.is_mine(),
        &msg.chat_id[..12.min(msg.chat_id.len())],
        msg.text().chars().take(50).collect::<String>()
    );
    if msg.is_mine() {
        return;
    }
    let raw = msg.text().trim();
    if raw.is_empty() {
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

    // In a group channel we ingest *every* message for context, but only engage when addressed
    // (mentioned or replied to). An unaddressed message is recorded and we stop there.
    if is_group && !is_addressed(&bot, &msg, &ctx.bot_npub, &ctx.bot_hex).await {
        info!("🔇 group msg not addressed to us — recorded for context, no reply");
        ctx.store.update(&chat_id, is_group, |m| {
            m.push_user(raw.to_string(), author);
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
    if text.is_empty() {
        text = raw.to_string();
    }

    // Record the user turn (race-safe) and take a trimmed snapshot for the prompt.
    let mem = ctx.store.update(&chat_id, is_group, |m| {
        m.push_user(text, author);
        m.trim(limit);
    });

    if ctx.cfg.typing_indicator {
        let _ = msg.channel().typing().await;
    }

    let messages = prompt::build_messages(&mem, &ctx.cfg.system_prompt, &ctx.names).await;

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

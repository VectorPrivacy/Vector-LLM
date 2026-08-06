//! Configuration, loaded once from the environment. [`load_env_file`] folds `.env` into the
//! process environment first, so everything below is a plain `std::env::var` read.

use anyhow::{bail, Result};
use std::str::FromStr;

/// How the bot handles inbound Community invites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvitePolicyKind {
    /// Auto-accept invites from anyone (the bot is openly addable to communities).
    Public,
    /// Auto-accept only invites from npubs in `invite_whitelist`.
    Whitelist,
    /// Never auto-accept; invites are parked for manual handling.
    Manual,
}

#[derive(Clone)]
pub struct Config {
    // --- LLM endpoint ---
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    /// Cap on response length. Important for reasoning models (else replies truncate or come back
    /// empty). Sent only when set.
    pub max_tokens: Option<u32>,

    // --- Context window + storage ---
    /// Max user/assistant turns retained per channel for LLM context.
    pub history_limit: usize,
    /// Root directory for the SDK's identity + database and the bot's per-channel memory.
    pub data_dir: String,

    // --- Identity ---
    /// Secret key for the bot (`nsec1…`, hex, or a BIP-39 mnemonic). When unset, the SDK
    /// mints and persists a stable identity under `data_dir/identity.nsec`.
    pub secret_key: Option<String>,
    /// Whether to (re)broadcast the bot's profile (kind:0) on startup.
    pub publish_profile: bool,
    pub bot_name: String,
    pub bot_about: String,
    pub bot_avatar: String,
    pub bot_banner: String,

    // --- Behaviour ---
    pub typing_indicator: bool,
    /// Persona prompt used for both DMs and group channels.
    pub system_prompt: String,

    // --- Modes ---
    /// Respond to direct messages.
    pub dm_mode: bool,
    /// Participate in Community group channels (only when mentioned or replied to).
    pub group_mode: bool,

    // --- Community invites ---
    pub invite_policy: InvitePolicyKind,
    pub invite_whitelist: Vec<String>,

    // --- Private channels ---
    /// Post a short greeting when a private channel's key arrives, so the admin who granted
    /// access sees it worked. Off by default: an unprompted message in a private channel is
    /// the caller's call to make, and the startup/keyed log lines already make it diagnosable.
    pub announce_channel_access: bool,

    // --- Public firehose mode (kind:1 monitoring) ---
    // Always parsed (so PUBLIC_MODE=true can be rejected with a clear error rather than silently
    // ignored), but only *read* by the feature-gated `public` module.
    pub public_mode: bool,
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub public_system_prompt: String,
    pub public_dry_run: bool,
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub public_rate_limit_secs: u64,
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub public_whitelist: Vec<String>,
    /// If non-empty, only public posts containing one of these keywords are considered.
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub public_keywords: Vec<String>,
}

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Keep responses concise and factual.";

const DEFAULT_PUBLIC_SYSTEM_PROMPT: &str =
    "You are a friendly and knowledgeable participant in public discussions. Keep responses \
     concise, relevant, and engaging. Be helpful but not intrusive. Match the tone of the \
     conversation.";

const DEFAULT_BOT_ABOUT: &str =
    "I'm Viktor, an AI assistant on Vector.\n\nBest employee at https://docs.vectorapp.io";

impl Config {
    pub fn load() -> Result<Self> {
        load_env_file();

        let base_url = env_str("LLM_BASE_URL", "https://api.openai.com/v1");
        if base_url.is_empty() {
            bail!("LLM_BASE_URL cannot be empty");
        }
        let model = env_str("LLM_MODEL", "gpt-4o-mini");
        if model.is_empty() {
            bail!("LLM_MODEL cannot be empty");
        }

        let invite_policy = match env_str("INVITE_POLICY", "public").to_ascii_lowercase().as_str() {
            "whitelist" => InvitePolicyKind::Whitelist,
            "manual" => InvitePolicyKind::Manual,
            _ => InvitePolicyKind::Public,
        };

        Ok(Self {
            base_url,
            api_key: env_opt("LLM_API_KEY"),
            model,
            temperature: env_parse("LLM_TEMPERATURE", 0.2_f32),
            max_tokens: env_opt("LLM_MAX_TOKENS").and_then(|v| v.parse().ok()),

            history_limit: env_parse("HISTORY_LIMIT", 16_usize).max(2),
            data_dir: env_str("DATA_DIR", "data"),

            // VECTOR_NSEC is the canonical name; VECTOR_SECRET_KEY is accepted for back-compat.
            secret_key: env_opt("VECTOR_NSEC").or_else(|| env_opt("VECTOR_SECRET_KEY")),
            publish_profile: env_bool("PUBLISH_PROFILE", true),
            bot_name: env_str("BOT_NAME", "Viktor"),
            bot_about: unescape(&env_str("BOT_ABOUT", DEFAULT_BOT_ABOUT)),
            bot_avatar: env_str("BOT_AVATAR", "https://jskitty.cat/vector/img/viktor.png"),
            bot_banner: env_str("BOT_BANNER", "https://jskitty.cat/vector/img/app-logo.jpg"),

            typing_indicator: env_bool("TYPING_INDICATOR", true),
            system_prompt: unescape(&env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)),

            // DM_MODE is canonical; PRIVATE_MODE is the legacy name.
            dm_mode: env_bool_either("DM_MODE", "PRIVATE_MODE", true),
            group_mode: env_bool("GROUP_MODE", true),

            invite_policy,
            invite_whitelist: env_list("INVITE_WHITELIST"),

            announce_channel_access: env_bool("ANNOUNCE_CHANNEL_ACCESS", false),

            public_mode: env_bool("PUBLIC_MODE", false),
            public_system_prompt: unescape(&env_str(
                "PUBLIC_SYSTEM_PROMPT",
                DEFAULT_PUBLIC_SYSTEM_PROMPT,
            )),
            public_dry_run: env_bool("PUBLIC_DRY_RUN", false),
            public_rate_limit_secs: env_parse("PUBLIC_RATE_LIMIT_SECS", 30_u64),
            public_whitelist: env_list("PUBLIC_WHITELIST"),
            public_keywords: env_list("PUBLIC_KEYWORDS")
                .into_iter()
                .map(|k| k.to_lowercase())
                .collect(),
        })
    }

    /// Whether an npub is in the public-mode whitelist (bypasses rate limit + blacklist).
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub fn is_public_whitelisted(&self, npub: &str) -> bool {
        self.public_whitelist.iter().any(|w| w == npub)
    }

    /// Whether public post content/hashtags match the keyword filter (empty filter = match all).
    #[cfg_attr(not(feature = "public"), allow(dead_code))]
    pub fn matches_keywords(&self, content: &str, hashtags: &[String]) -> bool {
        if self.public_keywords.is_empty() {
            return true;
        }
        let content_lower = content.to_lowercase();
        self.public_keywords.iter().any(|kw| {
            content_lower.contains(kw) || hashtags.iter().any(|t| t.to_lowercase().contains(kw))
        })
    }
}

/// Load `.env` from the working directory into the process environment.
///
/// Hand-rolled and lenient on purpose: `dotenvy` aborts the whole file at the first line it can't
/// parse, and an unquoted multi-word value (a prose `SYSTEM_PROMPT`, say) trips it — silently
/// dropping every variable after it, including the secret key. Here a bad line is skipped, not
/// fatal. Real environment variables always win, so this never clobbers an explicit override.
fn load_env_file() {
    let Ok(contents) = std::fs::read_to_string(".env") else {
        return;
    };
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || std::env::var_os(key).is_some() {
            continue;
        }
        // Strip one layer of matching surrounding quotes, if present.
        let value = value.trim();
        let value = match value.as_bytes() {
            [b'"', .., b'"'] | [b'\'', .., b'\''] => &value[1..value.len() - 1],
            _ => value,
        };
        std::env::set_var(key, value);
    }
}

// --- Environment helpers (trim, treat empty as unset) ---

fn env_opt(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_str(key: &str, default: &str) -> String {
    env_opt(key).unwrap_or_else(|| default.to_string())
}

fn env_bool(key: &str, default: bool) -> bool {
    match env_opt(key) {
        Some(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => default,
    }
}

/// Read a boolean from `key`, falling back to `legacy_key`, then `default`.
fn env_bool_either(key: &str, legacy_key: &str, default: bool) -> bool {
    if env_opt(key).is_some() {
        env_bool(key, default)
    } else if env_opt(legacy_key).is_some() {
        env_bool(legacy_key, default)
    } else {
        default
    }
}

fn env_parse<T: FromStr>(key: &str, default: T) -> T {
    env_opt(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_list(key: &str) -> Vec<String> {
    env_opt(key)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Turn literal `\n` / `\t` escape sequences into real characters, so multi-line prompts
/// survive a single-line `.env` entry.
fn unescape(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

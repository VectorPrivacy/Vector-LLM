//! Resolves npubs to human display names for multi-user (group + public) context.
//!
//! The Vector core already maintains a profile cache in its own state, so this is a thin,
//! in-process memo on top of it: check our map, else read the core's cached profile, else
//! fetch from relays once. No separate on-disk metadata cache to keep in sync.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use vector_sdk::{SlimProfile, VectorBot};

#[derive(Clone)]
pub struct NameResolver {
    bot: VectorBot,
    // npub -> resolved name (None = looked up, has no usable name).
    cache: Arc<Mutex<HashMap<String, Option<String>>>>,
}

impl NameResolver {
    pub fn new(bot: VectorBot) -> Self {
        Self { bot, cache: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// The best display name for an npub, or `None` if it has none. Cached after first lookup;
    /// hits the network at most once per npub (when the core has nothing cached).
    pub async fn name_of(&self, npub: &str) -> Option<String> {
        if let Some(hit) = self.cache.lock().unwrap().get(npub).cloned() {
            return hit;
        }

        // Prefer the core's already-cached profile; only fetch from relays on a miss.
        let name = match self.bot.cached_profile(npub).await {
            Some(p) => best_name(&p),
            None => self.bot.fetch_profile(npub).await.as_ref().and_then(best_name),
        };

        self.cache.lock().unwrap().insert(npub.to_string(), name.clone());
        name
    }

    /// A short, stable display label for an npub: its name if it has one, else a truncated npub.
    pub async fn label_of(&self, npub: &str) -> String {
        self.name_of(npub).await.unwrap_or_else(|| short_npub(npub))
    }
}

/// Pick the best name from a profile: nickname (local) > display name > name.
fn best_name(p: &SlimProfile) -> Option<String> {
    for candidate in [&p.nickname, &p.display_name, &p.name] {
        let trimmed = candidate.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// `npub1abcd…wxyz` — a compact, recognizable stand-in when a user has no name.
pub fn short_npub(npub: &str) -> String {
    if npub.len() > 18 {
        format!("{}…{}", &npub[..10], &npub[npub.len() - 4..])
    } else {
        npub.to_string()
    }
}

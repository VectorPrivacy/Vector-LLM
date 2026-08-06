//! Small shared helpers.

use std::time::{SystemTime, UNIX_EPOCH};

#[cfg_attr(not(feature = "public"), allow(dead_code))]
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Tidy an LLM reply for sending: normalize dashes to commas (no em/en dashes in user-facing
/// text) and trim surrounding whitespace.
pub fn sanitize_reply(text: &str) -> String {
    text.replace(['—', '–', '―'], ",").trim().to_string()
}

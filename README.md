# Vector LLM Bot

An LLM chatbot for [Vector](https://vectorapp.io), built on the
[Vector SDK](https://crates.io/crates/vector_sdk). It chats in **direct messages** and in
**Community group channels** through one unified handler (the SDK hides the difference, the way
discord.js does), keeps per-channel memory, understands multi-person conversations, and can
optionally monitor the public Nostr `kind:1` firehose.

Point it at any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, llama.cpp, Ollama, …).

## What it does

- **Direct messages** — replies to every DM, with a rolling per-conversation history.
- **Group chats** — joins Community channels and replies **only when @-mentioned or replied to**,
  so it isn't spammy. It understands who is talking via per-speaker labels and an npub roster
  (see [Multi-user context](#multi-user-context-group-chats)).
- **Per-channel memory** — each chat or channel has its own history file under `data/chats/`.
- **Commands** — `!clear` and `!model` (see [Commands](#commands)).
- **Public firehose mode** *(optional, off by default)* — watches public `kind:1` notes, judges
  relevance with the LLM, replies with NIP-10 threading, and auto-blacklists chronically
  irrelevant authors to save credits.
- **Self-managing identity** — with no key set, it mints and persists one on first run.
- **Resilient** — reconnects and catches up after a network drop (handled by the SDK).

## Architecture

All Nostr/encryption/transport work lives in the SDK; this crate is just the bot logic:

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Bootstrap + the unified `on_message` handler (DM/group gating, commands) |
| `src/config.rs` | Environment configuration |
| `src/llm.rs` | OpenAI-compatible client (uses the SDK's hardened HTTP client) |
| `src/memory.rs` | Per-channel memory + a race-safe `MemoryStore` |
| `src/prompt.rs` | Builds the LLM messages — speaker labels + roster for groups |
| `src/names.rs` | Resolves npubs to display names (over the core's profile cache) |
| `src/public.rs` | Public firehose mode (kind:1 monitoring, relevance, blacklist, threading) |
| `src/util.rs` | Small helpers |

## Quick start

```bash
cp .env.example .env      # then set LLM_API_KEY (and tweak anything else)
cargo run
```

On first run with no `VECTOR_NSEC`, the bot creates a stable identity at
`data/identity.nsec` and prints its npub. Message that npub from a Vector client and it replies.

## Configuration

Everything is read from the environment (a `.env` file is loaded at startup). Every value has a
default, so an empty `.env` runs a working DM + group bot.

### LLM

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_API_KEY` | *(unset)* | API key (optional for local servers) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |

### Context & storage

| Variable | Default | Meaning |
| --- | --- | --- |
| `HISTORY_LIMIT` | `16` | Max user/assistant turns kept per channel |
| `DATA_DIR` | `data` | Identity, database, and per-channel memory live here |

### Identity

| Variable | Default | Meaning |
| --- | --- | --- |
| `VECTOR_NSEC` | *(auto)* | Secret key: `nsec`, hex, or a mnemonic. Auto-created + persisted if unset |
| `PUBLISH_PROFILE` | `true` | Broadcast the bot's profile (kind:0) on startup |
| `BOT_NAME` / `BOT_ABOUT` / `BOT_AVATAR` / `BOT_BANNER` | *(Viktor)* | Profile fields |

> `VECTOR_SECRET_KEY` is still accepted as an alias for `VECTOR_NSEC`.

### Behaviour & modes

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYSTEM_PROMPT` | *(helpful assistant)* | Persona for DMs and groups (a roster is appended in groups) |
| `TYPING_INDICATOR` | `true` | Show a "typing…" indicator while generating |
| `DM_MODE` | `true` | Respond to direct messages |
| `GROUP_MODE` | `true` | Participate in group channels (mention/reply only) |

### Community invites

| Variable | Default | Meaning |
| --- | --- | --- |
| `INVITE_POLICY` | `public` | `public` (accept any) · `whitelist` · `manual` (accept none) |
| `INVITE_WHITELIST` | *(empty)* | Comma-separated npubs trusted to invite the bot (for `whitelist`) |

To be useful in group chats the bot must accept invites. `public` lets any community add it;
use `whitelist` to restrict who can.

### Public firehose mode

| Variable | Default | Meaning |
| --- | --- | --- |
| `PUBLIC_MODE` | `false` | Monitor public `kind:1` notes |
| `PUBLIC_SYSTEM_PROMPT` | *(participant)* | Persona + relevance prompt for public posts |
| `PUBLIC_DRY_RUN` | `true` | Generate replies but don't post them |
| `PUBLIC_RATE_LIMIT_SECS` | `30` | Min seconds between non-addressed replies (mentions bypass) |
| `PUBLIC_WHITELIST` | *(empty)* | npubs that bypass the rate limit + blacklist |
| `PUBLIC_KEYWORDS` | *(empty)* | If set, only posts containing a keyword are considered |

## Direct messages

The bot replies to every DM, keeping the last `HISTORY_LIMIT` turns as context. Storage is one
JSON file per peer, keyed by their npub, under `data/chats/`.

## Group chats

When invited to a Community, the bot joins and listens, but to avoid being noisy it **only replies
when addressed** — that is, when its npub is mentioned in the message, or the message is a reply to
one of its own messages. Otherwise it stays quiet and just keeps context.

Replies in groups are sent as threaded replies (so it's clear who the bot answered). Each channel
gets its own memory file, keyed by the channel id.

### Multi-user context

A group is a many-people conversation, so the bot needs to know *who said what* without burning
tokens on a 63-character npub before every line. So:

- Each stored user turn records its author.
- When building the prompt, every participant gets a short **label** — their display name if they
  have one, otherwise a truncated npub — and the system prompt gains a **roster** mapping each label
  to its full npub:

  ```
  This is a group chat with multiple participants. Every user message below is prefixed
  with the speaker's name and a colon. ... Participants:
  - JSKitty = npub1abc…
  - npub1xyz…qrs = npub1xyz…
  ```

- User messages are then prefixed cheaply with just the label (`JSKitty: hey bot, …`), and the
  bot's own turns are plain assistant messages. Names are resolved from Vector's profile cache and
  memoized, so each npub is looked up at most once.

## Commands

Work in DMs, and in groups when the bot is addressed:

- `!clear` — clear this channel's conversation history.
- `!model` — show the model used in this channel.
- `!model <name>` — set a per-channel model override.
- `!model reset` — revert to the default model.

## Public firehose mode

Separate from Vector's encrypted DMs/groups, this watches the open Nostr `kind:1` stream:

1. Gate each post by direct-mention / whitelist / blacklist / rate-limit / keyword filter.
2. Reconstruct NIP-10 thread context (root/reply markers, with a legacy positional fallback).
3. If we're addressed, reply; otherwise ask the LLM whether the post is relevant.
4. Generate a reply (thread context mapped to proper User/Assistant turns) and post it with NIP-10
   threading tags — or just log it under `PUBLIC_DRY_RUN`.

Authors who get `N` consecutive "not relevant" verdicts are blacklisted (persisted to
`data/blacklist.json`) to stop wasting LLM credits; a later relevant post clears the score.
Start with `PUBLIC_DRY_RUN=true` to watch what it *would* post before going live.

## Endpoint examples

**OpenAI**
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

**llama.cpp** (`./llama-server -m model.gguf --port 8080`)
```env
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=local-model
```

## License

MIT

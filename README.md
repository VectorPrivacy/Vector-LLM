# Vector LLM Bot

A minimal LLM chatbot for Vector built with:
- Vector SDK (Vector Bot client)
- dotenvy for environment config
- reqwest for OpenAI-compatible LLM calls
- Serde + std::fs for simple per-chat JSON memory stores

Core behavior:
- Per-chat memory: one JSON file per chat under `data/`, retaining the system prompt and up to N recent turns
- LLM calls use the OpenAI-compatible `/chat/completions` schema, so you can point it to OpenAI, llama.cpp servers, Groq, etc.
- Listens for gift-wrap (NIP-59) events, unwraps via Vector SDK, processes direct messages only, and replies privately


## Project layout

- `Cargo.toml` — crate metadata and dependencies
- `src/main.rs` — Vector bot bootstrap, subscription loop, unwrap and reply flow
- `src/config.rs` — loads configuration from environment variables
- `src/llm.rs` — OpenAI-compatible client
- `src/memory.rs` — simple per-chat JSON memory with trimming
- `.env.example` — template for configuration
- `data/` — per-chat memory files (gitignored)


## Prerequisites

- Rust (stable) and Cargo installed
- Network access to your LLM endpoint (OpenAI, local llama.cpp, etc.)


## Configure

Copy the example environment file and edit values:

```bash
cp .env.example .env
```

Environment variables (defaults shown):

- `LLM_BASE_URL=https://api.openai.com/v1`
  - Base URL for an OpenAI-compatible API. Examples:
    - OpenAI: `https://api.openai.com/v1`
    - llama.cpp server: `http://localhost:8080/v1`
    - Groq OpenAI compatible: `https://api.groq.com/openai/v1`
- `LLM_API_KEY=sk-...` (optional for local servers)
- `LLM_MODEL=gpt-4o-mini` (e.g., `llama-3.1-8b-instruct`)
- `LLM_TEMPERATURE=0.2`
- `HISTORY_LIMIT=16` (number of user/assistant messages to keep per chat)
- `DATA_DIR=data`
- `SYSTEM_PROMPT="You are a helpful assistant. Keep responses concise and factual."`
- `VECTOR_SECRET_KEY=` (optional; nsec bech32 or hex). If unset, ephemeral keys are generated
- `TYPING_INDICATOR=true` (send kind 30078 typing indicator before replying)


## Run

```bash
cargo run
```

On start, the bot will:
1) Load environment config
2) Create keys (generate if `VECTOR_SECRET_KEY` is not set)
3) Create a VectorBot with default metadata and relays
4) Subscribe to gift-wrap events and process direct messages
5) For each message:
   - Load per-chat memory (JSON) and append the user message
   - Trim memory to `HISTORY_LIMIT`
   - Call your configured LLM (`/chat/completions`)
   - Persist the assistant reply and send it back privately


## llama.cpp (OpenAI server) example

Run your llama.cpp server with OpenAI-compatible HTTP (ensure `/v1/chat/completions` is served):

```bash
# Example; your binary/flags may differ
./llama.cpp/server -m ./models/llama-3.1-8b-instruct.gguf --port 8080 --api
```

Use a `.env` like:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=llama-3.1-8b-instruct
LLM_TEMPERATURE=0.2
HISTORY_LIMIT=12
DATA_DIR=data
TYPING_INDICATOR=true
# No API key required for local server
```

Then:
```bash
cargo run
```


## OpenAI example

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
HISTORY_LIMIT=16
DATA_DIR=data
TYPING_INDICATOR=true
```

```bash
cargo run
```


## Memory format

For a chat ID (sender npub), the bot stores:
```
data/npub1abc...xyz.json
```

It contains:
- system prompt (string)
- history_limit (usize)
- messages: array of user/assistant messages

Oldest turns are trimmed when exceeding `HISTORY_LIMIT`. The system prompt is not counted against this limit.


## Implementation notes

- Unwrapping messages uses Vector SDK (no direct nostr-sdk usage required in your code).
- The bot filters only `Kind::PrivateDirectMessage` after unwrap and ignores other kinds.
- If `TYPING_INDICATOR=true`, it sends a brief typing indicator before LLM completion.


## Troubleshooting

- Build errors about missing environment variables usually mean your `.env` isn’t loaded; confirm it exists and values are present.
- If HTTP 401/403 from LLM, confirm `LLM_API_KEY` and `LLM_BASE_URL` are correct.
- For local servers, confirm they implement `/v1/chat/completions` with the standard OpenAI payload/response shape.


## License

MIT
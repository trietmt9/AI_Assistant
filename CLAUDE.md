# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A local-first, always-on, voice-driven personal assistant ("Friday") built on Ollama.
RAG over personal documents — with heavy math/physics/science content — plus tool use that
acts on the real world. Everything runs on hardware the user owns; nothing goes to a
hosted API.

**Status: planning. There is no source code yet** — only `.venv/` and the planning docs.
When scaffolding begins, follow the layout in `PLAN.md` §13.

## Documents

| File | Contents |
|---|---|
| `PLAN.md` | Master plan — hardware, models, stack, tools, phases. Read first. |
| `TRAINING_PLAN.md` | Fine-tuning, executed on the lab machine. Read before any training work. |

Keep these two current. Do not create additional top-level planning docs — fold new
decisions into the existing ones.

## Hardware constraints — check before proposing anything

**Laptop (dev + deployment target):** RTX 3060 Mobile, **~6 GB VRAM**, 16 GB RAM,
i7-11800H. Verified: a 7–8B model at Q4_K_M fits with ~98% GPU offload. **A 12B+ model does
not fit** and drops from ~30 tok/s to ~3 tok/s on CPU spill. Never propose a 14B/32B model
for laptop-local serving.

**Lab machine (training + model serving):** RTX Titan, 24 GB. Probably a Titan RTX =
**Turing (sm_75) = no bfloat16, no Flash Attention 2**. Verify with
`torch.cuda.get_device_capability()` before writing training configs. See `TRAINING_PLAN.md` §1.

**Topology:** lab machine runs LLM/VLM/embeddings; laptop runs voice I/O, tool execution
and the daemon.

## Environment

- Python **3.13.5**, managed with **`uv`** — not pip, not conda. Existing `.venv/` already
  has `torch`, `transformers`, `faster_whisper`, `ollama`, `bitsandbytes`.
- Ollama **0.13.0** runs as a systemd service on `:11434` (OpenAI-compatible at `/v1`).
- **No Docker on this machine.** Do not propose docker-compose solutions.
- Training uses a **separate venv** (`.venv-train`, Python 3.11) — serving and training
  dependency trees conflict.

## Design rules

These are decisions already made. Do not relitigate them without being asked.

1. **Facts live in RAG, never in weights.** Fine-tuning teaches behaviour — tool-call
   format, reasoning style, output conventions. Never propose fine-tuning to make the model
   "know" documents.
2. **≤ 8 tools in any single prompt.** An 8B model degrades sharply past that. Group tools
   into MCP servers by domain and route to 1–2 domains per turn.
3. **All computation goes through `calc` (SymPy).** Never let the model do arithmetic
   in-context. Its job is setting up the problem.
4. **Everything streams.** The voice loop has a <1 s budget from end-of-speech to first
   audio. Non-streaming code anywhere in that path is a bug, not an optimisation target.
5. **Embeddings run on CPU**, not through Ollama — sharing the GPU causes model eviction
   and a reload penalty per query.
6. **Equations never become standalone chunks**, and equation LaTeX is never stripped
   during ingestion. Chunk on section headings.
7. **The interrupt policy is rule-based, not model-decided.** The LLM does not choose when
   to speak unprompted.

## Code conventions

- Tools are `pydantic-ai` tools with Pydantic-validated arguments. Return validation errors
  to the model to retry rather than raising.
- Tool functions stay small and pure where possible; side effects go behind an explicit
  confirmation gate.
- Type hints throughout. Prefer `pathlib` over string paths.
- Log every agent turn — system prompt, retrieval results, tool calls, tool results, final
  answer — to SQLite. **These traces are the fine-tuning dataset**; treat the logging path
  as production code, not debug scaffolding.
- Tests for retrieval and tool argument handling before UI polish.

## Safety rules for generated code

- **Allowlist, never blocklist** for shell commands, file paths and home devices.
- Any irreversible action (file write, email send, calendar write, smart-home actuation,
  `run_shell`) requires explicit user confirmation showing the exact payload.
- Email and messaging tools **draft only** — never auto-send.
- `run_python` executes in a subprocess with a timeout, no network, and a scratch cwd.
- File tools operate on an explicit directory allowlist, never all of `/home/stephen`.
- Secrets in `.env`. Never in prompts, system messages, or committed files.
- `data/` is gitignored — index, SQLite DBs, traces and raw personal corpus never get
  committed.

## Working notes

- Phase order in `PLAN.md` §11 is deliberate. Phase 1 (retrieval + eval set) gates
  everything; do not skip ahead to UI or voice work because it's more visible.
- Build the eval set before the agent. Retrieval failures and generation failures need
  opposite fixes and must be measured separately.
- When benchmarking model changes, always compare against the current baseline on the
  golden question set — never ship a change on vibes.

# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A local-first, always-on, voice-driven personal assistant ("Eve") built on Ollama.
RAG over personal documents — with heavy math/physics/science content — plus tool use that
acts on the real world. Everything runs on hardware the user owns; nothing goes to a
hosted API.

**Status: planning. There is no source code yet** — only the planning docs. There is no
`.venv/` yet either. When scaffolding begins, follow the layout in `PLAN.md` §13.

## Documents

| File | Contents |
|---|---|
| `PLAN.md` | Master plan — hardware, models, stack, tools, phases. Read first. |
| `TRAINING_PLAN.md` | Fine-tuning, executed on this machine. Read before any training work. |

Keep these two current. Do not create additional top-level planning docs — fold new
decisions into the existing ones.

## Hardware constraints — check before proposing anything

**This machine (`cgu-ubuntu`) — dev, training and the reasoning backend.** Measured
2026-08-04:

| Resource | Value |
|---|---|
| GPU | **NVIDIA TITAN RTX, 24 GB**, driver 595.58.03 |
| Compute capability | **Turing, sm_75** — confirmed by product name, not assumed |
| CPU / RAM | i9-9900K (16 threads) / 62 GB |
| Disk | 207 GB free |
| OS | Ubuntu 24.04.4, X11 session |
| CUDA toolkit | 13.2 (`nvcc`) |

Turing means **no bfloat16, no Flash Attention 2, no `torch.compile`**. This is settled —
do not re-derive it, and do not copy training configs that assume bf16 or FA2. 4-bit NF4
via `bitsandbytes` does work. See `TRAINING_PLAN.md` §1.

**Jetson (always-on edge server) — board model not yet confirmed.** The Jetson is the
front door: it holds the API endpoint the phone and laptop talk to, plus voice I/O, tool
execution and the proactive daemon. It does **not** serve the primary model. Before proposing
anything that runs on it, confirm which board it is — `PLAN.md` §2 has the branch table
and the verification command. A 4 GB Jetson Nano and a 16 GB Orin NX are not the same
machine.

**Do not propose serving the primary model from any Jetson.** It is 17–20 GB at Q4_K_M;
token generation is memory-bandwidth-bound and prefill on Orin-class silicon puts
time-to-first-token at 10–30 s for a RAG-sized prompt. That fails the voice budget by an
order of magnitude.

**Topology:** TITAN RTX box serves the primary model (`PLAN.md` §3 —
`qwen3.6:27b-mtp-q4_K_M`, settled by the phase-0 bake-off on 2026-08-04) and runs training. Jetson runs voice I/O,
tools, the daemon, and the small fallback model. Phone and laptop are thin clients that
talk to the Jetson.

**Two different models, deliberately.** What gets *served* (27–35B) and what gets
*fine-tuned* (`Qwen3.5-9B`) are not the same checkpoint — a 27B QLoRA does not fit 24 GB at
the 4096 sequence length the training data needs. See `TRAINING_PLAN.md` §2 before
proposing otherwise.

**Model choice is researched, not remembered.** The plan was updated 2026-08-04 against
current benchmarks; the 14B/Mistral-Small tier it originally named is superseded. Re-check
`ollama.com/library` and BFCL before proposing a model — this landscape moves faster than
any assistant's training data.

## Environment

- Python **3.12.3** (system), managed with **`uv` 0.9.21** — not pip, not conda.
- **Nothing is installed yet.** No `.venv`, no `torch`, no `transformers`, no
  `faster_whisper`. Ollama is **not installed** and its systemd unit is inactive. Treat
  `PLAN.md` phase 0 as genuinely unstarted.
- **No Docker on this machine.** Do not propose docker-compose solutions.
- Training uses a **separate venv** (`.venv-train`) from serving — the dependency trees
  conflict. Pin the torch/CUDA variant against the installed driver, not against the
  examples in the docs; CUDA 13.2 is much newer than most published Unsloth extras.

## Design rules

These are decisions already made. Do not relitigate them without being asked.

1. **Facts live in RAG, never in weights.** Fine-tuning teaches behaviour — tool-call
   format, reasoning style, output conventions. Never propose fine-tuning to make the model
   "know" documents.
2. **≤ 8 tools in any single prompt.** Small models degrade sharply past that. Group tools
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
8. **Respect the network seam.** Model access goes through the `llm/` client interface —
   nothing outside it imports an Ollama client or assumes the model is local. Likewise the
   voice layer does not import retrieval or tools directly. Phases 0–4 all run on one
   machine, but phase 5 splits them across the workstation and the Jetson, and that must
   be a deployment change rather than a rewrite. See `PLAN.md` §11 and §13.

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
- File tools operate on an explicit directory allowlist, never all of `/home/cgu`.
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

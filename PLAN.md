# Personal AI Assistant — Master Plan

A local-first, always-on, voice-driven assistant in the spirit of Jarvis / Friday.
No data leaves the machines you own.

**Requirements (locked in):**

| | |
|---|---|
| **Data** | Local files & documents · calendar & tasks · code & projects · math/physics/science material |
| **Capability** | Retrieval *and* actions from day one |
| **Interface** | Terminal CLI first, voice as the primary interface from phase 3 |
| **Character** | Always on, proactive, speaks first, acts on the world |

Companion document: `TRAINING_PLAN.md` (fine-tuning, to be executed on the lab machine).

---

## 1. What actually makes it Jarvis

A blunt observation before any tool list: **the tool list is not what separates Jarvis
from a chatbot.** A chatbot with forty tools is still a chatbot. Four properties do the work:

| Property | Meaning | Section |
|---|---|---|
| **Voice, low latency** | Speak, answer in under ~1 s, interruptible | §5 |
| **Always on** | A daemon, not a command you run — no session boundary | §8 |
| **Proactive** | *It* speaks first: warnings, reminders, briefings | §8 |
| **Actuation** | It changes the world, not just describes it | §7 |

Of the four, **proactivity creates the illusion of a mind** — and it is mostly plumbing
(event watchers plus an interrupt policy), not model capability. Adding thirty tools to a
text CLI does not get you closer to Friday. The streaming voice loop and the proactive
daemon do.

---

## 2. Hardware and topology

### Laptop — measured 2026-08-04

| Resource | Value |
|---|---|
| GPU | RTX 3060 Mobile, ~6 GB VRAM — driver 610.43.02, CUDA verified working |
| Verified | `mistral:7b` Q4_K_M @ 4096 ctx = 5.2 GB, **98% GPU offload** |
| CPU / RAM | i7-11800H (16 threads) / 16 GB |
| Disk | 176 GB free |
| Installed | Ollama 0.13.0 (systemd), Python 3.13.5 + `uv`, venv with `torch`, `transformers`, `faster_whisper`, `ollama` |
| Missing | Docker, vector store, agent framework |

### Lab machine — RTX Titan, 24 GB

Used for fine-tuning and, in the final topology, for serving the large models.
**Verify its compute capability before writing any training config** — see
`TRAINING_PLAN.md` §1. If it is a Titan RTX (Turing, sm_75) there is no bfloat16 and no
Flash Attention 2, which breaks most copy-pasted configs.

### The split

An always-on assistant does not fit in 6 GB:

```
LLM 8B Q4      5.0 GB  ─┐
VLM (screen)   3.5 GB   ├─  ~9 GB wanted, 6 GB available
embeddings     0.6 GB  ─┘
```

| Runs where | What |
|---|---|
| **Lab machine (24 GB)** | LLM, VLM, embeddings — served over the network |
| **Laptop** | Wake word, VAD, STT, TTS, tool execution, proactive daemon |

The laptop is the ears, mouth and hands; the lab box is the brain. This is both the
correct engineering answer and, conveniently, the actual Iron Man architecture — Jarvis
lives in the basement, not in the suit.

**Offline fallback:** a local 4B model on the laptop so voice control of lights, media and
timers keeps working when the lab machine is unreachable.

---

## 3. Models

| Job | Model | Where | Notes |
|---|---|---|---|
| Reasoning, math, tool calling | `qwen3:8b` (14B if lab-served) | Lab GPU | Strong 8B for math/science, native tool calling. Replaces `mistral:7b`, a weak tool caller. |
| Embeddings | `bge-m3` | **CPU** | Multilingual, 8k context, good on technical text |
| Vision (screen understanding) | `qwen3-vl` or `moondream2` | Lab GPU | Phase 7 |
| Speech-to-text | `faster-whisper` small/medium int8 | Laptop CPU | **Already installed** |
| Text-to-speech | `Piper` (fastest) or `Kokoro-82M` (better voice) | Laptop CPU | |
| Wake word | `openWakeWord` (ONNX) | Laptop CPU | ~1% of one core |

**Never serve embeddings through the same Ollama instance as the LLM.** It evicts and
reloads models from VRAM on every alternation and you pay a reload penalty per query. Run
`bge-m3` via `sentence-transformers` on CPU — a query embedding is one short string
(~20 ms), and bulk ingestion is a batch job.

Add early, cheap, high value: `bge-reranker-v2-m3` on CPU — large retrieval quality gain,
zero VRAM cost.

### Ollama tuning, apply before anything else

```
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_FLASH_ATTENTION=1      # less KV-cache memory
OLLAMA_KV_CACHE_TYPE=q8_0     # ~half the KV cache, negligible quality loss
OLLAMA_KEEP_ALIVE=30m         # stop reloading the model every request
OLLAMA_NUM_PARALLEL=1         # single user; don't split VRAM
```

Default context is 4096 — too small for a system prompt plus retrieved chunks plus a tool
trace. Set `num_ctx: 8192` per request or bake it into a Modelfile.

---

## 4. Stack

No Docker, no web server. One Python package plus a CLI entry point and a daemon.

| Layer | Pick | Why |
|---|---|---|
| Inference | Ollama | Already running; OpenAI-compatible endpoint at `:11434/v1` |
| Agent loop | `pydantic-ai` | Typed tools, validation, retries, streaming, native MCP support |
| Vector store | **LanceDB** | Embedded, no server, disk-backed, built-in full-text search |
| Retrieval | Hybrid vector + BM25, then rerank | Keyword matters here — theorem names, symbols, project codes |
| Doc parsing | `docling` (scientific PDF) · `markitdown` (office/plain) | See §6 |
| CLI | `typer` + `rich` + `prompt_toolkit` | Streaming output, markdown/code rendering, real REPL |
| Storage | SQLite | Memory, metadata, chat history, trace logs — one file, easy to back up |
| Tool packaging | MCP servers, grouped by domain | See §9 |

---

## 5. Voice loop — the interface

Not tools the model calls; the loop it lives inside. Everything local.

| Stage | Component | Notes |
|---|---|---|
| Wake word | `openWakeWord` / Porcupine | Custom phrase — "Friday" |
| VAD / endpointing | `silero-vad` | Decides when you stopped talking. Tuning this matters more than STT accuracy. |
| STT | `faster-whisper` small int8 | CPU, ~200 ms |
| TTS | `Piper` or `Kokoro-82M` | CPU, streaming |
| Barge-in | Duck/kill TTS on detected speech | Non-negotiable — not being able to interrupt feels broken |

**Latency budget — the whole game.** Target under 1000 ms from end-of-speech to first audio:

```
VAD endpoint 300ms → STT 200ms → LLM TTFT 300ms → TTS first chunk 150ms  ≈ 950ms
```

This only works if **every stage streams**. Sentence-chunk LLM output into TTS so speech
begins on the first clause rather than the completed response. Non-streaming, the identical
pipeline takes 4–5 s and feels dead.

---

## 6. Knowledge layer

### The math/science corpus is the hard part

Generic RAG pipelines destroy scientific documents. Budget real time here.

- **Parsing.** `markitdown` and plain PyMuPDF mangle equations into garbage
  (`Z 1 0 f (x)dx`). Use **`docling`** — formula and table understanding, emits LaTeX.
  It is slow and GPU-hungry; run it as an offline batch job, not on demand.
- **Keep equations inline as LaTeX** (`$...$`). Do not strip them — the model reads LaTeX
  fine and the surrounding prose is what makes them retrievable.
- **Never let an equation become its own chunk.** A bare formula has almost no embedding
  signal. Chunk on section headings so each equation travels with its explanation.
- **Metadata per chunk**: source path, title, section heading, page, created/modified date,
  doc type (`paper` / `textbook` / `notes` / `code`). Many real questions are scoped by
  source or time, and metadata filters answer those far better than embeddings do.
- **Code chunks differently** — split on function/class boundaries, keep file path and
  imports in the chunk header.
- **Incremental re-index** keyed on path + mtime + content hash.

```
source files → parse to markdown → chunk (512–1024 tok, ~100 overlap, respect headings)
             → embed (bge-m3, CPU) → LanceDB + FTS index
```

### Terminal rendering

LaTeX does not render in a terminal. Use SymPy `pretty_print` for unicode math, `rich` for
markdown and code, and write plots to PNG in a scratch dir (inline images work in Kitty or
any sixel-capable terminal).

---

## 7. Tools

### 7.1 Perception
| Tool | Purpose |
|---|---|
| `see_screen()` | Screenshot → VLM. "What's this error?" without describing it. |
| `system_state()` | Battery, CPU/GPU load, temps, network, focused window |
| `whats_playing()` | MPRIS media state |
| `presence()` | Camera or BT proximity — gates proactive speech |
| `context_now()` | Time, location, weather, current calendar block |

`context_now()` is cheap and disproportionately valuable — inject it into every system
prompt. "It's Tuesday 23:40 and you have a 9am" improves answers for free.

### 7.2 Knowledge
`search_docs(query, source?, date_range?)` · `search_web(query)` ·
`remember(fact)` / `recall(query)` · `read_file(path)` / `list_files(dir)`

### 7.3 Computation
| Tool | Notes |
|---|---|
| `calc(expression)` | **SymPy** — solve, integrate, differentiate, simplify; `pint` for units |
| `run_python(code)` | Sandboxed: subprocess, timeout, no network, scratch cwd. NumPy/SciPy/Matplotlib. |

**`calc` is not optional.** An 8B model cannot do reliable arithmetic, let alone symbolic
manipulation. Route every computation through SymPy and the model's job reduces to setting
up the problem — which it is genuinely good at.

### 7.4 Actuation — digital
| Tool | Notes |
|---|---|
| `run_shell(cmd)` | **Allowlisted commands only.** Highest-risk tool here. |
| `control_desktop()` | Launch apps, focus/move windows — `xdotool` (X11) / `ydotool` (Wayland) |
| `browse(url, task)` | Playwright — read pages, fill forms |
| `media_control(action)` | `playerctl` |
| `notify(msg)` | Desktop notification + optional spoken announcement |

### 7.5 Actuation — physical
| Tool | Notes |
|---|---|
| `home_*()` | **Home Assistant** REST/WebSocket API — lights, climate, locks, sensors, cameras |

For the Iron Man feeling specifically, Home Assistant is the single highest-impact
integration in this document. Everything else makes a better assistant; this makes it feel
like it lives in the room.

### 7.6 Personal ops & comms
`calendar_list(range)` / `calendar_create(...)` · `email_search()` / `email_draft()`
(draft only, never auto-send) · `send_message()` (Telegram/Signal bot — doubles as a remote
interface when away from the desk)

### 7.7 Lab operations
| Tool | Notes |
|---|---|
| `training_status()` | SSH to the lab box — GPU util, VRAM, step/loss, ETA |
| `watch_job(id)` | Proactively announce completion, OOM, NaN loss, plateau |

"Friday, how's the run?" — and more importantly, *it* tells you at 3am that the loss went
NaN. Given `TRAINING_PLAN.md`, this is daily value, not a demo.

---

## 8. The proactive engine

Not a tool. A daemon running beside the agent that can initiate conversation.

**Triggers:** cron/schedule · file watchers (`watchdog`) · new email · calendar approach ·
Home Assistant sensor thresholds · training-job events · system events (battery, disk)

**Behaviours:**
- **Morning brief** — calendar, weather, overnight email, training results
- **Pre-meeting nudge** — 10 minutes ahead, with relevant docs already retrieved
- **Threshold alerts** — "the run OOM'd", "disk at 95%"
- **End-of-day summary** — what you worked on, what's still open

**The interrupt policy is the hard part.** An assistant that speaks at the wrong moment
gets muted permanently. Gate every proactive utterance on: is the user present, are they in
a meeting or focused, does urgency exceed the interruption cost, has it spoken in the last
N minutes. Build this as an explicit, tunable rule layer — **do not let the LLM decide when
to interrupt.**

---

## 9. Tool routing — ~30 tools against an 8B model

An 8B model degrades badly past ~6–8 tools in a single prompt. Three layers:

1. **Domain grouping.** Package tools as **MCP servers** by domain — `knowledge`, `compute`,
   `desktop`, `home`, `comms`, `lab`. Clean boundaries, independently testable,
   `pydantic-ai` speaks MCP natively.
2. **Intent router.** A fast first pass (small model or plain classifier) picks 1–2 domains
   from the utterance; only those tools enter the prompt. The model never sees more than ~8.
3. **Fine-tune for routing.** Exactly what `TRAINING_PLAN.md` targets — logged traces of
   correct routing decisions are the highest-value training data available.

---

## 10. Evaluation — build in phase 1, not at the end

This is where local-RAG projects quietly fail. Write **20–30 real questions** about your own
data with known-correct answers *before* building the agent. Track separately:

| Metric | Catches |
|---|---|
| Retrieval hit rate @5 | Was the right chunk even fetched? |
| Answer correctness (manual 1–5) | End-to-end quality |
| Tool-call validity | Well-formed arguments? |
| Correct tool selection | Including correctly calling *nothing* |
| Latency / tok/s | Whether the voice loop is viable |

Retrieval failures and generation failures need opposite fixes — never collapse them into
one number.

---

## 11. Build phases

| Phase | Deliverable | Why here |
|---|---|---|
| 0 | Pull `qwen3:8b` + `bge-m3`, apply Ollama tuning, verify a tool call round-trips | Foundation |
| 1 | Ingestion (docling → chunks → LanceDB) + hybrid retrieval + **eval set** | Everything depends on retrieval quality |
| 2 | `pydantic-ai` agent loop + `calc` + `run_python`, terminal CLI, **trace logging** | Prove tool calling before adding voice |
| 3 | **Voice loop** — wake word → STT → agent → TTS, fully streaming | Biggest perceived jump. Do before adding tools. |
| 4 | **Daemon + proactive triggers + interrupt policy** | Second biggest. Now it feels alive. |
| 5 | Actuation: Home Assistant, desktop, media, comms | The Iron Man layer |
| 6 | MCP grouping + intent router | Needed once tool count passes ~8 |
| 7 | Lab-job monitoring, perception/VLM, calendar & email | |
| 8 | Fine-tune on logged traces (`TRAINING_PLAN.md`) | Needs phases 2–6 running to generate data |

**Phase 1 decides whether any of it is useful** — everything downstream is only as good as
what retrieval hands the model. **Phases 3 and 4 are where "chatbot" becomes "Friday."**

---

## 12. Safety — it can act on the world

- **Confirm before any irreversible action** — writes, sends, purchases, locks, `run_shell`.
  Speak the exact payload back before executing.
- **Allowlist, never blocklist** — shell commands, file paths, home devices.
- **No auto-send** on email or messages. Draft, then confirm.
- **Rate-limit proactive speech** with a hard cap per hour, independent of the policy layer.
- **Physical-world actions get the strictest gate.** Locks and heating are not where you
  want to discover an 8B model misparsed an argument.
- **A global mute/kill word**, handled outside the model, that stops everything instantly.
- Fully local by default. Only optional web search, calendar API and messaging leave the
  machine.
- `.gitignore` the index, SQLite files and raw corpus. Secrets in `.env`, never in prompts.

---

## 13. Repository layout

```
AI_assistant/
├── assistant/
│   ├── ingest/       # parsers, chunkers, incremental indexer
│   ├── retrieval/    # hybrid search, reranker
│   ├── tools/        # knowledge, compute, desktop, home, comms, lab
│   ├── voice/        # wake word, VAD, STT, TTS, streaming loop
│   ├── daemon/       # triggers, interrupt policy, briefings
│   ├── agent.py      # pydantic-ai agent, routing, mode definitions
│   └── cli.py        # typer + prompt_toolkit REPL
├── eval/             # golden question set + scoring script
├── data/             # gitignored: LanceDB index, sqlite, traces, scratch
├── CLAUDE.md
├── PLAN.md
└── TRAINING_PLAN.md
```

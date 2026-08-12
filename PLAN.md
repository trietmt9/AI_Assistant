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

Companion document: `TRAINING_PLAN.md` (fine-tuning, executed on the workstation).

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
text CLI does not get you closer to Eve. The streaming voice loop and the proactive
daemon do.

---

## 2. Hardware and topology

### Workstation `cgu-ubuntu` — measured 2026-08-04

The brain. Development, fine-tuning, and serving the reasoning model.

| Resource | Value |
|---|---|
| GPU | **NVIDIA TITAN RTX, 24 GB** — driver 595.58.03 |
| Compute capability | **Turing, sm_75** — no bf16, no Flash Attention 2, no `torch.compile` |
| CPU / RAM | i9-9900K (16 threads) / 62 GB |
| Disk | 207 GB free of 915 GB |
| OS | Ubuntu 24.04.4, X11 (so `xdotool`, not `ydotool`, for §7.4) |
| CUDA toolkit | 13.2 |
| Audio | ALC1220 capture + playback present — a voice loop can be developed here directly |
| Installed | Python 3.12.3, `uv` 0.9.21 |
| **Missing** | **Ollama, all Python deps, vector store, agent framework** — phase 0 is unstarted |

The 24 GB card removes the constraint that shaped the original plan. Everything fits at
once:

**Usable VRAM is 22.4 GiB, not 23.5** — X11 holds ~0.8 GB. Budget against 22.4.

```
Qwen3.6-27B Q4_K_M   16.2 GiB  ─┐
KV cache @8k (q8_0)   ~1.0 GiB  ├─  ~17 GiB of 22.4 GiB usable
                               ─┘   (vision is built in — no separate VLM)
```

**The 35B-A3B MoE was ruled out on 2026-08-04, measured not estimated.** Its Q4_K_M layer
is 23,938,321,664 bytes = **22.3 GiB of weights against 22.4 GiB available**, leaving
nothing for a KV cache. It would spill to CPU — the exact failure this plan exists to avoid.
The "~17–20 GB" figure previously recorded here came from a secondary source and was wrong;
the registry manifest is authoritative:

```bash
curl -s https://registry.ollama.ai/v2/library/<model>/manifests/<tag> | \
  python3 -c "import sys,json;print(json.load(sys.stdin)['layers'][0]['size']/2**30)"
```

**Check this before pulling any model.** A 24 GB card does not hold a 24 GB model.

Embeddings still run on CPU (§3) — that rule is about avoiding model eviction, not about
capacity, and it survives the hardware change.

### Topology decided 2026-08-12: the workstation is the server. No Jetson.

Superseding the Jetson-fronted design below. The workstation serves everything — model,
retrieval, agent, **and speech** — and the laptop and phone are audio clients that connect
to it directly.

```
   laptop  `eve` CLI  ─┐                workstation
   (terminal, native)  ├─ Tailscale ─→   ├─ STT   faster-whisper  (GPU)
                       │                 ├─ agent + retrieval
   phone  app / PWA   ─┘                 ├─ TTS   Piper           (CPU)
                                         └─ Ollama 27B      (127.0.0.1 only)
```

**Speech runs on the server, not the clients.** A phone cannot run Whisper at useful speed,
the workstation has a GPU that is otherwise idle between turns, and one implementation then
serves every client. Clients capture microphone audio and play audio back; that is all.

**The laptop client is the `eve` CLI itself, installed globally** (`uv tool install`), the
way `claude` is. Same command, two modes:

| Mode | Where | What it does |
|---|---|---|
| local | workstation | Builds the agent in-process, as phases 1–2 already do |
| **remote** | laptop | Talks to the workstation's endpoint; no model, no index, no GPU |

This matters for packaging: **the remote client must not pull `torch`, `lancedb` or
`docling`.** Those are ~3 GB and exist only to serve. A `client` extra keeps the laptop
install to `httpx`, `websockets`, `typer`, `rich` and `sounddevice` — seconds, not gigabytes.

Being a native process rather than a browser, the laptop client also gets what §5 says a
browser cannot: direct `sounddevice` access, and therefore a real always-on wake word.

**This machine has a public IP — `120.126.10.44`, no NAT, gateway in the same /24.** That is
a globally routable TANet address, not a private LAN. So the §12 rule "LAN or Tailscale
only" has no LAN to fall back on: **binding the endpoint to `0.0.0.0` publishes it to the
internet.** Two things follow, and neither is optional:

- **Use Tailscale.** Bind the API to the Tailscale interface only, never `0.0.0.0`. It also
  issues real HTTPS certificates, which the browser clients *require* — `getUserMedia` only
  works in a secure context, so without HTTPS the phone will not grant microphone access at
  all. And it works off-campus over cellular, which a LAN-only design would not.
- **Ollama stays on `127.0.0.1`.** It has no authentication of its own. The API server is
  the only thing that should ever be reachable, and it holds the bearer token.

The Jetson notes below are retained because they still describe the *fallback-model* and
always-on tradeoffs accurately, and because a Jetson may still be worth adding later purely
so the front door survives the workstation sleeping. It is no longer on the critical path.

### Jetson — the always-on edge server

**Status: deferred, and as of 2026-08-12 no longer required** (see above). Build and deploy
everything on the workstation first (phases 0–4).
The Jetson, and portability generally, are revisited when there is a working assistant to
move — the durable state is ~1 GB of index, memory, traces and adapter, so relocating is a
copy, not a rebuild. Do not design for it in advance beyond keeping the §11 seam intact.

The workstation is a desktop, not an appliance. The Jetson is what stays up: it holds the
API endpoint that the **phone and laptop connect to**, and it runs the parts of the system
that must be always-on and physically present — voice I/O, tool execution, the proactive
daemon. It proxies reasoning to the workstation.

**Identify the board before designing anything for it.** "Jetson Nano" is as ambiguous as
"RTX Titan" was, and the answer changes what is possible by an order of magnitude:

```bash
# on the Jetson
cat /etc/nv_tegra_release
cat /proc/device-tree/model
free -h && nvidia-smi 2>/dev/null || tegrastats --interval 1000
```

| Board | Unified RAM | Bandwidth | Arch | Largest sane model |
|---|---|---|---|---|
| Jetson Nano (2019) | 4 GB | 25.6 GB/s | Maxwell sm_53 | ~1.5–3B Q4, slowly. EOL: JetPack 4.6, CUDA 10.2, Python 3.6 |
| Orin Nano 8 GB (Super) | 8 GB | 102 GB/s | Ampere sm_87 | **3–4B Q4 comfortably**, 8B Q4 tight |
| Orin NX 16 GB | 16 GB | 102 GB/s | Ampere sm_87 | **`qwen3.5:9b` comfortably** — the §3 fallback |
| AGX Orin 32/64 GB | 32–64 GB | 204 GB/s | Ampere sm_87 | 27B Q4 viable, slowly |

RAM is **unified** — shared with the OS and the display, so usable model budget is roughly
total minus 2 GB.

### Why the primary model does not live on the Jetson

Two independent limits, either one fatal:

- **Bandwidth.** Generation is memory-bandwidth-bound. Even a 9 GB model at 102 GB/s has a
  hard ceiling near 11 tok/s and realistically lands at 6–8 — barely above the ~4 tok/s
  that continuous speech consumes, with nothing left for a KV cache that grows. The §3
  primary is 17–20 GB and does not fit at all.
- **Prefill, which is worse.** Prefill is compute-bound, and a RAG turn is 1500–3000
  tokens of system prompt plus retrieved chunks. On Orin-class silicon that is **10–30 s
  to first token** against the 950 ms budget in §5. The voice loop would not work.

So: **the primary model is served from the workstation over the network**, and the Jetson
runs only the small fallback. This is the same architecture as before with the Jetson in
the laptop's old role — ears, mouth and hands at the edge, brain in the basement.

**Offline fallback:** a 3–4B model on the Jetson itself, sized to the board above, so voice
control of lights, media and timers keeps working when the workstation is asleep or
unreachable. It is a degraded mode, not the main path — it does not get RAG or the full
tool set.

### Client access

Phone and laptop are **thin clients**. They talk to the Jetson's endpoint, never directly
to the workstation and never to a model. Keep the transport boring — HTTP + SSE for
streaming, bearer token, LAN or Tailscale only, no port forwarding to the open internet.

---

## 3. Models

Researched 2026-08-04. The 14B tier this plan originally targeted is superseded; so is
Mistral Small 3.2. The 24 GB tier is now 27–35B, and all current Qwen3.5/3.6 models are
Apache 2.0, 262K context, and **vision + tools + thinking** in one checkpoint.

| Job | Model | Where | Notes |
|---|---|---|---|
| **Primary — reasoning, math, tools, vision** | **`qwen3.6:27b-mtp-q4_K_M`** (15.7 GiB) | Workstation GPU | **Chosen by the phase-0 bake-off, 2026-08-04.** Dense 27B with multi-token prediction. **MTP works on Turing** — 1.58× measured, see below. |
| Runner-up, kept pulled | `qwen3.5:27b` Q4_K_M (16.2 GiB) | Workstation GPU | Tied on tool calling, 37% slower. Keep as the A/B control for phase-1 retrieval work — it is stronger on science and instruction-following per independent tests. |
| Control, no longer needed | `qwen3.6:27b` Q4_K_M (16.2 GiB) | — | Would have isolated whether MTP works on sm_75. It does; do not bother pulling this. |
| Offline fallback **+ fine-tune target** | `qwen3.5:9b` | Jetson / training | BFCL-V4 0.661 — only 0.024 behind the 27B. The one size that trains at 4096 seq on 24 GB. |
| Embeddings | `bge-m3` | **CPU** | Multilingual, 8k context, good on technical text |
| Reranker | `bge-reranker-v2-m3` | **CPU** | Large retrieval gain, zero VRAM |
| Vision (screen understanding) | *(none — built into the primary)* | — | Qwen3.5/3.6 are natively multimodal; §7.1 `see_screen()` uses the primary model |
| Speech-to-text | `faster-whisper` small/medium int8 | Jetson | Dev on the workstation first — it has a mic |
| Text-to-speech | `Piper` (fastest) or `Kokoro-82M` (better voice) | Jetson | |
| Wake word | `openWakeWord` (ONNX) | Jetson | ~1% of one core |

### Measured on this card, 2026-08-04 — the bake-off result

`eval/bakeoff.py`, thinking off, 2.5k-token RAG-shaped prompt, `num_ctx` 8192:

| | TTFT | tok/s | peak VRAM | cold load | tools ok | over-calls | malformed |
|---|---|---|---|---|---|---|---|
| **`qwen3.6:27b-mtp-q4_K_M`** | 4.29 s | **44.8** | 18.2 GB | 56 s | 21/22 | 0 | 0 |
| `qwen3.5:27b` | 4.05 s | 28.3 | 17.4 GB | 19 s | 21/22 | 0 | 0 |

Three findings, none of them available from any published benchmark:

- **Generation speed is a solved problem, and MTP is why.** 44.8 tok/s is 1.58× the plain
  model and more than 3× the 13.7 t/s a 3090 was expected to manage. The worry recorded
  above — that a dense 27B would land at 9–10 t/s and leave no margin over speech — was
  wrong. **MTP works on sm_75**, which is not documented publicly anywhere.
- **Tool calling needs no work.** 21/22 for both, zero malformed arguments, zero
  over-calls. Each missed a different `calendar_create` scenario, so it is noise rather
  than a pattern. This trips the gate in `TRAINING_PLAN.md` §0.
- **TTFT is the failure, and it is not the model's fault.** See §5.

Both cleared VRAM comfortably, so the 22.4 GiB ceiling is not binding at 27B — there is
room for the KV cache to grow through a long conversation.

**Vendor numbers and independent tests disagree — prefer the independent ones.** Alibaba
reports Qwen3.6-27B at 87.8% GPQA-Diamond; independent testing found it *underperforms*
Qwen3.5 on GPQA-Diamond and is "significantly worse" on IFBench instruction-following.
Qwen3.6 clearly wins on AIME math. So Qwen3.6 for math, Qwen3.5 for science and following
directions — hence the bake-off rather than a pick.

**Every throughput figure above is Ampere/Ada. None of these models has been benchmarked on
Turing.** Treat them as upper bounds and measure on this card in phase 0.

**Never serve embeddings through the same Ollama instance as the LLM.** It evicts and
reloads models from VRAM on every alternation and you pay a reload penalty per query. Run
`bge-m3` via `sentence-transformers` on CPU — a query embedding is one short string
(~20 ms), and bulk ingestion is a batch job.

Add early, cheap, high value: `bge-reranker-v2-m3` on CPU — large retrieval quality gain,
zero VRAM cost.

**"Cheap" was wrong, measured 2026-08-04.** Zero VRAM is true. Zero cost is not:
bge-reranker-v2-m3 is a 560M cross-encoder that scores every candidate against the query,
and on this i9-9900K that is **8.1 s at 10 candidates, 14.7 s at 20, 26.5 s at 30** — against
**0.08 s for hybrid search with no reranking at all**. Two consequences:

- LanceDB's `CrossEncoderReranker` never sets `max_length`, so it defaults to the model's
  full 8192-token window. That alone was 40–95 s per query until `retrieval/search.py`
  subclassed it to bound the length at 512. Keep that override if you ever swap rerankers.
- **Whether reranking earns its place here is now an empirical question, not an assumption.**
  In spot checks plain hybrid already put the correct document at rank 1. `eval/retrieval_eval.py
  --compare` exists to settle it on the golden set. If hit@5 is within a point or two, turn
  reranking off — a 100× latency saving is decisive for the §5 voice path.
- If it does prove necessary, `EVELYN_RERANK_DEVICE=cuda` puts it in ~1.1 GB alongside the
  18 GB served model. That trades against design rule 5, so measure before adopting it.

### Ollama tuning, apply before anything else

Ollama is **not installed yet** — this is the first task of phase 0. Once it is:

```
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_FLASH_ATTENTION=1      # less KV-cache memory
OLLAMA_KV_CACHE_TYPE=q8_0     # ~half the KV cache, negligible quality loss
OLLAMA_KEEP_ALIVE=30m         # stop reloading the model every request
OLLAMA_NUM_PARALLEL=1         # single user; don't split VRAM
OLLAMA_HOST=0.0.0.0:11434     # the Jetson must reach it — bind beyond localhost
```

`OLLAMA_FLASH_ATTENTION=1` is an Ollama/llama.cpp kernel flag, not Flash Attention 2 — it
works on Turing. The sm_75 restriction in `TRAINING_PLAN.md` §1 applies to the *training*
stack, not to Ollama serving.

`OLLAMA_HOST` opens the port to the LAN. Firewall it to the Jetson's address; do not expose
it to the internet — Ollama has no authentication of its own.

Default context is 4096 — too small for a system prompt plus retrieved chunks plus a tool
trace. Set `num_ctx: 8192` per request or bake it into a Modelfile.

**Correction, measured 2026-08-12: "per request" does not work on the `/v1` route.** Setting
`num_ctx` to 2048 via `extra_body={"options": {...}}` through `pydantic-ai` changed nothing —
`ollama ps` still reported 32768. Ollama's OpenAI-compatible shim silently drops the
`options` block. Only two things actually set it: a Modelfile `PARAMETER`, which applies
everywhere, or the native `/api/chat` endpoint that `llm/ollama_client.py` uses.

In practice Ollama auto-sizes to **32768** for this model, which exceeds what §3 asked for,
so nothing is broken. But the KV cache at 32k is ~4 GB against the ~1 GB budgeted at 8k,
putting total residency near 19.8 GB of 22.4 GiB usable. It fits with less headroom than
planned; pin it via Modelfile if long conversations start spilling to CPU.

---

## 4. Stack

No Docker, no web server. One Python package plus a CLI entry point and a daemon.

| Layer | Pick | Why |
|---|---|---|
| Inference | Ollama | Already running; OpenAI-compatible endpoint at `:11434/v1` |
| Agent loop | `pydantic-ai` | Typed tools, validation, retries, streaming, native MCP support |
| Vector store | **LanceDB** | Embedded, no server, disk-backed, built-in full-text search |
| Retrieval | Hybrid vector + BM25, **rerank off** | Keyword matters here — theorem names, symbols, project codes. Reranking was measured and dropped; see below |
| Doc parsing | `docling` (scientific PDF) · `markitdown` (office/plain) | See §6 |
| CLI | `typer` + `rich` + `prompt_toolkit` | Streaming output, markdown/code rendering, real REPL |
| Storage | SQLite | Memory, metadata, chat history, trace logs — one file, easy to back up |
| Tool packaging | MCP servers, grouped by domain | See §9 |

### The phase-1 retrieval ablation — measured 2026-08-04

1733 chunks from 197 documents (the `fast` tier), scored on the 8 human-verified questions
in `eval/golden.jsonl` with `eval/retrieval_eval.py --compare`:

| config | hit@5 | hit@1 | MRR | p50 |
|---|---|---|---|---|
| vector only | 88% | 88% | 0.875 | 92 ms |
| fts only | 88% | 50% | 0.688 | **17 ms** |
| **hybrid, no rerank** | **88%** | **88%** | **0.875** | **91 ms** |
| hybrid + rerank | 88% | 88% | 0.875 | 27,733 ms |

**Reranking is off.** It did not change a single ranking — identical hit@5, hit@1 and MRR —
at 300× the latency. The §3 claim that it is a "large retrieval gain" is not supported here.
That claim comes from benchmarks on adversarial open-domain sets; a 197-document personal
corpus where each question targets a distinctive document is a much easier ranking problem,
and the first-stage retrieval is already solving it.

**BM25 alone ranks badly** (hit@1 50%, MRR 0.688) while still finding the right document
inside the top 5. That is the expected shape: keyword matching locates, it does not rank.
Keep it in the fusion.

**Caveat, and it is not small: 8 questions.** This is enough to reject a badly wrong config,
not to separate two good ones. Vector-only and hybrid are indistinguishable here, and the
honest reading is that this set cannot tell them apart rather than that they are equivalent.
Revisit once the golden set reaches the 20–30 §10 asks for.

### The one miss is worth more than the score

`"Which cross-validation scheme should the seizure detection experiments use?"` fails against
`Seizure_detection/note.md`, which answers it directly with "ALWAYS USE LOSO —
Leave-One-Subject-Out". The note never contains the words *cross-validation*, *validation* or
*scheme*, so there is no lexical overlap at all, and a competing paper that does discuss
validation explicitly outranks it.

Two things fall out, neither of which reranking fixes:

- **Fusion can demote.** The correct chunk sits at rank 19 under pure vector search and
  rank **31** under hybrid — BM25 cannot see it, and RRF penalises a document only one arm
  found. Hybrid is not free; it trades vocabulary-mismatch recall for keyword precision.
- **This failure class needs query expansion, not better ranking.** Expanding the query with
  acronyms and synonyms before retrieval is the cheap fix, and it belongs in phase 1. Put it
  on the list before adding anything to phase 2.

### A large corpus addition can regress retrieval — re-measure every time

Ingesting the `datasheets` tier took the index from 1,733 to 11,255 chunks and **dropped
hit@5 from 88% to 75%**. A 1700-page reference manual contributes thousands of
near-identical chunks; for a query naming a part it swept the entire top-5, and the user's
own one-page note about that same part fell to **rank 9**.

The fix is a per-document cap, `MAX_PER_DOC = 2` in `retrieval/search.py`:

| cap | hit@5 | MRR |
|---|---|---|
| none | 88% | 0.604 |
| **2** ← chosen | **100%** | 0.656 |
| 1 | 100% | 0.729 |

`1` scores better but forbids two adjacent chunks from one document, which register tables
need. Two things generalise from this:

- **Result diversity is a retrieval parameter, not a nicety.** Five chunks from one source
  is a narrow context for the model to reason over, regardless of whether that source is
  correct.
- **`candidates` and reranking interact.** Raising candidates from 20 to 30 made hit@5
  *worse* (75% vs 88%) with reranking off — a larger pool without a second-stage scorer just
  admits more crowding. Never tune them independently.

This is exactly why §10 insists the eval set exists before the agent: the regression was
invisible without it, and the corpus grew by 6.5× between two sessions.

---

## 5. Voice loop — the interface

Not tools the model calls; the loop it lives inside. Everything local.

| Stage | Component | Runs on | Notes |
|---|---|---|---|
| Mic capture | `getUserMedia` + `AudioWorklet` | **Client browser** | Needs HTTPS. Push-to-talk first; see wake word below |
| Transport | WebSocket, PCM 16 kHz mono | Tailscale | Bidirectional — audio up, audio down |
| VAD / endpointing | `silero-vad` | **Server** | Decides when you stopped talking. Tuning this matters more than STT accuracy. |
| STT | `faster-whisper` small/medium int8 | **Server GPU** | A phone cannot run this; the workstation can |
| LLM | primary model, §3 | Server, `127.0.0.1` | **Thinking mode off on this path** — see below, this is worth 30× |
| TTS | `Piper` | **Server CPU** | Streaming, sentence-chunked |
| Barge-in | Client detects speech, sends cancel | Client → server | Non-negotiable — not being able to interrupt feels broken |

**Wake word depends on the client, and only the laptop can really have one.**

- **Laptop (`eve` CLI):** a native process with direct `sounddevice` access, so
  `openWakeWord` works properly. `eve talk` gets genuine hands-free.
- **Phone:** push-to-talk. Always-on listening needs a foreground app holding the mic, which
  costs battery and which mobile OSes actively fight. Not worth building.

**Do not use the browser's built-in Web Speech API**, however tempting a shortcut it looks.
Chrome's implementation streams audio to Google's servers, which breaks the first line of
this document — and it would do it for every spoken question about the user's own research.
Speech recognition happens on the workstation or it does not happen.

### The phone client — three options, and the tradeoff is not technical

| Option | Effort | Audio leaves your machines? | Proactive push (§8) |
|---|---|---|---|
| **PWA**, installed to home screen | Medium | **No** | Web Push, fiddly |
| **Telegram bot** | **Low** | **Yes** — via Telegram's servers | Free and excellent |
| Native app | High | No | Good |

§7.6 already contemplates a Telegram bot as "a remote interface when away from the desk",
and §12 already lists messaging as an accepted exception to fully-local. It is by far the
least work: no app to build or sideload, voice notes are a first-class Telegram feature,
and it solves phase 4's hardest problem — getting a proactive message onto a phone — for
free.

**But be clear about what it costs.** A Telegram voice note containing a question about
unpublished research travels through Telegram's infrastructure before your workstation ever
sees it. That is a different proposition from sending a text reminder, and it is the one
place in this design where the project's founding premise is genuinely traded away. The PWA
keeps everything on your own hardware and costs a weekend of work.

Decide deliberately rather than by convenience; both are defensible.

**Latency budget — the whole game.** Target under 1000 ms from end-of-speech to first audio:

```
VAD endpoint 300ms → STT 200ms → net 10ms → LLM TTFT 300ms → net 10ms → TTS first chunk 150ms  ≈ 970ms
                                            ^^^^^^^^^^^^^^
                                            measured 4290ms — see below
```

**The LLM term above is superseded by measurement.** Every other term still stands; the
budget as a whole does not. Read the next subsection before designing anything against it.

The two network hops are the new term versus the single-machine version. On wired LAN they
are ~5–10 ms each and irrelevant; **on congested Wi-Fi they are not**, and this is the most
likely thing to quietly break the budget. Measure the round trip early, prefer Ethernet to
the Jetson, and hold the LLM connection open rather than paying TCP/TLS setup per turn.

### The 300 ms TTFT term is wrong. Measured 2026-08-04: 4.3 s.

Phase 0 measured TTFT at **4.29 s** on a 2.5k-token RAG prompt — 14× the budgeted figure.
This was the number §11 predicted was most likely to fail, and it did.

**It is not a model defect and a different model does not fix it.** Both candidates landed
within 6% of each other (4.29 s and 4.05 s) despite a 58% gap in generation speed, because
prefill is compute-bound where generation is bandwidth-bound. Prefilling 2500 tokens through
a 27B is ~135 TFLOPs; this card delivers that in about four seconds at realistic utilisation.
The arithmetic, not the software, sets the floor.

Measured prefill throughput is ~580 tok/s, so cutting context helps roughly linearly
(*estimates — only the 2500-token point is measured; confirm in phase 3*):

| Prompt tokens | Estimated TTFT |
|---|---|
| 2500 (as measured) | 4.3 s |
| 1000 | ~1.7 s |
| 500 (one chunk) | ~0.9 s |
| 175 | ~0.3 s |

**So 300 ms would mean a system prompt and no retrieval at all.** Dropping to `qwen3.5:9b`
buys ~3× and still lands near 1.4 s. There is no configuration of this hardware where a RAG
turn starts speaking in under a second.

**Revised budget — split it by turn type, because the two are not the same problem:**

| Turn type | Example | Target | Reachable? |
|---|---|---|---|
| **No retrieval** | "pause the music", "what's on my calendar" | < 1 s | Yes, unchanged — prompt is short |
| **RAG turn** | "what did my notes say about Noether's theorem" | < 1 s to *first audio*, 3–5 s to substance | Only by masking |

**Masking is the real fix, and it is plumbing rather than model work.** Start TTS on a short
acknowledgement the instant endpointing fires, while prefill runs behind it. Speech buys
1–2 s of cover, which is most of the gap. This is what makes commercial assistants feel
responsive at latencies no better than these. Combine with:

- **Retrieve fewer, smaller chunks** for voice turns specifically — the text path can afford
  context the voice path cannot. This is a per-mode retrieval setting, not a global one.
- **Order the prompt static-content-first** so llama.cpp's prefix cache covers the system
  prompt and tool schemas. Retrieved chunks change per turn and will never cache; put them
  last so they are the only part re-prefilled.
- **Keep the connection open** — still true, and now a smaller term than prefill by far.

Do not spend phase 3 chasing 300 ms. It is not available. Spend it on the acknowledgement
path and per-mode retrieval budgets.

### Thinking mode is the real latency cost, and `/v1` cannot turn it off

Measured 2026-08-12, and it changes the phase-3 plan more than the prefill arithmetic above.

The table in §5 blamed prefill. Prefill is real but it is **not** what dominates. The same
one-sentence question, same model, same machine:

| Route | Thinking | TTFT |
|---|---|---|
| native `/api/chat` | on | 19.57 s |
| **native `/api/chat`** | **off** | **0.66 s** |

**30×.** The row above says "Thinking mode off on this path" and that instruction turns out
to be the single most important line in this section — reasoning traces do not merely blow
the budget, they *are* the budget.

**The catch: only the native endpoint can disable it.** The phase-2 agent runs on
`pydantic-ai` over Ollama's OpenAI-compatible `/v1` route, and every documented way to
suppress thinking there was tried and failed — thinking parts still came back, at 16–27 s:

| Attempt | Result |
|---|---|
| `extra_body={"think": False}` | still thinks |
| `/no_think` appended to the prompt | still thinks |
| `model_settings={"thinking": False}` | still thinks |
| Modelfile `PARAMETER` | no such parameter exists |

This is the same failure as `num_ctx` in §3: the `/v1` shim silently drops Ollama-specific
controls. Consequences:

- **The phase-2 text agent is running with thinking on** and costs 16–36 s per turn. That is
  tolerable at a terminal and fatal to a voice loop.
- **The voice path must go through `llm/ollama_client.py`**, which uses `/api/chat` and
  already honours `think`. This is precisely the seam design rule 8 exists to preserve —
  it just means the voice loop drives the client directly rather than through `pydantic-ai`,
  and needs its own small tool loop.
- **Re-measure the §5 prefill table with thinking off** before designing around it. The
  4.3 s figure from phase 0 was measured correctly with `think=False`; the 33–36 s
  single-shot numbers measured during phase-3 readiness were contaminated by thinking and
  should not be used.

This only works if **every stage streams**, including the network transport. Sentence-chunk
LLM output into TTS so speech begins on the first clause rather than the completed
response. Non-streaming, the identical pipeline takes 4–5 s and feels dead.

---

## 6. Knowledge layer

### The math/science corpus is the hard part

Generic RAG pipelines destroy scientific documents. Budget real time here.

- **Parsing.** `markitdown` and plain PyMuPDF mangle equations into garbage
  (`Z 1 0 f (x)dx`). Use **`docling`** — formula and table understanding, emits LaTeX.
  It is slow and GPU-hungry; run it as an offline batch job, not on demand.
  **Confirmed 2026-08-04**: an 11-page paper came back with 29 inline LaTeX spans
  and page provenance intact, e.g. `$$r _ { 2 } & = r _ { 4 } = 0 \\ ...$$`.
  Two traps found while getting there, both recorded in `ingest/parse.py`:
  `export_to_markdown` defaults to `escape_underscores=True`, which turns every
  subscript `x_1` into `x\_1` — the same corruption this section exists to
  prevent, wearing a different hat. And "GPU-hungry" understates it: **203 s on
  CUDA versus >40 minutes on CPU** for that same paper, so ~18.5 s/page on the
  GPU is the number to plan against. The served 27B must be unloaded first.
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

**`calc` is not optional.** No model at this scale does reliable arithmetic — a bigger or
newer model buys better problem *setup*, not a calculator. Route every computation through
SymPy and the model's job reduces to setting up the problem, which it is genuinely good at.

### 7.4 Actuation — digital
| Tool | Notes |
|---|---|
| `run_shell(cmd)` | **Allowlisted commands only.** Highest-risk tool here. |
| `control_desktop()` | Launch apps, focus/move windows — **`xdotool`**, the workstation is X11 |
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
| `training_status()` | SSH to the workstation — GPU util, VRAM, step/loss, ETA |
| `watch_job(id)` | Proactively announce completion, OOM, NaN loss, plateau |

"Eve, how's the run?" — and more importantly, *it* tells you at 3am that the loss went
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

## 9. Tool routing — ~30 tools against one prompt

Models degrade as the tool list grows, and the 9B fallback degrades sooner than the primary.
Current models tolerate more than the 8B this section was written for, so **treat ~8 as a
starting budget and let the phase-1 eval set tell you the real ceiling** rather than assuming
it. Three layers:

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
| 0 | **Install Ollama** + `uv` venv, **bake off the two 27B candidates** (see below), fetch `bge-m3` **to the HF cache, not Ollama** (§3 — embeddings never share the LLM's Ollama instance), apply tuning, verify a tool call round-trips | Foundation. **Substantially complete 2026-08-04** — see `scripts/phase0.md` for what remains |
| 1 | Ingestion (docling → chunks → LanceDB) + hybrid retrieval + **eval set** | Everything depends on retrieval quality. **Built 2026-08-04**; `fast` + `datasheets` tiers ingested, 11,255 chunks. See `scripts/phase1.md` |
| 2 | `pydantic-ai` agent loop + `calc` + `run_python`, terminal CLI, **trace logging** | Prove tool calling before adding voice. **Built 2026-08-12** — tool selection, citation and no-over-call all verified against the live 27B. See `scripts/phase2.md` |
| 3 | **Voice loop** — STT → retrieve → generate → TTS, fully streaming, served over WebSocket. | Biggest perceived jump. **Built 2026-08-12** — 5.5 s to first audio vs 19.6–31.4 s agentic. See `scripts/phase3.md` |
| 4 | **Daemon + proactive triggers + interrupt policy** | Second biggest. Now it feels alive. |
| 5 | **Jetson bring-up** — identify the board (§2), split the codebase across the network boundary, move voice + daemon to it, expose the client endpoint, add the fallback model | Everything before this runs single-machine; this is the step that makes it always-on |
| 6 | Actuation: Home Assistant, desktop, media, comms | The Iron Man layer |
| 7 | MCP grouping + intent router | Needed once tool count passes ~8 |
| 8 | Job monitoring, `see_screen()` via the primary model's vision, calendar & email | |
| 9 | Fine-tune on logged traces (`TRAINING_PLAN.md`) | Needs phases 2–7 running to generate data |

### The phase-0 bake-off — done 2026-08-04

**Result: `qwen3.6:27b-mtp-q4_K_M`.** Numbers and reasoning in §3; the TTFT consequence is
in §5. Recorded in `assistant/config.py`. Switching now is cheap; switching *after*
collecting traces means retraining, because the adapter is base-model-specific.

What was measured, and why each mattered:

| Measure | Why it decides things |
|---|---|
| **tok/s at 8k context** | Must clear ~3–4 t/s of speech consumption with real margin |
| **TTFT with a 2–3k-token RAG-shaped prompt** | The §5 budget assumes 300 ms. This is the number most likely to fail. |
| Peak VRAM at 8k, thinking off | Both landed near 18 GB — the 22.4 GiB ceiling is not binding at 27B |
| Tool-call validity over ~20 scripted calls | The §10 metric that matters most here |
| Science/math spot check | Independent tests split these two — settle it on *your* questions |

Run with **thinking mode off**, since that is how the voice path will run it.

**How it actually resolved.** Generation speed passed with far more margin than expected —
MTP turned the risk this section was written about into a non-issue. Tool calling was a tie
at 21/22 and needs no work. The decision rule that mattered in the end was none of the ones
written in advance: both models failed TTFT identically, which meant the number was
measuring the hardware rather than the candidates, and the correct response was to rewrite
§5 rather than to reach for `qwen3.5:9b`. A criterion that both options fail by the same
margin is not a tie-break — it is a finding about the budget.

**Build phases 0–4 entirely on the workstation.** It has a GPU, a mic and speakers, and no
network hop to debug through. Do not develop against the Jetson before the thing works
locally — you would be debugging two problems at once.

The one thing to get right early, though, is **the seam**. From phase 2 onward, keep model
access behind a client interface rather than calling Ollama inline, and keep the voice
layer free of direct imports from retrieval and tools. Phase 5 is then a deployment change
instead of a rewrite.

**Phase 1 decides whether any of it is useful** — everything downstream is only as good as
what retrieval hands the model. **Phases 3 and 4 are where "chatbot" becomes "Eve."**

---

## 12. Safety — it can act on the world

- **Confirm before any irreversible action** — writes, sends, purchases, locks, `run_shell`.
  Speak the exact payload back before executing.
- **Allowlist, never blocklist** — shell commands, file paths, home devices.
- **No auto-send** on email or messages. Draft, then confirm.
- **Rate-limit proactive speech** with a hard cap per hour, independent of the policy layer.
- **Physical-world actions get the strictest gate.** Locks and heating are not where you
  want to discover the model misparsed an argument.
- **A global mute/kill word**, handled outside the model, that stops everything instantly.
- Fully local by default. Only optional web search, calendar API and messaging leave the
  machine.
- `.gitignore` the index, SQLite files and raw corpus. Secrets in `.env`, never in prompts.

---

## 13. Repository layout

```
AI_Assistant/
├── assistant/
│   ├── ingest/       # parsers, chunkers, incremental indexer      → workstation
│   ├── retrieval/    # hybrid search, reranker                     → workstation
│   ├── llm/          # model client — the network seam (§11)       → workstation
│   ├── tools/        # knowledge, compute, desktop, home, comms, lab → workstation
│   ├── voice/        # VAD, STT, TTS — server-side (§5)            → workstation
│   ├── daemon/       # triggers, interrupt policy, briefings       → workstation
│   ├── server/       # WebSocket audio endpoint + static PWA       → workstation
│   ├── web/          # browser client: mic capture, playback, PTT  → phone + laptop
│   ├── agent.py      # pydantic-ai agent, routing, mode definitions
│   └── cli.py        # typer + prompt_toolkit REPL
├── eval/             # golden question set + scoring script
├── data/             # gitignored: LanceDB index, sqlite, traces, scratch
├── CLAUDE.md
├── PLAN.md
└── TRAINING_PLAN.md
```

One repository, deployed to both machines — the split is configuration, not a fork. The
arrows are where each package *runs* after phase 5; before that everything runs on the
workstation. `llm/` is the seam that makes that switch cheap, so nothing outside it should
import an Ollama client directly.

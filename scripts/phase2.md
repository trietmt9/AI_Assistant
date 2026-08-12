# Phase 2 runbook — agent loop, tools, trace logging

`PLAN.md` §11 phase 2: **prove tool calling before adding voice.** Phase 3 puts a
microphone in front of whatever this produces, so anything broken here becomes
much harder to debug once latency and audio are in the loop.

---

## What got built

| Module | Does |
|---|---|
| `assistant/llm/provider.py` | Builds the `pydantic-ai` model from settings — the §11 network seam |
| `assistant/tools/compute.py` | `calc` (SymPy + pint) and `run_python` (sandboxed subprocess) |
| `assistant/tools/knowledge.py` | `search_docs` over the phase-1 index |
| `assistant/agent.py` | The agent: 4 tools, system prompt, per-turn state |
| `assistant/tracing.py` | Every turn to SQLite — the phase-9 dataset |
| `assistant/cli.py` | `eve chat` / `ask` / `traces` / `index` |

**Four tools, not eight.** §9 budgets ~8 per prompt; phase 2 uses `search_docs`,
`calc`, `run_python`, `context_now`. The MCP domain routing §9 describes is a
phase-7 problem and building it now would solve a problem that does not exist.

---

## Run it

```bash
.venv/bin/python -m assistant.cli chat          # interactive REPL
.venv/bin/python -m assistant.cli ask "..."     # one-shot, still traced
.venv/bin/python -m assistant.cli traces        # what has accumulated
```

In the REPL: `/stats`, `/clear`, `/help`, Ctrl-D to exit.

The first turn after an idle period pays the ~56 s model load measured in phase 0
(`OLLAMA_KEEP_ALIVE=30m`, so only once per half hour of inactivity).

---

## Verified behaviour, 2026-08-12

The three cases that matter, run against the live 27B:

| Question | Tools called | Result |
|---|---|---|
| "Which STM32 pin is the ADS1298 chip select on my FES board?" | `search_docs` | **PA4 (pin 40)**, cited to `ADS1298_STM32_pinout.md`, plus the reason it is a manual GPIO rather than SPI1_NSS |
| "What is 17 percent of 4320?" | `calc` | 734.4 — routed to SymPy, not computed in-context |
| "Hello Eve." | *(none)* | No over-call |

That is design rule 3 holding (arithmetic goes to `calc`), rule 1 holding (facts
come from RAG with a citation), and the over-calling failure mode from
`TRAINING_PLAN.md` §3 not appearing.

Multi-tool turns work too — *"What cross-validation scheme should I use for the
seizure work, and what is 2^10?"* called `search_docs` **and** `calc`, and
answered both halves correctly with citations.

The strongest result was *"How is the ADS1298 reset pin wired, and what is
sqrt(2) to 8 places?"*, which took three tool calls and cross-referenced two
different source types:

> Per `ADS1298_STM32_pinout.md` p.1, the RESET net (`/~{STM32_ADS_RESET}`) is
> wired from STM32F767 **PB0 (pin 46)** as a GPIO output to ADS1298 pin 36,
> active-low. In firmware (`main_id_read_test.c`) it is configured
> `GPIO_OUTPUT_ACTIVE` (held low on boot), then released high after a 1 ms delay.
> …√2 to 8 dp: 1.41421356

Schematic notes and firmware source, joined in one answer. That is the payoff for
indexing code alongside prose in phase 1, and it is not something the base model
could do — none of that is in its training data.

### The agent reformulates queries, and that matters for how you read the eval

That seizure question is the one the retrieval eval **fails** (`seizure-cv`, the
"cross-validation" vs "LOSO" vocabulary mismatch). The agent got it right anyway,
because it did not search with the raw question — it picked its own search terms.

So `eval/retrieval_eval.py` measures first-shot retrieval on the literal question,
which is a **lower bound** on end-to-end performance, not an estimate of it. Two
consequences worth holding onto:

- Do not conclude that a retrieval miss means a wrong answer. Score generation
  separately, exactly as §10 insists.
- Do not conclude that query expansion is unnecessary either. The model
  compensating is model-dependent, costs an extra round trip, and will be the
  first thing to degrade on the 9B fallback (`PLAN.md` §3).

---

## Two things measured that contradict the plan

### `num_ctx` cannot be set through `pydantic-ai`

`PLAN.md` §3 says to set `num_ctx: 8192` per request. **That does not work on the
OpenAI-compatible route.** Setting `EVELYN_NUM_CTX=2048` and passing
`extra_body={"options": {"num_ctx": 2048}}` changed nothing — `ollama ps` still
reported `32768`. Ollama's `/v1` shim drops the `options` block entirely.

`llm/provider.py` no longer passes it, because configuration that silently does
nothing is worse than no configuration. Ollama auto-sizes instead and currently
picks **32768**, comfortably above the 8192 the plan wanted.

To pin it, bake it into a Modelfile so it applies on every route:

```bash
printf 'FROM qwen3.6:27b-mtp-q4_K_M\nPARAMETER num_ctx 8192\n' > /tmp/Modelfile
ollama create eve-27b -f /tmp/Modelfile
# then set EVELYN_PRIMARY_MODEL=eve-27b
```

**Worth watching:** a 32k KV cache on a 27B is roughly 4 GB, against the ~1 GB
§2 budgeted at 8k. Total sits near 19.8 GB of 22.4 GiB usable — it fits, but with
less headroom than planned. If you see spillover to CPU during long
conversations, the Modelfile above is the fix.

### The retrieval regression from the `datasheets` tier

Ingesting datasheets took the index from 1,733 to 11,255 chunks — **85% of the
corpus is now reference manuals** — and hit@5 on the golden set fell from 88% to
75%.

Cause: for a query naming a part, a 1700-page manual contributes thousands of
near-identical chunks and swept the entire top-5. The user's own distilled note on
that same part fell to **rank 9**.

Fixed with a per-document cap (`MAX_PER_DOC = 2` in `retrieval/search.py`):

| cap | hit@5 | MRR |
|---|---|---|
| none | 88% | 0.604 |
| **2** ← chosen | **100%** | 0.656 |
| 1 | 100% | 0.729 |

`max_per_doc=1` scores better on MRR but forbids a document contributing two
adjacent chunks, which datasheet register tables need. 2 is the compromise; the
8-question set cannot justify a finer distinction.

**Also learned:** raising `candidates` from 20 to 30 made things *worse* without a
reranker (75% vs 88%) — a bigger candidate pool with no second-stage scorer just
admits more crowding. The two settings interact; do not tune them separately.

---

### `run_stream()` truncates turns that narrate before a tool call

Found while renaming, and it would have been much worse to find in phase 3.

The obvious streaming shape is wrong:

```python
async with agent.run_stream(...) as result:        # DON'T
    async for chunk in result.stream_text(delta=True):
        ...
```

`run_stream()` streams the **first** model response. When the model emits text
*and* a tool call in the same response — "My name is Eve. And for 12 factorial:"
followed by `calc(factorial(12))` — leaving the context manager abandons the rest
of the loop. The tool ran, returned `479001600`, and the run simply ended. The
user saw a sentence ending in a colon.

Use `run()` with an `event_stream_handler` instead, which streams every response
and still runs the loop to completion:

```python
async def on_event(_ctx, events):
    async for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            parts.append(event.delta.content_delta)

result = await agent.run(text, deps=deps, event_stream_handler=on_event)
```

One follow-on: **`result.output` is only the final response.** Using it as the
answer drops any pre-tool-call narration from both the screen and the trace, so
`cli.py` keeps the accumulated deltas and falls back to `result.output` only when
nothing streamed.

This matters more in phase 3 than here. Design rule 4 makes non-streaming code in
the voice path a bug, and a truncated turn is silent failure — the TTS would just
stop mid-sentence.

---

## Traces — treat these as production data

```bash
.venv/bin/python -m assistant.cli traces
sqlite3 data/traces.db 'SELECT tool_name, count(*), sum(ok=0) FROM tool_calls GROUP BY 1'
```

`CLAUDE.md` calls the logging path production code, and `TRAINING_PLAN.md` §3
ranks corrected real traces above every synthetic alternative. Two schema
decisions follow from that:

- `turns.messages` holds the **full** `pydantic-ai` message history, not a
  summary. Anything dropped here cannot be recovered later.
- `turns.corrected` and `turns.exclude` exist for hand-correcting bad traces —
  the exact workflow `TRAINING_PLAN.md` §3 describes as the real work.

A failed trace write is logged and swallowed. Losing one trace is a rounding
error; crashing the assistant because a disk filled is not.

---

## What phase 2 does *not* have

Deliberate omissions, so they are not mistaken for oversights:

- **No confirmation gate yet.** §12 requires one before any irreversible action.
  Nothing here is irreversible — `search_docs` and `calc` are pure, `run_python`
  is sandboxed to a scratch dir. The gate arrives with the phase-6 actuation
  tools, using `pydantic-ai`'s `requires_approval` / `ApprovalRequired`.
- **`run_python` network isolation is best-effort.** It uses `unshare -rn` when
  available and logs a warning when not. Verify on this host with
  `unshare -rn true`; if that fails, the subprocess has network access and the
  §12 claim does not hold.
- **No generation scoring.** `eval/retrieval_eval.py` measures retrieval only.
  Scoring answer correctness needs the `expect_answer` fields in
  `eval/golden.jsonl`, which are still empty for 10 of 18 questions.

---

## Before phase 3

**Blocking — thinking mode.** The agent runs on `/v1`, which cannot disable it, so
every turn carries a reasoning trace: measured **2.9 s** for a no-tool turn,
**19.6 s** for one `search_docs`, **31.4 s** for three. The same question on the
native endpoint with `think=False` returns first tokens in **0.66 s**. Full
measurements in `PLAN.md` §5. Phase 3 cannot start until this is decided, because
the voice loop's entire budget is smaller than the thinking overhead.

The choice is architectural, not a config tweak:

| Option | Cost |
|---|---|
| Voice loop drives `llm/ollama_client.py` directly (native, `think=False`) | Needs its own small tool loop; `pydantic-ai` stays for the text path |
| Wait for Ollama `/v1` to honour `think` | Unbounded; not in your control |
| Run the voice path with no tools at all (pre-retrieve, single generation) | Simplest, one prefill, but loses `calc` and multi-step |

The first is the one the architecture was already built for — design rule 8 put
the seam exactly here, and `ollama_client.py` already honours `think`.

**Also worth doing, unchanged:**

1. **Finish the golden set** — 8 verified of 18, against the 20–30 §10 wants.
   Gates generation scoring too.
2. **Add query expansion.** The known `seizure-cv` miss is a pure vocabulary
   mismatch ("cross-validation" vs "LOSO") that no amount of ranking fixes.
3. **Use it for a week.** `TRAINING_PLAN.md` §3 wants real traces from real use.
   Note the traces collected so far were made with thinking on — if the text path
   changes, the input distribution changes with it.

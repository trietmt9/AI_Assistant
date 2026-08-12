# Phase 0 runbook

Commands for you to run. Read the "what success looks like" line under each step before
moving on. See `PLAN.md` §11 for why phase 0 exists and what it gates.

---

## State as of 2026-08-04

| | |
|---|---|
| Ollama | 0.32.5 installed, service active, GPU detected as `compute=7.5` |
| Tuning | 4 of 5 applied — `OLLAMA_HOST` deliberately deferred, see below |
| venv | `.venv` on Python 3.12.3, core deps installed |
| Scaffold | `assistant/`, `eval/`, `data/` per `PLAN.md` §13 |
| Usable VRAM | **22.4 GiB** (23.5 total minus ~0.8 for X11) |
| Models | `qwen3.6:27b-mtp-q4_K_M`, `qwen3.5:27b`, `qwen3.5:0.8b` |
| **Primary model** | **`qwen3.6:27b-mtp-q4_K_M`** — decided, set in `assistant/config.py` |
| Embeddings | **not installed — this is what remains** |

**Steps 0–4 are complete.** Jump to **step 5**. The earlier steps are kept because they
document how the decisions were reached, not because they need running again.

---

## Steps 0–4 — done, kept for the record

<details>
<summary>Step 0 — check size before pulling (how the 35B MoE got ruled out)</summary>

A 24 GB card does not hold a 24 GB model. The MoE's weights are 22.3 GiB against 22.4 GiB
available. Run this before pulling anything new, ever:

```bash
check_size() {
  curl -s "https://registry.ollama.ai/v2/library/${1}/manifests/${2}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
m=[l for l in d['layers'] if 'model' in l['mediaType']][0]
print(f\"${1}:${2} = {m['size']/2**30:.2f} GiB\")"
}
```
</details>

<details>
<summary>Steps 1–3 — pull candidates, smoke-test the harness, run the bake-off</summary>

The smoke test on `qwen3.5:0.8b` caught a real bug: a reported 50 s TTFT that turned out to
be **cold model load**, not latency. `warmup()` now runs before timing and load time gets
its own column. Both 27B candidates were then measured with `eval/bakeoff.py`.
</details>

**Step 4 — the result.** `qwen3.6:27b-mtp-q4_K_M` wins on speed at a tie on tool calling:

| | TTFT | tok/s | peak VRAM | cold load | tools ok | over | bad |
|---|---|---|---|---|---|---|---|
| **`qwen3.6:27b-mtp-q4_K_M`** | 4.29 s | **44.8** | 18.2 GB | 56 s | 21/22 | 0 | 0 |
| `qwen3.5:27b` | 4.05 s | 28.3 | 17.4 GB | 19 s | 21/22 | 0 | 0 |

Three things came out of it, all now written into the plan:

1. **MTP works on sm_75** — 1.58×, undocumented publicly. `PLAN.md` §3.
2. **Tool calling needs no fine-tuning.** `TRAINING_PLAN.md` §0 gate has fired against
   training.
3. **TTFT fails by 14× and no model fixes it** — prefill is compute-bound. `PLAN.md` §5 was
   rewritten around masking rather than chasing 300 ms.

Keep `qwen3.5:27b` pulled as the A/B control for phase-1 retrieval work. Do not pull the
plain `qwen3.6:27b` control — MTP is proven, so it would answer a settled question.

---

## Step 5 — Embeddings and reranker (the last phase-0 item)

`bge-m3` is the one §11 deliverable still missing. **It does not go through Ollama** —
`PLAN.md` §3 and CLAUDE.md design rule 5: sharing the GPU with the LLM causes model eviction
and a reload penalty per query, and you just measured that penalty at **56 s**. Embeddings
run on CPU via `sentence-transformers`.

```bash
uv sync --extra retrieval
```

Heavy — it pulls `torch`, `docling` and `lancedb`. Expect several minutes and ~3 GB.

Then fetch both models into the HF cache and prove they work on CPU. ~4.4 GB of downloads
on first run, cached afterwards:

```bash
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder

e = SentenceTransformer('BAAI/bge-m3', device='cpu')
v = e.encode(['the canonical partition function of an ideal gas'])
print('bge-m3       ', v.shape, '(expect (1, 1024))')

r = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
s = r.predict([
    ('what is entropy', 'entropy measures the number of accessible microstates'),
    ('what is entropy', 'the cat sat on the mat'),
])
print('reranker     ', s, '(first must score higher)')
"
```

**Success:** `(1, 1024)` from the embedder, and a reranker score that is clearly higher for
the relevant passage than the irrelevant one.

Now confirm rule 5 actually holds — that nothing touched the GPU:

```bash
ollama ps                       # the 27B may be resident; that is fine
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Success:** no `python` process appears in the compute-apps list. If one does, a
`device='cpu'` argument was dropped somewhere and it will cost you a model reload on every
single query once phase 1 is running.

---

## Step 6 — Phase-0 exit check

One command that verifies the whole seam: config resolves to the chosen model, the endpoint
answers, and a tool call round-trips through `assistant/llm/`.

```bash
.venv/bin/python -c "
import asyncio, json
from assistant.config import settings
from assistant.llm.base import Message
from assistant.llm.ollama_client import OllamaClient

TOOL = [{'type':'function','function':{
    'name':'calc',
    'description':'Evaluate a mathematical expression with SymPy.',
    'parameters':{'type':'object','properties':{'expression':{'type':'string'}},
                  'required':['expression']}}}]

async def main():
    c = OllamaClient()
    print('primary_model:', settings.primary_model)
    assert settings.primary_model == 'qwen3.6:27b-mtp-q4_K_M', 'config drifted'
    assert await c.health(), 'ollama unreachable'
    calls = []
    async for ch in c.chat([Message('user','What is 17% of 4320?')],
                           tools=TOOL, think=False):
        calls.extend(ch.tool_calls)
    assert calls, 'no tool call — the seam is broken'
    print('tool call    :', json.dumps(calls[0], indent=2))
    await c.aclose()
    print('\nphase 0 complete.')

asyncio.run(main())
"
```

**Success:** it prints a well-formed `calc` call and `phase 0 complete.` The model must
route arithmetic to the tool rather than answering `734.4` itself — that is design rule 3
working end to end.

Then start phase 1. Per `PLAN.md` §11 it gates everything downstream, and the eval set comes
*before* the agent — retrieval failures and generation failures need opposite fixes and have
to be measured separately.

---

## Deliberately not done

**`OLLAMA_HOST=0.0.0.0:11434` is not set**, though `PLAN.md` §3 lists it. Its only purpose is
letting the Jetson reach the workstation, and §2 defers the Jetson entirely. Setting it now
exposes an unauthenticated port to the LAN for no benefit — Ollama has no auth of its own.
Set it in phase 5, together with the firewall rule, not before.

---

## Useful checks

```bash
# is the service healthy
systemctl is-active ollama && curl -s http://127.0.0.1:11434/api/tags | head -c 200

# what is actually loaded in VRAM right now
ollama ps
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# did the tuning actually apply
systemctl show ollama --property=Environment | tr ' ' '\n' | grep -i ollama

# re-run the bake-off (results land in eval/bakeoff_results.json)
.venv/bin/python eval/bakeoff.py qwen3.6:27b-mtp-q4_K_M qwen3.5:27b

# disk
df -h /
```

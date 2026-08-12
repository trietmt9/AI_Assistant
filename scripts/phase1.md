# Phase 1 runbook — ingestion, retrieval, eval

Phase 1 gates everything downstream (`PLAN.md` §11). Nothing here is optional and
the order matters: **the eval set exists before the agent**, because retrieval
failures and generation failures need opposite fixes and cannot be told apart
after the fact.

---

## What got built

| Module | Does |
|---|---|
| `assistant/ingest/sources.py` | The allowlist. Corpus scope lives here and nowhere else. |
| `assistant/ingest/parse.py` | docling for PDF/office, plain read for md/code. Caches to `data/parsed/`. |
| `assistant/ingest/chunk.py` | Heading-aware prose chunker, function-boundary code chunker. |
| `assistant/ingest/index.py` | Incremental indexer: mtime → content hash → text hash ladder. |
| `assistant/retrieval/embed.py` | bge-m3 on CPU. |
| `assistant/retrieval/store.py` | LanceDB, vector + BM25 in one table. |
| `assistant/retrieval/search.py` | Hybrid search, cross-encoder rerank, metadata filters. |
| `eval/golden.jsonl` | Golden questions. 8 verified, ~12 stubs. |
| `eval/retrieval_eval.py` | hit@k, MRR, latency, and the config ablation. |

---

## Corpus scope, as surveyed 2026-08-04

```bash
.venv/bin/python -c "
from assistant.ingest.sources import discover, summarise, TIERS
for t in TIERS: print(f'--- {t} ---'); print(summarise(discover((t,))))"
```

| Tier | Files | Size | What |
|---|---|---|---|
| `fast` | 222 | 28 MB | 11 research papers, 39 notes/READMEs, 170 hand-written source files, 2 admin PDFs |
| `datasheets` | 18 | 117 MB | ADS1298, WT02C40C, RM0090, RM0390, ARM DUI0553 |
| `books` | 3 | 500 MB | Serway 10e (duplicated), Fuzzy Logic for Engineers |

**A naive walk finds ~36,000 files.** The allowlist rejects ~27,000 of them:
virtualenvs, Zephyr build output, vendored STM32 HAL, and
`AI_Pattern_Recognition/classwork` alone at 20,545 files. Precision matters more
than recall on a corpus this small — burying 150 hand-written files under 27,000
vendored ones makes every retrieval worse.

**`ACQ_Read/tSCS_data/` is excluded deliberately.** It holds 145 raw measurement
files in directories named after individual research participants. Numeric data
retrieves badly anyway, but the real point is that indexing it would copy those
names into the index and into every trace log downstream. Opt in consciously if
you ever want it.

---

## Step 1 — Free the GPU before ingesting

**docling on CPU is not viable.** Measured on an 11-page paper:

| Device | Time |
|---|---|
| CUDA | **203 s** (~18.5 s/page) |
| CPU | still running at 40 min when killed — >12× slower |

The GPU holds the served 27B under `OLLAMA_KEEP_ALIVE=30m`, so unload it first.
This is reversible; the next chat request reloads it (at the 56 s cold-load cost
phase 0 measured).

```bash
ollama stop qwen3.6:27b-mtp-q4_K_M
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect ~550 MiB (X11)
```

---

## Step 2 — Ingest, one tier at a time

```bash
.venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from assistant.ingest.index import build_index
print(build_index(('fast',), device='cuda', embed_device='cuda').summary())"
```

**Success:** `parsed N | ... | +M chunks`, few or no failures. Re-running is
near-instant — the manifest skips unchanged files on mtime alone.

### Where the time actually goes — measured on the `fast` tier

The first run took 75 minutes and produced 1566 chunks. The breakdown was not
what §6 implies:

| Stage | Time | Note |
|---|---|---|
| docling parse (11 PDFs) | **9.2 min** | on CUDA |
| everything else | ~1 min | 200 markdown/code files |
| **bge-m3 embedding, CPU** | **~65 min** | 1566 chunks at ~2.5 s each |

**Embedding was 87% of the run, not parsing.** bge-m3 is a 568M XLM-RoBERTa
model, so ~2.5 s/chunk is its honest CPU rate rather than a misconfiguration.

Hence `embed_device='cuda'` above. This looks like it violates design rule 5, and
it does not: that rule exists so a GPU embedder does not evict the served model
and charge a reload on **every query**. Batch ingestion is offline, runs with the
27B already unloaded for docling, and is the one place the rule's rationale does
not apply. Query embedding stays on CPU, where it costs ~50 ms and the rule holds.

Re-measure before committing to the big tiers — at CPU rates, `datasheets` and
`books` are days rather than hours, and that alone may decide whether they are
worth ingesting whole.

Then the expensive tiers, when you are ready to leave the machine alone:

```bash
# datasheets -- RM0090 is ~1700 pages by itself. Budget overnight.
.venv/bin/python -c "
from assistant.ingest.index import build_index
print(build_index(('datasheets',), device='cuda').summary())"

# books -- Serway is ~1500 pages. The duplicate copy is dropped automatically
# by text hash, so you pay for it once.
.venv/bin/python -c "
from assistant.ingest.index import build_index
print(build_index(('books',), device='cuda').summary())"
```

**Question the `datasheets` tier before running it.** At 18.5 s/page, ~4000
pages of reference manual is roughly 20 hours. Two full STM32 reference manuals
are the bulk of that, and you rarely need all 1700 pages of one — consider
splitting out the chapters you actually use rather than ingesting them whole.
The `books` tier deserves the same question for a different reason: a general
physics textbook is largely material the 27B already knows, so it buys less than
the papers and datasheets, which are genuinely yours.

---

## Step 3 — Decide the retrieval config with the ablation

Do not take `PLAN.md` §4's "hybrid + rerank" on faith. Measure it here:

```bash
.venv/bin/python eval/retrieval_eval.py --compare --verified-only
```

This prints hit@5, hit@1, MRR and latency for vector-only, FTS-only, hybrid, and
hybrid+rerank. **Pick on hit@k, break ties on MRR, use latency only to reject.**

### Result, 2026-08-04 — reranking is off

| config | hit@5 | hit@1 | MRR | p50 |
|---|---|---|---|---|
| vector only | 88% | 88% | 0.875 | 92 ms |
| fts only | 88% | 50% | 0.688 | 17 ms |
| **hybrid, no rerank** ← chosen | **88%** | **88%** | **0.875** | **91 ms** |
| hybrid + rerank | 88% | 88% | 0.875 | 27,733 ms |

`rerank_enabled = False` is now the default in `config.py`. Re-run this after
growing the golden set — 8 questions can reject a bad config but cannot separate
two good ones, and vector-only vs hybrid is currently a tie this set cannot break.

### Why the reranker looked essential and was not

`PLAN.md` §3 lists `bge-reranker-v2-m3` as "large retrieval gain, zero VRAM" and
puts it on the CPU. The VRAM claim is true; the implied cheapness is not.
Measured, warm, at `max_length=512`:

| Candidates | Rerank latency |
|---|---|
| 10 | 8.1 s |
| 20 | 14.7 s |
| 30 | 26.5 s |
| **no rerank** | **0.08 s** |

It is a 560M-parameter cross-encoder scoring every candidate against the query,
and this CPU is an i9-9900K. Two things follow:

1. **A bug worth knowing about:** LanceDB's `CrossEncoderReranker` never sets
   `max_length`, so it defaults to the model's full 8192-token window — that was
   40–95 s per query before `search.py` subclassed it to bound the length. If you
   ever swap rerankers, keep that override.
2. **It changed nothing.** Not "helped a little" — identical hit@5, hit@1 and MRR.
   §3's "large retrieval gain" comes from adversarial open-domain benchmarks; a
   197-document personal corpus where each question targets a distinctive
   document is a far easier ranking problem that first-stage retrieval already
   solves.

If a larger golden set ever shows reranking separating configs, the fallback is
`EVELYN_RERANK_DEVICE=cuda` in `.env` (~1.1 GB alongside the 18 GB served model,
fits in 22.4 GiB) plus `EVELYN_RERANK_ENABLED=true`. That trades against design
rule 5, so measure before adopting it.

### The known failure, and what actually fixes it

One of the 8 fails: *"Which cross-validation scheme should the seizure detection
experiments use?"* The answer is in `Seizure_detection/note.md` — "ALWAYS USE
LOSO — Leave-One-Subject-Out" — but that note never contains the words
*cross-validation*, *validation* or *scheme*. Zero lexical overlap.

The correct chunk ranks **19 under pure vector search and 31 under hybrid**: BM25
cannot see it at all, and RRF penalises a document only one arm found. So hybrid
is not free — it trades vocabulary-mismatch recall for keyword precision.

Reranking does not fix this (it was in the candidate pool and still lost). **Query
expansion does**: expand acronyms and synonyms before retrieval. That is the next
retrieval-quality task, and it belongs here in phase 1 rather than being deferred.

---

## Step 4 — Finish the golden set

`eval/golden.jsonl` ships with **8 verified questions and ~12 unverified stubs**.
`PLAN.md` §10 wants 20–30 verified. The 8 were written by reading the source
files directly, so both `expect_paths` and `expect_answer` are checked; the stubs
have plausible `expect_paths` and an empty `expect_answer`.

```bash
# only the human-checked ones -- the honest number
.venv/bin/python eval/retrieval_eval.py --verified-only

# everything, including stubs -- provisional
.venv/bin/python eval/retrieval_eval.py
```

Fill in a stub by opening the source, writing the true answer into
`expect_answer`, and setting `verified: true`. Add questions as you use the
system: **the questions you actually ask are worth more than the ones you invent
for a test set.** Cover the awkward cases deliberately — questions scoped by
source ("in my FES notes"), questions needing a specific page, and questions the
corpus genuinely cannot answer.

Keep `expect_answer` filled even though phase 1 does not score it. Phase 2 grades
generation against exactly those strings, and writing them now, while you are
reading the source anyway, costs nothing.

---

## Useful checks

```bash
# what is in the index
.venv/bin/python -c "
from assistant.retrieval.store import ChunkStore
s=ChunkStore(); print(s.count(), 'chunks')
import collections
t=s.table.to_arrow().select(['doc_type']).column('doc_type').to_pylist()
print(collections.Counter(t))"

# one-off query
.venv/bin/python -c "
from assistant.retrieval.search import search
for h in search('ADS1298 chip select pin', k=5, rerank=False): print(f'{h.score:.3f} {h.citation}')"

# what has been parsed and cached
du -sh data/parsed data/index 2>/dev/null
sqlite3 data/index_manifest.db 'SELECT doc_type, count(*), sum(n_chunks) FROM documents GROUP BY 1'

# force a full re-index (after changing the chunker)
# parses are cached by content hash, so this does NOT re-run docling
.venv/bin/python -c "
from assistant.ingest.index import build_index
print(build_index(('fast',), force=True).summary())"
```

That last point is the reason `data/parsed/` exists: **changing the chunker costs
minutes, not hours.** docling output is cached by content hash, so re-chunking
and re-embedding replays from disk. Bump `CACHE_VERSION` in `parse.py` only when
the parse itself changes.

"""Hybrid retrieval: vector + BM25, fused, then cross-encoder reranked.

PLAN.md §4. Both halves earn their place on this corpus:

* **Dense** finds "how do I stop the motor overshooting" against prose that
  never uses those words.
* **BM25** finds `ADS1298`, `RM0090`, `Noether`, `tSCS`. Dense embeddings place
  `ADS1298` and `ADS1299` almost on top of each other; BM25 does not, and they
  are different chips.

Reranking is the cheapest large quality win available (PLAN.md §3: "large
retrieval gain, zero VRAM"), because the cross-encoder sees query and passage
together instead of comparing two independently-computed vectors.

Everything here runs on the CPU. Nothing in this module touches Ollama or the
GPU -- see `embed.py` for why that is deliberate.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from assistant.config import settings
from assistant.retrieval.embed import embed_query
from assistant.retrieval.store import ChunkStore

log = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Fetch this many before reranking. The cross-encoder is the expensive stage, so
# this is the main latency dial: more candidates means better recall and a
# slower turn.
DEFAULT_CANDIDATES = 20
DEFAULT_K = 5

# **This is the difference between a usable retriever and an unusable one.**
#
# bge-reranker-v2-m3 is XLM-RoBERTa-large with an 8192-token window, and
# LanceDB's CrossEncoderReranker never sets `max_length` -- so by default it
# scores every candidate at full window width. Measured on this CPU that is
# 40-95 s per query, against ~1.5 s at 512. Cross-encoder rerankers are trained
# and benchmarked at 512; the tail of a 768-token chunk contributes almost
# nothing to a relevance judgement, so this costs accuracy far less than it
# saves time. Verify with `retrieval_eval.py --compare` rather than trusting it.
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16

# At most this many chunks from any one document in a result set.
#
# Added after the `datasheets` tier landed and dropped hit@5 from 88% to 75%.
# A 1700-page reference manual contributes thousands of near-identical chunks,
# and for a query naming a part it swept the entire top-5 -- the user's own
# distilled note on that same part fell to rank 9. Capping per document fixes
# both halves of that: it restores the note, and it stops any single source
# monopolising the context window the model gets to reason over.
#
# Set to 0 to disable.
MAX_PER_DOC = 2


@dataclass(slots=True)
class Hit:
    text: str
    doc_path: str
    doc_title: str
    doc_type: str
    section: str
    page: int
    chunk_index: int
    score: float
    has_equation: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """What gets shown to the user and put in the model's context.

        PLAN.md §6 keeps page and section in metadata precisely so an answer can
        point at "Serway §7.3, p.212" rather than "your documents".
        """
        bits = [Path(self.doc_path).name]
        if self.section:
            bits.append(self.section)
        if self.page:
            bits.append(f"p.{self.page}")
        return " | ".join(bits)

    def render(self, max_chars: int = 1200) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " ..."
        return f"[{self.citation}]\n{body}"


@dataclass(slots=True)
class SearchTiming:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms + self.rerank_ms


@lru_cache(maxsize=1)
def _reranker():
    """bge-reranker-v2-m3, CPU, with the sequence length actually bounded.

    Subclassed rather than configured because LanceDB exposes no way to set
    `max_length` -- see the constant above for why that matters so much here.
    """
    from functools import cached_property

    from lancedb.rerankers import CrossEncoderReranker

    class BoundedCrossEncoderReranker(CrossEncoderReranker):
        @cached_property
        def model(self):
            from sentence_transformers import CrossEncoder

            log.info("loading %s on cpu (max_length=%d)", self.model_name, RERANK_MAX_LENGTH)
            return CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                max_length=RERANK_MAX_LENGTH,
            )

    return BoundedCrossEncoderReranker(
        model_name=RERANKER_MODEL, column="text", device=settings.rerank_device
    )


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _build_filter(
    doc_type: str | Iterable[str] | None,
    source: str | None,
    min_mtime: float | None,
) -> str | None:
    """SQL predicate for LanceDB's `where`.

    Metadata filters answer scoped questions ("in my FES board notes", "papers
    only") far better than embeddings do -- PLAN.md §6 makes that point and it is
    why these fields are in the schema at all.
    """
    clauses: list[str] = []
    if doc_type:
        types = [doc_type] if isinstance(doc_type, str) else list(doc_type)
        joined = ", ".join(f"'{_escape(t)}'" for t in types)
        clauses.append(f"doc_type IN ({joined})")
    if source:
        clauses.append(f"doc_path LIKE '%{_escape(source)}%'")
    if min_mtime is not None:
        clauses.append(f"mtime >= {float(min_mtime)}")
    return " AND ".join(clauses) if clauses else None


def _diversify(hits: list[Hit], *, k: int, max_per_doc: int) -> list[Hit]:
    """Take the top `k`, allowing at most `max_per_doc` chunks per document.

    Order is otherwise preserved, so this only ever demotes — a document that
    genuinely owns the answer still leads. If the cap leaves fewer than `k`
    results, the overflow is appended back in original order rather than
    returning a short list: a strict cap would be worse than a crowded result.
    """
    if max_per_doc <= 0:
        return hits[:k]

    kept: list[Hit] = []
    overflow: list[Hit] = []
    seen: dict[str, int] = {}
    for h in hits:
        n = seen.get(h.doc_path, 0)
        if n < max_per_doc:
            kept.append(h)
            seen[h.doc_path] = n + 1
        else:
            overflow.append(h)
        if len(kept) == k:
            return kept
    return (kept + overflow)[:k]


def _to_hits(rows: list[dict]) -> list[Hit]:
    hits: list[Hit] = []
    for r in rows:
        # LanceDB names the score differently per path: `_relevance_score` after
        # a reranker, `_distance` for pure vector, `_score` for pure FTS.
        score = r.get("_relevance_score")
        if score is None:
            score = r.get("_score")
        if score is None and "_distance" in r:
            score = 1.0 - float(r["_distance"])  # cosine distance -> similarity
        hits.append(
            Hit(
                text=r.get("text", ""),
                doc_path=r.get("doc_path", ""),
                doc_title=r.get("doc_title", ""),
                doc_type=r.get("doc_type", ""),
                section=r.get("section", ""),
                page=int(r.get("page") or 0),
                chunk_index=int(r.get("chunk_index") or 0),
                score=float(score if score is not None else 0.0),
                has_equation=bool(r.get("has_equation")),
            )
        )
    return hits


class Retriever:
    """Holds the store handle and the lazily-loaded reranker."""

    def __init__(self, store: ChunkStore | None = None) -> None:
        self.store = store or ChunkStore()

    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        candidates: int = DEFAULT_CANDIDATES,
        doc_type: str | Iterable[str] | None = None,
        source: str | None = None,
        min_mtime: float | None = None,
        rerank: bool | None = None,
        mode: str = "hybrid",  # hybrid | vector | fts
        max_per_doc: int | None = None,
        timing: SearchTiming | None = None,
    ) -> list[Hit]:
        if not self.store.exists() or self.store.count() == 0:
            log.warning("index is empty -- run the ingest first")
            return []

        if rerank is None:
            rerank = settings.rerank_enabled
        t = timing or SearchTiming()
        table = self.store.table
        where = _build_filter(doc_type, source, min_mtime)

        t0 = time.perf_counter()
        vector = embed_query(query) if mode in {"hybrid", "vector"} else None
        t.embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if mode == "hybrid":
            builder = table.search(query_type="hybrid").vector(vector).text(query)
        elif mode == "vector":
            builder = table.search(vector, query_type="vector")
        elif mode == "fts":
            builder = table.search(query, query_type="fts")
        else:
            raise ValueError(f"unknown mode {mode!r}")

        builder = builder.limit(max(candidates, k))
        if where:
            # `prefilter` keeps the filter from silently shrinking the result set
            # below k by applying it after the top-n cut.
            builder = builder.where(where, prefilter=True)

        if rerank:
            builder = builder.rerank(_reranker())

        try:
            rows = builder.to_list()
        except Exception as exc:
            # An FTS query with only stopwords, or a filter matching nothing,
            # should return nothing rather than take the caller down.
            log.warning("search failed for %r: %s", query, exc)
            return []
        t.search_ms = (time.perf_counter() - t0) * 1000

        cap = MAX_PER_DOC if max_per_doc is None else max_per_doc
        hits = _diversify(_to_hits(rows), k=k, max_per_doc=cap)
        if timing is not None:
            timing.embed_ms, timing.search_ms = t.embed_ms, t.search_ms
        return hits


# Module-level convenience, so callers do not each build a Retriever.
@lru_cache(maxsize=1)
def _default_retriever() -> Retriever:
    return Retriever()


def search(query: str, **kw) -> list[Hit]:
    """Search the index. See `Retriever.search` for options."""
    return _default_retriever().search(query, **kw)


__all__ = ["Hit", "Retriever", "search", "SearchTiming", "DEFAULT_K", "DEFAULT_CANDIDATES"]

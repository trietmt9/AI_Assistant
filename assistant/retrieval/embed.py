"""bge-m3 embeddings, on the CPU, deliberately.

CLAUDE.md design rule 5 and PLAN.md §3: embeddings never go through Ollama.
Sharing the GPU with the served model causes eviction and a reload penalty on
every alternation -- and phase 0 measured that penalty at **56 seconds** for the
27B. A query embedding is one short string and costs ~20 ms on this CPU, so
there is nothing to gain and a great deal to lose.

bge-m3 is multilingual, which this corpus needs: the admin PDFs are Chinese and
the research notes are English.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from assistant.config import DATA_DIR

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
MODELS_DIR = DATA_DIR / "models"

# Chunks target 768 tokens and rarely exceed 1024, but a preserved table or code
# fence can run longer. Capping here bounds the worst-case CPU cost; bge-m3
# would otherwise happily attempt its full 8192 window on a single chunk.
MAX_SEQ_LENGTH = 2048


def _local_copy() -> Path | None:
    """A local safetensors copy of bge-m3, if one has been made.

    **Worth 4x on startup.** The HuggingFace cache for `BAAI/bge-m3` holds
    `pytorch_model.bin` — a 2.2 GB legacy pickle — and carries a `.no_exist`
    marker for `model.safetensors`, so `sentence-transformers` deserialises the
    whole pickle into RAM on every load. Measured 2026-08-12: **5.9 s from the
    HF cache versus 1.4 s from a local safetensors copy**, repeatably.

    This is also the pause that looks like a hang: the "Loading weights 100%"
    progress bar covers reading the *shard index*, and the real work happens
    silently afterwards.

    Create it once with `eve prepare-models`.
    """
    path = MODELS_DIR / "bge-m3"
    return path if (path / "model.safetensors").exists() else None


@lru_cache(maxsize=2)
def _model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    source = _local_copy()
    log.info("loading %s on %s%s", MODEL_NAME, device, " (local safetensors)" if source else "")
    model = SentenceTransformer(str(source) if source else MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def make_local_copy(force: bool = False) -> Path:
    """Convert the cached model to safetensors once, for fast loads thereafter."""
    from sentence_transformers import SentenceTransformer

    target = MODELS_DIR / "bge-m3"
    if (target / "model.safetensors").exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("converting %s to safetensors (once)...", MODEL_NAME)
    SentenceTransformer(MODEL_NAME, device="cpu").save(str(target))
    return target


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    show_progress: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a batch of passages. Returns (n, 1024) float32, L2-normalised.

    Normalising means cosine similarity is a plain dot product, which is what
    LanceDB's default metric expects.

    **`device="cuda"` is correct for batch ingestion and wrong for queries.**
    Design rule 5 exists to stop the GPU embedder evicting the served model and
    charging a reload on *every query* -- that is a serving-time concern. Batch
    ingestion is offline, runs with the 27B already unloaded for docling, and is
    dominated by exactly this cost: phase 1 spent 66 of its 75 minutes embedding
    1566 chunks on CPU at ~2.5 s each. bge-m3 is a 568M XLM-R model, so that is
    the expected CPU rate, not a misconfiguration.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    vecs = _model(device).encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. ~20 ms.

    bge-m3 needs no asymmetric query prefix -- unlike bge-v1.5, which wanted
    "Represent this sentence for searching relevant passages:". Adding one here
    would degrade retrieval rather than improve it.
    """
    return embed_texts([text])[0]


def warm() -> None:
    """Force the model into memory. Call before timing anything."""
    embed_query("warm")


__all__ = ["embed_texts", "embed_query", "warm", "make_local_copy", "EMBED_DIM", "MODEL_NAME"]

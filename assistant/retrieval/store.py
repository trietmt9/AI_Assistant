"""LanceDB: the vector store and the full-text index, in one embedded file.

PLAN.md §4 picked LanceDB because it needs no server and holds both halves of
hybrid retrieval (§4: "Hybrid vector + BM25, then rerank"). Keyword search is
not optional for this corpus -- half the real questions name a theorem, a
register, a part number or a project, and those are exactly the tokens dense
embeddings blur together. `ADS1298` and `ADS1299` are near-identical vectors and
completely different chips.

`data/` is gitignored in full (CLAUDE.md), index included.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from assistant.config import settings
from assistant.ingest.chunk import Chunk
from assistant.retrieval.embed import EMBED_DIM

log = logging.getLogger(__name__)

TABLE_NAME = "chunks"

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        # `text` is what gets retrieved and shown; `embed_text` is what was
        # embedded (text plus title/section breadcrumb). Keeping both means the
        # FTS index can search the breadcrumb without polluting what the model
        # is handed back.
        pa.field("text", pa.string()),
        pa.field("embed_text", pa.string()),
        pa.field("doc_path", pa.string()),
        pa.field("doc_title", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("section", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("n_tokens", pa.int32()),
        pa.field("content_hash", pa.string()),
        pa.field("mtime", pa.float64()),
        pa.field("has_equation", pa.bool_()),
    ]
)

# BM25 runs over `embed_text`, not `text`. LanceDB's native FTS indexes exactly
# one field, and `embed_text` is already "title | section\n\ntext" -- so a single
# index covers the chunk body *and* its title and heading breadcrumb. That
# breadcrumb is worth having in the keyword index: "Managing Labels" or
# "SPI + Control Signal Table" are precisely the phrases a keyword query hits.
FTS_COLUMN = "embed_text"


def _escape(value: str) -> str:
    return value.replace("'", "''")


class ChunkStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.index_dir
        self.path.mkdir(parents=True, exist_ok=True)
        import lancedb

        self._db = lancedb.connect(str(self.path))

    # --- table lifecycle -------------------------------------------------

    @property
    def table(self):
        if TABLE_NAME not in self._db.table_names():
            return self._db.create_table(TABLE_NAME, schema=SCHEMA)
        return self._db.open_table(TABLE_NAME)

    def exists(self) -> bool:
        return TABLE_NAME in self._db.table_names()

    def count(self) -> int:
        return self.table.count_rows() if self.exists() else 0

    # --- writes ----------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Append chunks with their vectors. Caller handles delete-before-add."""
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        rows: list[dict[str, Any]] = []
        for ch, vec in zip(chunks, vectors, strict=True):
            rows.append(
                {
                    "id": f"{ch.content_hash[:16]}:{ch.chunk_index}",
                    "vector": vec.tolist(),
                    "text": ch.text,
                    "embed_text": ch.embed_text,
                    "doc_path": ch.doc_path,
                    "doc_title": ch.doc_title,
                    "doc_type": ch.doc_type,
                    "section": ch.section,
                    "page": int(ch.page),
                    "chunk_index": int(ch.chunk_index),
                    "n_tokens": int(ch.n_tokens),
                    "content_hash": ch.content_hash,
                    "mtime": float(ch.mtime),
                    "has_equation": bool(ch.has_equation),
                }
            )
        self.table.add(rows)
        return len(rows)

    def delete_doc(self, doc_path: str) -> None:
        """Drop every chunk of one document -- the re-index path."""
        if self.exists():
            self.table.delete(f"doc_path = '{_escape(doc_path)}'")

    def indexed_docs(self) -> dict[str, str]:
        """doc_path -> content_hash for everything currently indexed."""
        if not self.exists() or self.count() == 0:
            return {}
        tbl = self.table.to_arrow().select(["doc_path", "content_hash"])
        return dict(
            zip(
                tbl.column("doc_path").to_pylist(),
                tbl.column("content_hash").to_pylist(),
                strict=True,
            )
        )

    # --- indexes ---------------------------------------------------------

    def build_fts_index(self, replace: bool = True) -> None:
        """BM25 over `embed_text` (body + title + section breadcrumb).

        Uses LanceDB's native FTS rather than the tantivy-backed one: native
        supports incremental updates, where tantivy needs a full rebuild after
        every write.
        """
        if self.count() == 0:
            log.warning("no rows; skipping FTS index")
            return
        self.table.create_fts_index(FTS_COLUMN, replace=replace, use_tantivy=False)

    def build_vector_index(self, replace: bool = True) -> None:
        """ANN index over the vectors.

        Only worth building past a few thousand rows -- below that LanceDB's
        brute-force scan is faster than traversing an IVF-PQ structure, and
        exact search has no recall loss. This corpus may well stay under that
        line, which is fine.
        """
        n = self.count()
        if n < 5000:
            log.info("only %d rows; brute-force scan beats an ANN index", n)
            return
        self.table.create_index(
            metric="cosine",
            num_partitions=max(1, int(np.sqrt(n))),
            replace=replace,
        )

    def optimize(self) -> None:
        if self.exists():
            self.table.optimize()

    def drop(self) -> None:
        if self.exists():
            self._db.drop_table(TABLE_NAME)


__all__ = ["ChunkStore", "TABLE_NAME", "SCHEMA", "FTS_COLUMN"]

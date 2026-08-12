"""discover -> parse -> chunk -> embed -> LanceDB, incrementally.

PLAN.md §6: "Incremental re-index keyed on path + mtime + content hash." The
manifest here implements that as a three-stage ladder, cheapest test first:

  1. `mtime` and size unchanged  -> skip without reading the file
  2. content hash unchanged      -> skip, refresh the recorded mtime
  3. otherwise                   -> re-parse, re-chunk, re-embed that document

Stage 1 matters more than it looks. Hashing the whole corpus is ~645 MB of I/O
per run, and the point of an incremental indexer is that a no-op run is fast
enough to put on a file watcher later (PLAN.md §8).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from assistant.config import settings
from assistant.ingest.chunk import Chunk, chunk
from assistant.ingest.parse import parse
from assistant.ingest.sources import Discovered, discover
from assistant.retrieval.embed import embed_texts
from assistant.retrieval.store import ChunkStore

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    path          TEXT PRIMARY KEY,
    mtime         REAL NOT NULL,
    size          INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    text_hash     TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    n_chunks      INTEGER NOT NULL,
    parse_s       REAL NOT NULL,
    indexed_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_text_hash ON documents(text_hash);
"""


@dataclass
class IndexStats:
    scanned: int = 0
    skipped_unchanged: int = 0
    skipped_duplicate: int = 0
    skipped_empty: int = 0
    parsed: int = 0
    failed: int = 0
    removed: int = 0
    chunks_added: int = 0
    elapsed_s: float = 0.0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned {self.scanned} | parsed {self.parsed} | "
            f"unchanged {self.skipped_unchanged} | duplicate {self.skipped_duplicate} | "
            f"empty {self.skipped_empty} | failed {self.failed} | removed {self.removed} | "
            f"+{self.chunks_added} chunks in {self.elapsed_s:.1f}s"
        )


class Manifest:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.manifest_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def get(self, path: str) -> sqlite3.Row | None:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM documents WHERE path = ?", (path,))
        return cur.fetchone()

    def text_hash_owner(self, text_hash: str, exclude: str) -> str | None:
        cur = self._conn.execute(
            "SELECT path FROM documents WHERE text_hash = ? AND path != ? LIMIT 1",
            (text_hash, exclude),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def touch(self, path: str, mtime: float) -> None:
        self._conn.execute("UPDATE documents SET mtime = ? WHERE path = ?", (mtime, path))
        self._conn.commit()

    def record(self, **kw) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (path, mtime, size, content_hash, text_hash, doc_type,
                n_chunks, parse_s, indexed_at)
               VALUES (:path, :mtime, :size, :content_hash, :text_hash,
                       :doc_type, :n_chunks, :parse_s, :indexed_at)""",
            kw,
        )
        self._conn.commit()

    def forget(self, path: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        self._conn.commit()

    def all_paths(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT path FROM documents")}

    def close(self) -> None:
        self._conn.close()


def _text_hash(markdown: str) -> str:
    """Hash of normalised extracted text, for cross-file dedup.

    The physics textbook exists twice in `Books/` -- a 447 MB original and a
    47 MB recompression. Their bytes differ so `content_hash` sees two files,
    but the extracted text is near-identical and indexing both would double
    every retrieval hit from it. Whitespace is collapsed so trivial layout
    differences do not defeat the match.
    """
    normalised = " ".join(markdown.split())
    return sha256(normalised.encode("utf-8", errors="ignore")).hexdigest()


def _embed_chunks(chunks: list[Chunk], batch_size: int, device: str):
    """(n, 1024) float32. See `retrieval/embed.py` on device choice."""
    return embed_texts(
        [c.embed_text for c in chunks], batch_size=batch_size, device=device
    )


def build_index(
    tiers: tuple[str, ...] | None = None,
    *,
    device: str = "cpu",
    force: bool = False,
    limit: int | None = None,
    batch_size: int = 8,
    embed_device: str = "cpu",
    store: ChunkStore | None = None,
    items: Iterable[Discovered] | None = None,
) -> IndexStats:
    """Bring the index up to date for the requested tiers.

    Two separate devices, deliberately:

    * `device` is **docling's**. CPU is unusable here -- 203 s vs >40 min on a
      single paper -- so pass `"cuda"` and unload the served model first.
    * `embed_device` is **bge-m3's**. Defaults to CPU per design rule 5, but that
      rule is about query-time eviction, and batch ingestion is not query time.
      Phase 1 spent 66 of 75 minutes here on CPU; `"cuda"` is the right choice
      for a large tier. See `retrieval/embed.py`.
    """
    started = time.perf_counter()
    stats = IndexStats()
    store = store or ChunkStore()
    manifest = Manifest()

    discovered = list(items) if items is not None else discover(tiers)
    if limit:
        discovered = discovered[:limit]
    stats.scanned = len(discovered)

    seen_paths: set[str] = set()

    for i, item in enumerate(discovered, 1):
        path_str = str(item.path)
        seen_paths.add(path_str)

        try:
            st = item.path.stat()
        except OSError:
            stats.failed += 1
            stats.failures.append(f"stat failed: {path_str}")
            continue

        # Zero-byte placeholder files are common in half-scaffolded firmware
        # trees -- this corpus has 9. They are not failures and should not be
        # reported as such, or a clean run looks broken.
        if st.st_size == 0:
            stats.skipped_empty += 1
            continue

        row = manifest.get(path_str)

        # Stage 1 -- cheapest: mtime and size both unchanged.
        if row and not force and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
            stats.skipped_unchanged += 1
            continue

        log.info("[%d/%d] %s", i, len(discovered), item.path.name)
        doc = parse(item, device=device)
        if doc is None:
            stats.failed += 1
            stats.failures.append(f"parse failed: {path_str}")
            continue

        # Stage 2 -- content unchanged despite a touched mtime.
        if row and not force and row["content_hash"] == doc.content_hash:
            manifest.touch(path_str, st.st_mtime)
            stats.skipped_unchanged += 1
            continue

        # Stage 3 -- cross-file duplicate.
        th = _text_hash(doc.markdown)
        owner = manifest.text_hash_owner(th, exclude=path_str)
        if owner and not force:
            log.info("  duplicate of %s -- skipping", Path(owner).name)
            stats.skipped_duplicate += 1
            manifest.record(
                path=path_str,
                mtime=st.st_mtime,
                size=st.st_size,
                content_hash=doc.content_hash,
                text_hash=th,
                doc_type=doc.doc_type,
                n_chunks=0,
                parse_s=doc.parse_s,
                indexed_at=time.time(),
            )
            continue

        chunks = chunk(doc)
        if not chunks:
            stats.failed += 1
            stats.failures.append(f"no chunks: {path_str}")
            continue

        vectors = _embed_chunks(chunks, batch_size, embed_device)
        store.delete_doc(path_str)  # replace, never duplicate
        added = store.add(chunks, vectors)

        manifest.record(
            path=path_str,
            mtime=st.st_mtime,
            size=st.st_size,
            content_hash=doc.content_hash,
            text_hash=th,
            doc_type=doc.doc_type,
            n_chunks=added,
            parse_s=doc.parse_s,
            indexed_at=time.time(),
        )
        stats.parsed += 1
        stats.chunks_added += added

    # Documents that vanished from the corpus. Only prune when indexing the
    # full corpus -- a tier-limited run legitimately does not see the others.
    if items is None and tiers is None:
        for stale in manifest.all_paths() - seen_paths:
            store.delete_doc(stale)
            manifest.forget(stale)
            stats.removed += 1

    if stats.chunks_added or stats.removed:
        store.build_fts_index()
        store.build_vector_index()
        store.optimize()

    manifest.close()
    stats.elapsed_s = time.perf_counter() - started
    return stats


__all__ = ["build_index", "IndexStats", "Manifest"]

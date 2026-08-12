"""Documents in, markdown out. The expensive half of ingestion.

PLAN.md §6 is unambiguous about why this uses docling rather than PyMuPDF or
markitdown: generic extractors mangle equations into garbage (`Z 1 0 f (x)dx`),
and this corpus is EMG papers, a physics textbook and register-table datasheets.
An equation or a register table destroyed at parse time cannot be recovered by
any amount of retrieval cleverness downstream.

**Parsed markdown is cached to `data/parsed/`, keyed by content hash.** This is
the single most important design decision in the file. docling on the 447 MB
textbook is hours; the chunker will be tuned a dozen times during phase 1, and
re-parsing on every tweak would make that impossible. Parse once, chunk often.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from assistant.config import settings
from assistant.ingest.sources import Discovered

log = logging.getLogger(__name__)

# docling handles these; anything else is read as text.
DOCLING_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".html", ".epub"})

# Bump to invalidate every cached parse. v2: stopped escaping underscores and
# HTML, which was corrupting LaTeX subscripts.
CACHE_VERSION = 2


@dataclass(slots=True)
class ParsedDoc:
    path: Path
    doc_type: str
    title: str
    markdown: str
    content_hash: str
    parser: str  # docling | text
    n_pages: int = 0
    parse_s: float = 0.0
    mtime: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.markdown.strip()) < 32


def content_hash(path: Path) -> str:
    """SHA-256 of file bytes, streamed.

    Doubles as the dedup key: the physics textbook exists twice in `Books/`
    (447 MB and a 47 MB recompression). Those differ byte-wise so the hash will
    not catch them -- see `index.py`, which dedups on extracted text instead.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_path(digest: str) -> Path:
    return settings.parsed_dir / f"{digest[:2]}" / f"{digest}.json"


def _load_cached(digest: str) -> dict | None:
    p = _cache_path(digest)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if data.get("cache_version") == CACHE_VERSION else None


def _store_cached(digest: str, payload: dict) -> None:
    p = _cache_path(digest)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload["cache_version"] = CACHE_VERSION
    p.write_text(json.dumps(payload))


# --- docling -------------------------------------------------------------

_CONVERTERS: dict[str, object] = {}


def _converter(doc_type: str, device: str = "auto"):
    """One converter per profile, built lazily and reused.

    Profiles differ because the corpus does. Formula enrichment runs a model per
    detected equation -- essential for papers and the textbook, pure cost for a
    datasheet whose value is in its register tables. Table structure is the
    mirror image.
    """
    if doc_type in _CONVERTERS:
        return _CONVERTERS[doc_type]

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    equations = doc_type in {"paper", "textbook"}
    tables = doc_type in {"paper", "textbook", "datasheet"}

    opts = PdfPipelineOptions()
    opts.do_formula_enrichment = equations
    opts.do_table_structure = tables
    opts.do_code_enrichment = False
    opts.do_picture_classification = False
    opts.do_picture_description = False
    # Scanned pages are rare here and OCR is the slowest stage by far. The
    # Chinese admin PDFs are the one place it may be needed -- flip it for that
    # profile if their extracted text comes back empty.
    opts.do_ocr = False
    opts.document_timeout = 3 * 60 * 60  # the textbook is genuinely hours

    # sm_75. FA2 needs Ampere or newer -- enabling it here compiles and then
    # fails at runtime, exactly as TRAINING_PLAN.md §1 warns for the training
    # stack. Same trap, different library.
    opts.accelerator_options = AcceleratorOptions(
        device=device,
        cuda_use_flash_attention2=False,
    )

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    _CONVERTERS[doc_type] = conv
    return conv


# Emitted at every page boundary so the chunker can attribute a page number to
# each chunk (PLAN.md §6 lists page in the required metadata). Nothing else in
# the pipeline should ever emit this string.
PAGE_BREAK = "<!-- docling-page-break -->"


def _parse_with_docling(path: Path, doc_type: str, device: str) -> tuple[str, int, str]:
    conv = _converter(doc_type, device)
    result = conv.convert(str(path))
    doc = result.document
    markdown = doc.export_to_markdown(
        page_break_placeholder=PAGE_BREAK,
        # Both default to True and both corrupt LaTeX. `escape_underscores`
        # turns every subscript `x_1` into `x\_1`, and this corpus is EMG and
        # physics papers where subscripts are in nearly every equation.
        # PLAN.md §6: equation LaTeX is never stripped -- silently escaping it
        # is the same failure wearing a different hat.
        escape_underscores=False,
        escape_html=False,
    )
    n_pages = len(getattr(doc, "pages", {}) or {})
    title = (getattr(doc, "name", "") or "").strip()
    return markdown, n_pages, title


# --- text ----------------------------------------------------------------


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except OSError:
            return ""
    return ""


def _derive_title(markdown: str, path: Path, docling_title: str = "") -> str:
    """First markdown heading, else docling's name, else the filename."""
    for line in markdown.splitlines()[:40]:
        line = line.strip()
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text:
                return text[:200]
    if docling_title and docling_title.lower() not in {"document", path.stem.lower()}:
        return docling_title[:200]
    return path.stem


# --- public --------------------------------------------------------------


def parse(item: Discovered, *, device: str = "auto", use_cache: bool = True) -> ParsedDoc | None:
    """Parse one file to markdown. Returns None if it yielded nothing usable.

    `device` is passed to docling. It defaults to `auto`, which will take the
    GPU -- fine for an offline batch, but note the served 27B holds ~18 GB of the
    card under `OLLAMA_KEEP_ALIVE=30m`. Either pass `device="cpu"` or stop the
    model first; see `scripts/phase1.md`.
    """
    path = item.path
    try:
        digest = content_hash(path)
        mtime = path.stat().st_mtime
    except OSError as exc:
        log.warning("unreadable %s: %s", path, exc)
        return None

    if use_cache and (cached := _load_cached(digest)):
        return ParsedDoc(
            path=path,
            doc_type=item.doc_type,
            title=cached["title"],
            markdown=cached["markdown"],
            content_hash=digest,
            parser=cached["parser"],
            n_pages=cached.get("n_pages", 0),
            parse_s=0.0,
            mtime=mtime,
            meta={"cached": True},
        )

    started = time.perf_counter()
    suffix = path.suffix.lower()
    docling_title = ""

    try:
        if suffix in DOCLING_SUFFIXES:
            markdown, n_pages, docling_title = _parse_with_docling(path, item.doc_type, device)
            parser = "docling"
        else:
            markdown, n_pages, parser = _read_text(path), 0, "text"
    except Exception as exc:  # docling raises a wide variety; none should abort a batch
        log.warning("parse failed %s: %s", path, exc)
        return None

    elapsed = time.perf_counter() - started
    title = _derive_title(markdown, path, docling_title)

    doc = ParsedDoc(
        path=path,
        doc_type=item.doc_type,
        title=title,
        markdown=markdown,
        content_hash=digest,
        parser=parser,
        n_pages=n_pages,
        parse_s=elapsed,
        mtime=mtime,
    )
    if doc.is_empty:
        # Usually a scanned PDF with OCR off, which is the documented tradeoff
        # above rather than a bug. Worth logging loudly so it is not silent.
        log.warning("empty after parse (scanned? try do_ocr=True): %s", path)
        return None

    if use_cache:
        _store_cached(
            digest,
            {
                "title": title,
                "markdown": markdown,
                "parser": parser,
                "n_pages": n_pages,
                "source_path": str(path),
            },
        )
    return doc


__all__ = ["ParsedDoc", "parse", "content_hash"]

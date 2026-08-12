"""Markdown in, retrievable chunks out.

Three rules from PLAN.md §6, all of which this file exists to enforce:

1. **Chunk on section headings**, so an equation travels with the prose that
   explains it.
2. **An equation never becomes its own chunk.** A bare formula has almost no
   embedding signal -- `$Z = \\sum_i e^{-\\beta E_i}$` retrieves nothing useful,
   while the same formula under "Canonical Ensemble" with a paragraph of
   explanation retrieves well.
3. **LaTeX is never stripped.** The model reads it fine.

Code chunks differently -- on function and class boundaries, with the file path
and imports repeated in each chunk header, because a function body alone rarely
says what project it belongs to.

Token counts use bge-m3's own tokenizer. Approximating with `len(text)/4` is off
by 2-3x on LaTeX and on Chinese, both of which this corpus contains, and the
error lands exactly where chunks would silently exceed the embedding window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from assistant.ingest.parse import PAGE_BREAK, ParsedDoc

# PLAN.md §6: 512-1024 tokens with ~100 overlap.
TARGET_TOKENS = 768
MAX_TOKENS = 1024
MIN_TOKENS = 64  # below this a chunk is merged forward rather than emitted
OVERLAP_TOKENS = 100

# Headings are the preferred split point, but breaking at *every* heading
# fragments heading-dense documents badly -- the project READMEs here have a
# heading every few lines, which produced a median chunk of 179 tokens against a
# 768 target. Only treat a heading as a boundary once there is a real chunk's
# worth of content behind it; otherwise keep accumulating through it.
HEADING_BREAK_MIN = TARGET_TOKENS // 2

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Display math in the forms docling emits, plus common LaTeX environments.
DISPLAY_MATH_RE = re.compile(
    r"^\s*(\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{(equation|align|gather|eqnarray|multline)\*?\})",
    re.DOTALL,
)
INLINE_MATH_RE = re.compile(r"\$[^$\n]{2,}?\$")

# Python/C-family definition starts, used for code chunking.
DEF_RE = re.compile(
    r"^(?:\s*)(?:"
    r"(?:async\s+)?def\s+\w+|"  # python
    r"class\s+\w+|"
    r"(?:static\s+|inline\s+|extern\s+)*"  # C/C++
    r"[A-Za-z_][\w\s\*&:<>,]*\s+[A-Za-z_]\w*\s*\([^;]*\)\s*\{?\s*$"
    r")"
)
IMPORT_RE = re.compile(r"^\s*(?:#include\s*[<\"]|import\s+|from\s+\w+\s+import|using\s+)")


@dataclass(slots=True)
class Chunk:
    """One retrievable unit. Fields map 1:1 onto the LanceDB schema."""

    text: str
    doc_path: str
    doc_title: str
    doc_type: str
    section: str  # heading breadcrumb, e.g. "Methods > EMG Preprocessing"
    page: int
    chunk_index: int
    n_tokens: int
    content_hash: str
    mtime: float
    has_equation: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        The heading breadcrumb and title are prepended because a chunk from the
        middle of a section otherwise carries no indication of what it is about
        -- PLAN.md §6's point about equations travelling with their explanation
        applies to plain prose too.
        """
        header = f"{self.doc_title} | {self.section}" if self.section else self.doc_title
        return f"{header}\n\n{self.text}"


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("BAAI/bge-m3")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


# --- prose ---------------------------------------------------------------


@dataclass(slots=True)
class _Block:
    """A markdown block: a paragraph, heading, table, fence or equation."""

    text: str
    page: int
    section: str
    is_heading: bool = False
    is_math: bool = False
    n_tokens: int = 0


def _split_blocks(markdown: str) -> list[_Block]:
    """Split into blocks, tracking heading breadcrumb and page number.

    Fenced code and tables are held together -- splitting a markdown table
    mid-row makes both halves useless, and datasheet register tables are among
    the most valuable things in this corpus.
    """
    blocks: list[_Block] = []
    heading_stack: list[tuple[int, str]] = []
    page = 1
    buf: list[str] = []
    in_fence = False
    fence_marker = ""

    def breadcrumb() -> str:
        return " > ".join(h for _, h in heading_stack)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        blocks.append(
            _Block(
                text=text,
                page=page,
                section=breadcrumb(),
                is_math=bool(DISPLAY_MATH_RE.match(text)),
            )
        )

    for raw in markdown.splitlines():
        line = raw.rstrip()

        if PAGE_BREAK in line:
            flush()
            page += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            if not in_fence:
                flush()
                in_fence, fence_marker = True, fence.group(1)
                buf.append(line)
            elif line.strip().startswith(fence_marker):
                buf.append(line)
                in_fence = False
                flush()
            else:
                buf.append(line)
            continue

        if in_fence:
            buf.append(line)
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level, text = len(heading.group(1)), heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            blocks.append(
                _Block(
                    text=line,
                    page=page,
                    section=breadcrumb(),
                    is_heading=True,
                )
            )
            continue

        if not line.strip():
            flush()
            continue

        buf.append(line)

    flush()

    for b in blocks:
        b.n_tokens = count_tokens(b.text)
    return blocks


def _chunk_prose(doc: ParsedDoc) -> list[Chunk]:
    blocks = _split_blocks(doc.markdown)
    if not blocks:
        return []

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0

    def emit() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        # Drop chunks that are only headings -- no content to retrieve.
        if all(b.is_heading for b in current):
            current, current_tokens = [], 0
            return
        body = "\n\n".join(b.text for b in current).strip()
        if not body:
            current, current_tokens = [], 0
            return
        first = current[0]
        chunks.append(
            Chunk(
                text=body,
                doc_path=str(doc.path),
                doc_title=doc.title,
                doc_type=doc.doc_type,
                section=first.section,
                page=first.page,
                chunk_index=len(chunks),
                n_tokens=current_tokens,
                content_hash=doc.content_hash,
                mtime=doc.mtime,
                has_equation=any(b.is_math for b in current)
                or bool(INLINE_MATH_RE.search(body)),
            )
        )
        # Carry the tail forward as overlap, so a sentence split across a
        # boundary is still retrievable from the following chunk.
        overlap: list[_Block] = []
        total = 0
        for b in reversed(current):
            if total >= OVERLAP_TOKENS or b.is_heading:
                break
            overlap.insert(0, b)
            total += b.n_tokens
        current = overlap
        current_tokens = total

    for i, block in enumerate(blocks):
        # Rule 2: an equation is never allowed to start a chunk on its own. If
        # the buffer is empty and this is display math, pull the previous block
        # back in so the formula keeps its introduction.
        if block.is_math and not current and chunks and i > 0:
            prev = blocks[i - 1]
            if not prev.is_heading:
                current, current_tokens = [prev], prev.n_tokens

        # A heading is a natural boundary -- break before it if we already have
        # a reasonable amount of content.
        if block.is_heading and current_tokens >= HEADING_BREAK_MIN:
            emit()

        # A single block over the limit (a huge table, a long fence) becomes its
        # own chunk rather than being split mid-structure.
        if block.n_tokens > MAX_TOKENS:
            emit()
            current, current_tokens = [block], block.n_tokens
            emit()
            continue

        if current_tokens + block.n_tokens > MAX_TOKENS:
            emit()

        current.append(block)
        current_tokens += block.n_tokens

        if current_tokens >= TARGET_TOKENS and not block.is_math:
            emit()

    emit()

    # Final pass for rule 2: anything that survived as pure math gets merged
    # backwards into its neighbour.
    return _merge_orphan_math(chunks)


def _merge_orphan_math(chunks: list[Chunk]) -> list[Chunk]:
    """Fold any chunk that is nothing but an equation into the previous one."""
    out: list[Chunk] = []
    for ch in chunks:
        stripped = ch.text.strip()
        only_math = bool(DISPLAY_MATH_RE.match(stripped)) and len(stripped) < 400
        if only_math and out and out[-1].n_tokens + ch.n_tokens <= MAX_TOKENS + 256:
            prev = out[-1]
            prev.text = f"{prev.text}\n\n{ch.text}"
            prev.n_tokens += ch.n_tokens
            prev.has_equation = True
            continue
        out.append(ch)
    for i, ch in enumerate(out):
        ch.chunk_index = i
    return out


# --- code ----------------------------------------------------------------


def _chunk_code(doc: ParsedDoc) -> list[Chunk]:
    """Split on definition boundaries, repeating file path and imports.

    PLAN.md §6. The header repetition is deliberate duplication: a chunk holding
    `static void adc_init(void) {...}` is unretrievable without the context that
    it lives in `FES_Board/firmware/drivers/`, and that context is exactly what a
    question like "how do I initialise the ADC on the FES board" matches on.
    """
    lines = doc.markdown.splitlines()
    if not lines:
        return []

    imports = [ln for ln in lines[:80] if IMPORT_RE.match(ln)][:12]
    rel = doc.path.name
    header = f"# file: {doc.path}\n" + ("\n".join(imports) + "\n" if imports else "")
    header_tokens = count_tokens(header)

    # Definition start lines become the split points.
    starts = [i for i, ln in enumerate(lines) if DEF_RE.match(ln)]
    if not starts:
        spans = [(0, len(lines))]
    else:
        bounds = [0, *starts, len(lines)]
        bounds = sorted(set(bounds))
        spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    first_line = 0

    def emit() -> None:
        nonlocal buf, buf_tokens, first_line
        body = "\n".join(buf).strip()
        if not body:
            buf, buf_tokens = [], 0
            return
        chunks.append(
            Chunk(
                text=header + body,
                doc_path=str(doc.path),
                doc_title=doc.title or rel,
                doc_type="code",
                section=f"lines {first_line + 1}-{first_line + len(buf)}",
                page=0,
                chunk_index=len(chunks),
                n_tokens=header_tokens + buf_tokens,
                content_hash=doc.content_hash,
                mtime=doc.mtime,
            )
        )
        buf, buf_tokens = [], 0

    for start, end in spans:
        segment = lines[start:end]
        seg_text = "\n".join(segment)
        seg_tokens = count_tokens(seg_text)

        if seg_tokens > MAX_TOKENS:
            emit()
            # A single oversized function: split on blank lines as a last resort.
            first_line = start
            for ln in segment:
                ln_tokens = count_tokens(ln) if ln.strip() else 1
                if buf_tokens + ln_tokens > MAX_TOKENS - header_tokens:
                    emit()
                    first_line = start
                buf.append(ln)
                buf_tokens += ln_tokens
            emit()
            continue

        if buf_tokens + seg_tokens + header_tokens > MAX_TOKENS:
            emit()
        if not buf:
            first_line = start
        buf.extend(segment)
        buf_tokens += seg_tokens

    emit()
    return chunks


# --- public --------------------------------------------------------------


def chunk(doc: ParsedDoc) -> list[Chunk]:
    """Chunk one parsed document according to its type."""
    if doc.doc_type == "code":
        chunks = _chunk_code(doc)
    else:
        chunks = _chunk_prose(doc)
    return [c for c in chunks if c.n_tokens >= 8]


__all__ = ["Chunk", "chunk", "count_tokens", "TARGET_TOKENS", "MAX_TOKENS"]

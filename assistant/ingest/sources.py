"""What goes in the index, and what emphatically does not.

CLAUDE.md safety rules: **allowlist, never blocklist**, and never all of
`/home/cgu`. Nothing is indexed unless a `Source` below names it.

The exclusions are not paranoia. A naive walk of `~/Documents` finds ~36,000
source files, of which ~27,000 are vendored SDK headers, virtualenvs and Zephyr
build artifacts. Indexing those would bury the ~150 files actually written by
hand and make every retrieval worse -- the corpus is small enough that precision
matters more than recall.

Tiers exist because docling cost is wildly uneven (PLAN.md §6: it is slow and
GPU-hungry, run it as an offline batch). `fast` is minutes; `books` is hours for
a single 447 MB textbook. Ingest `fast` first and confirm retrieval works before
spending a night on the rest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
DOCS = HOME / "Documents"

# Document formats. Code is handled separately -- it chunks on function/class
# boundaries rather than headings (PLAN.md §6).
# `.txt` is deliberately absent. Every one of the 170 `.txt` files in this corpus
# is either build config (`CMakeLists.txt`, `requirements.txt`) or a raw
# measurement dump -- numeric data with no embedding signal that would flood the
# index. Add it back only alongside a source that genuinely holds prose in .txt.
PROSE_SUFFIXES = (".pdf", ".md", ".docx", ".pptx", ".html", ".epub")
CODE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".rs", ".js", ".ts", ".sh")

# Pruned during the walk, at any depth. Directory names, not paths.
EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",  # Zephyr west build output -- thousands of generated headers
        "twister-out",
        "Drivers",  # vendored STM32 HAL/CMSIS, not hand-written
        "Middlewares",
        "cmake",
        "zephyr",
        "modules",
        "site-packages",
        "dist",
        ".cache",
        # Raw measurement data. `tSCS_data` holds 145 numeric dumps in
        # per-participant directories -- see the note on SOURCES below.
        "tSCS_data",
        "datasets",
        "recordings",
    }
)

EXCLUDE_GLOBS = (
    "*.min.js",
    "*.lock",
    "*-lock.json",
    "*.pyc",
    "*.o",
    "*.elf",
    "*.map",
)

# Files above this are almost always generated, vendored or a binary blob that
# slipped past the suffix filter. The physics textbook is the deliberate
# exception and is allowlisted by living in its own tier.
MAX_PROSE_MB = 600.0
MAX_CODE_KB = 400.0


@dataclass(frozen=True, slots=True)
class Source:
    """One allowlisted root.

    `doc_type` lands in chunk metadata and is worth getting right -- PLAN.md §6
    notes that many real questions are scoped by source or type, and a metadata
    filter answers those far better than an embedding does.
    """

    root: Path
    doc_type: str  # paper | textbook | datasheet | notes | code | admin
    tier: str  # fast | datasheets | books
    suffixes: tuple[str, ...] = PROSE_SUFFIXES
    exclude_subdirs: tuple[str, ...] = ()
    note: str = ""

    def enabled(self) -> bool:
        return self.root.exists()


# --- The allowlist -------------------------------------------------------
#
# Surveyed 2026-08-04. Edit this list to change what Eve knows about; it is
# the single place corpus scope is decided.

SOURCES: tuple[Source, ...] = (
    # -- tier: fast ------------------------------------------------------
    # Biomedical signal-processing research -- the master's work. Equation- and
    # figure-heavy, and the highest-value material in the corpus because it is
    # the part the base model has never seen.
    Source(
        DOCS / "Master Research Paper",
        doc_type="paper",
        tier="fast",
        note="EMG, freezing-of-gait, ADS1298 front-end papers",
    ),
    Source(
        DOCS / "Seizure_detection",
        doc_type="notes",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
        exclude_subdirs=("datasets",),
        note="rsos paper, note.md, API + UI_App source",
    ),
    Source(
        DOCS / "VBT  Research",  # two spaces -- real directory name
        doc_type="paper",
        tier="fast",
        note="quadrotor kinematics/dynamics, state-space",
    ),
    Source(
        DOCS / "Master Research Information",
        doc_type="admin",
        tier="fast",
        note="certificates, enrolment, NSTC grant listing. Chinese -- bge-m3 is multilingual",
    ),
    # Embedded projects: hand-written firmware plus the READMEs that explain it.
    Source(
        DOCS / "FES_Board",
        doc_type="notes",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
        exclude_subdirs=("datasheet",),  # own tier -- large and table-heavy
        note="ADS1298 pinout, WT02C40C reference, firmware + software",
    ),
    Source(
        DOCS / "STM32_Driver",
        doc_type="code",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
        exclude_subdirs=("Documents",),  # reference manuals -> datasheets tier
        note="bare-metal register drivers",
    ),
    Source(
        DOCS / "ZephyrPractise",
        doc_type="code",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
        note="RTOS primitives: threads, mutex, semaphore, queues, DMA",
    ),
    # NOTE: `tSCS_data/` is excluded globally. It holds 145 raw measurement
    # files in directories named after individual research participants. That is
    # identifiable human-subject data, it is numeric rather than prose so it
    # would retrieve badly anyway, and embedding it would copy those names into
    # the index and every trace log downstream. Opt in deliberately if you ever
    # want it -- do not let it arrive by accident.
    Source(
        DOCS / "ACQ_Read",
        doc_type="code",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
    ),
    Source(
        DOCS / "VBT_project",
        doc_type="code",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
    ),
    Source(
        DOCS / "Simulation" / "RLC_circuit",
        doc_type="notes",
        tier="fast",
        suffixes=PROSE_SUFFIXES + CODE_SUFFIXES,
    ),
    # -- tier: datasheets ------------------------------------------------
    # Table-heavy and long (RM0090 alone is ~1700 pages). docling's table
    # understanding is the reason it is worth the parse time: a register table
    # mangled into prose is unretrievable, and register lookups are exactly what
    # this material gets asked for.
    Source(
        DOCS / "FES_Board" / "datasheet",
        doc_type="datasheet",
        tier="datasheets",
        note="ADS1298 (sbau171d), WT02C40C, tps7a20, bq24074",
    ),
    Source(
        DOCS / "STM32_Driver" / "Documents",
        doc_type="datasheet",
        tier="datasheets",
        note="RM0090, RM0390, ARM DUI0553, bare-metal cookbook",
    ),
    # -- tier: books -----------------------------------------------------
    # Hours, not minutes. Serway is 447 MB and duplicated -- the indexer's
    # content hash drops the second copy automatically.
    #
    # Worth questioning before you run it: a general physics textbook is largely
    # material the 27B already knows, so it buys less than the papers and
    # datasheets did. Run it if you want citations to *your* edition and page
    # numbers; skip it if you only want answers.
    Source(
        DOCS / "Books",
        doc_type="textbook",
        tier="books",
        note="Serway 10e (447 MB, duplicated), Fuzzy Logic for Engineers",
    ),
)

TIERS: tuple[str, ...] = ("fast", "datasheets", "books")


@dataclass(slots=True)
class Discovered:
    path: Path
    source: Source
    is_code: bool = field(default=False)

    @property
    def doc_type(self) -> str:
        """Classify the file, not just the source it came from.

        Both directions matter: source code inside a prose source is `code`, and
        a README inside a code source is `notes`. Getting the second one wrong
        types 15 hand-written READMEs as source and sends them to the
        function-boundary splitter, which would shred them.
        """
        if self.is_code:
            return "code"
        if self.source.doc_type == "code":
            return "notes"
        return self.source.doc_type


def _pruned_walk(root: Path, exclude_subdirs: tuple[str, ...]) -> Iterator[Path]:
    """Walk `root`, pruning excluded directories instead of filtering after.

    Pruning matters: `AI_Pattern_Recognition/classwork` holds 20,545 files, and
    descending into it to reject each one wastes minutes per run.
    """
    skip = EXCLUDE_DIRS | {d.lower() for d in exclude_subdirs}
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in EXCLUDE_DIRS or entry.name.lower() in skip:
                    continue
                if entry.name.startswith("."):
                    continue
                stack.append(entry)
            elif entry.is_file():
                yield entry


def _too_big(path: Path, is_code: bool) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return True
    limit = MAX_CODE_KB * 1024 if is_code else MAX_PROSE_MB * 1024 * 1024
    return size > limit


def discover(tiers: tuple[str, ...] | None = None) -> list[Discovered]:
    """Every allowlisted file in the requested tiers, deduplicated by path.

    Sources overlap deliberately (`FES_Board` excludes `datasheet/`, which is its
    own source), so a path seen twice keeps the first, most specific match.
    """
    wanted = set(tiers or TIERS)
    seen: set[Path] = set()
    out: list[Discovered] = []

    for src in SOURCES:
        if src.tier not in wanted or not src.enabled():
            continue
        for path in _pruned_walk(src.root, src.exclude_subdirs):
            if path in seen:
                continue
            suffix = path.suffix.lower()
            if suffix not in src.suffixes:
                continue
            if any(path.match(g) for g in EXCLUDE_GLOBS):
                continue
            is_code = suffix in CODE_SUFFIXES
            if _too_big(path, is_code):
                continue
            seen.add(path)
            out.append(Discovered(path=path, source=src, is_code=is_code))

    return out


def summarise(items: list[Discovered]) -> str:
    """Human-readable breakdown -- what `eve index` prints before it starts."""
    from collections import Counter

    by_type = Counter(i.doc_type for i in items)
    by_tier = Counter(i.source.tier for i in items)
    total_mb = sum(i.path.stat().st_size for i in items) / 2**20

    lines = [f"{len(items)} files, {total_mb:.1f} MB"]
    lines.append("  by tier: " + ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))
    lines.append("  by type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return "\n".join(lines)


__all__ = ["Source", "Discovered", "SOURCES", "TIERS", "discover", "summarise"]

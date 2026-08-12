"""`search_docs` — the tool that makes phase 1 useful.

PLAN.md §7.2. Thin on purpose: all the retrieval intelligence lives in
`retrieval/search.py`, and this module's only job is presenting hits to the model
in a form it can cite from.

The output format matters more than it looks. Every passage is prefixed with its
source, section and page so the model can attribute claims, and the instruction
to cite is in the agent's system prompt rather than here. A model handed
unattributed text will produce unattributed answers, and on a corpus of the
user's own research notes that is the difference between a useful answer and one
that cannot be checked.
"""

from __future__ import annotations

import logging

from pydantic_ai import ModelRetry

from assistant.retrieval.search import Hit, Retriever

log = logging.getLogger(__name__)

MAX_CHARS_PER_HIT = 1100
VALID_DOC_TYPES = {"paper", "textbook", "datasheet", "notes", "code", "admin"}


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "No matching passages found."
    blocks = [f"[{i}] {h.render(MAX_CHARS_PER_HIT)}" for i, h in enumerate(hits, 1)]
    return "\n\n---\n\n".join(blocks)


def make_search_docs(retriever: Retriever | None = None):
    """Build the `search_docs` tool bound to a retriever.

    A factory rather than a bare function so the CLI, the eval harness and later
    the voice loop can share one warm retriever — loading bge-m3 costs seconds
    and there is no reason to pay it per call.
    """
    retriever = retriever or Retriever()

    def search_docs(
        query: str,
        doc_type: str | None = None,
        source: str | None = None,
        k: int = 5,
    ) -> tuple[str, list[Hit]]:
        """Search the user's personal documents: research papers, datasheets,
        project notes, and source code.

        Use this whenever the question refers to the user's own work, projects,
        hardware, papers or notes. Prefer specific technical terms — part
        numbers, register names, function names and acronyms all match well.

        Args:
            query: What to search for.
            doc_type: Optional filter — one of paper, textbook, datasheet,
                notes, code, admin.
            source: Optional path fragment to restrict the search, e.g.
                "FES_Board" or "Seizure_detection".
            k: How many passages to return (1-10).
        """
        query = (query or "").strip()
        if not query:
            raise ModelRetry("`query` was empty. Say what to search for.")

        if doc_type and doc_type not in VALID_DOC_TYPES:
            raise ModelRetry(
                f"{doc_type!r} is not a valid doc_type. "
                f"Use one of: {', '.join(sorted(VALID_DOC_TYPES))}."
            )

        k = max(1, min(int(k or 5), 10))
        hits = retriever.search(query, k=k, doc_type=doc_type, source=source)

        if not hits and (doc_type or source):
            # Distinguish "nothing there" from "your filter excluded it" -- the
            # model can usefully retry without the filter, and over-narrow
            # filters are a common small-model failure.
            broadened = retriever.search(query, k=k)
            if broadened:
                raise ModelRetry(
                    f"No results with doc_type={doc_type!r} source={source!r}, "
                    f"but {len(broadened)} without those filters. Retry unfiltered."
                )

        # Hits are returned alongside the rendered text so the caller can log
        # citations without paying for a second identical search.
        return format_hits(hits), hits

    return search_docs


__all__ = ["make_search_docs", "format_hits"]

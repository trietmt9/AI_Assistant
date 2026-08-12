"""The Evelyn agent — `pydantic-ai` loop over the phase-1 retrieval layer.

PLAN.md §11 phase 2: prove tool calling before adding voice. That is the whole
purpose of this file, and the reason it stays small — every tool here is one the
phase-0 bake-off already showed the model calls correctly.

**Four tools, deliberately.** §9 budgets ~8 per prompt and warns that small
models degrade past that. Phase 2 uses four; the domain routing that §9 describes
only becomes necessary in phase 7, and building it now would be solving a problem
that does not exist yet.

Every turn is logged (see `tracing.py`). That is not instrumentation — it is the
phase-9 training set being collected from day one.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic_ai import Agent, RunContext

from assistant.config import settings
from assistant.llm.provider import build_model, model_settings
from assistant.retrieval.search import Retriever
from assistant.tools import compute
from assistant.tools.knowledge import make_search_docs
from assistant.tracing import ToolCallRecord

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Evelyn, {user}'s local assistant. You run entirely on their own hardware.

Your knowledge of their world comes from `search_docs`, which searches their
research papers, hardware datasheets, project notes and source code. Their work
is biomedical signal processing (EMG, freezing-of-gait, seizure detection) and
embedded systems (STM32, Zephyr, ADS1298 analog front ends).

Rules you follow without exception:

1. **Search before answering anything about their work.** Your training data does
   not contain their papers, their board, or their notes. If a question touches
   their projects, hardware, measurements or writing, call `search_docs` first.
2. **Cite what you used.** Refer to sources by name and page, as they appear in
   the search results — "per ADS1298_STM32_pinout.md" or "RM0090 p.214". Never
   present a retrieved fact without saying where it came from.
3. **Never do arithmetic yourself.** Route every calculation, symbolic
   manipulation and unit conversion through `calc`. You are good at setting up
   problems and unreliable at evaluating them. Use `run_python` for numerics that
   `calc` cannot express.
4. **Say when you do not know.** If retrieval returns nothing relevant, say so
   and say what you searched for. A wrong answer about their own hardware is
   worse than no answer.
5. Be concise. Answer the question asked. This runs in a terminal today and will
   be spoken aloud from phase 3, so avoid long preambles and heavy formatting.
"""


@dataclass
class Deps:
    """Run-scoped state. Tools reach this through `RunContext`."""

    retriever: Retriever
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)


def _context_now() -> str:
    """Time, host and session context.

    PLAN.md §7.1 calls this "cheap and disproportionately valuable" and says to
    inject it into every system prompt rather than making the model ask. Knowing
    it is Tuesday 23:40 measurably improves answers for free.
    """
    now = datetime.now().astimezone()
    return (
        f"Current time: {now:%A %d %B %Y, %H:%M %Z}. "
        f"Host: {platform.node()}. Model: {settings.primary_model}."
    )


def build_agent(retriever: Retriever | None = None, user: str = "the user") -> Agent[Deps, str]:
    """Construct the phase-2 agent.

    The model comes from `llm/provider.py` rather than being named here, so the
    phase-5 move to the Jetson stays a config change (design rule 8).
    """
    retriever = retriever or Retriever()
    search_docs_fn = make_search_docs(retriever)

    agent: Agent[Deps, str] = Agent(
        build_model(),
        name="evelyn",
        deps_type=Deps,
        model_settings=model_settings(),
        retries={"tools": 2},
    )

    @agent.instructions
    def _instructions(ctx: RunContext[Deps]) -> str:
        return SYSTEM_PROMPT.format(user=user) + "\n\n" + _context_now()

    @agent.tool
    def search_docs(
        ctx: RunContext[Deps],
        query: str,
        doc_type: str | None = None,
        source: str | None = None,
        k: int = 5,
    ) -> str:
        """Search the user's personal documents: research papers, datasheets,
        project notes, and source code.

        Use this whenever the question refers to the user's own work, projects,
        hardware, papers or notes. Prefer specific technical terms — part
        numbers, register names, function names and acronyms all match well.

        Args:
            query: What to search for.
            doc_type: Optional filter — paper, textbook, datasheet, notes, code
                or admin.
            source: Optional path fragment, e.g. "FES_Board".
            k: How many passages to return (1-10).
        """
        started = time.perf_counter()
        args = {"query": query, "doc_type": doc_type, "source": source, "k": k}
        try:
            out, hits = search_docs_fn(query, doc_type=doc_type, source=source, k=k)
        except Exception as exc:
            ctx.deps.tool_calls.append(
                ToolCallRecord(
                    "search_docs", args, None, ok=False, error=str(exc),
                    elapsed_s=time.perf_counter() - started,
                )
            )
            raise
        # Citations are logged separately from the answer -- PLAN.md §10 scores
        # retrieval and generation apart, and that is only possible if the trace
        # keeps what was retrieved alongside what was said about it.
        ctx.deps.retrieval_log.append(
            {"query": query, "citations": [h.citation for h in hits],
             "scores": [round(h.score, 4) for h in hits]}
        )
        ctx.deps.tool_calls.append(
            ToolCallRecord("search_docs", args, out,
                           elapsed_s=time.perf_counter() - started)
        )
        return out

    @agent.tool
    def calc(ctx: RunContext[Deps], expression: str) -> str:
        """Evaluate or manipulate a mathematical expression with SymPy.

        Use for ALL arithmetic, algebra, calculus and unit conversion. Accepts
        SymPy syntax: `integrate(x**2*sin(x), x)`, `solve(x**2-5*x+6, x)`,
        `diff(exp(2*x)*cos(x), x)`, `3.5 eV to J`.

        Args:
            expression: The expression to evaluate.
        """
        started = time.perf_counter()
        try:
            out = compute.calc(expression)
        except Exception as exc:
            ctx.deps.tool_calls.append(
                ToolCallRecord("calc", {"expression": expression}, None, ok=False,
                               error=str(exc), elapsed_s=time.perf_counter() - started)
            )
            raise
        ctx.deps.tool_calls.append(
            ToolCallRecord("calc", {"expression": expression}, out,
                           elapsed_s=time.perf_counter() - started)
        )
        return out

    @agent.tool
    def run_python(ctx: RunContext[Deps], code: str) -> str:
        """Execute Python in a sandboxed subprocess and return its stdout.

        For numerics, signal processing and plotting that `calc` cannot express
        symbolically. NumPy is available as `np`. You must `print()` results.

        Args:
            code: Python source to execute.
        """
        started = time.perf_counter()
        try:
            out = compute.run_python(code)
        except Exception as exc:
            ctx.deps.tool_calls.append(
                ToolCallRecord("run_python", {"code": code}, None, ok=False,
                               error=str(exc), elapsed_s=time.perf_counter() - started)
            )
            raise
        ctx.deps.tool_calls.append(
            ToolCallRecord("run_python", {"code": code}, out,
                           elapsed_s=time.perf_counter() - started)
        )
        return out

    @agent.tool_plain
    def context_now() -> str:
        """Report the current date, time, host and active model."""
        return _context_now()

    return agent


__all__ = ["build_agent", "Deps", "SYSTEM_PROMPT"]

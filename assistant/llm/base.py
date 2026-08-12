"""The network seam (CLAUDE.md design rule 8).

Nothing outside this package imports an Ollama client or assumes the model is
local. Phases 0-4 run everything on the workstation; phase 5 moves the caller to
the Jetson and the model stays here. That must be a config change, not a rewrite.

Everything here streams. Non-streaming code in the voice path is a bug, not an
optimisation target (CLAUDE.md design rule 4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class Chunk:
    """One streamed delta. `done` marks the final chunk, which carries timings."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    # Populated on the final chunk. ttft_s is the number PLAN.md §5 budgets for.
    ttft_s: float | None = None
    total_s: float | None = None
    eval_count: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.eval_count is None or not self.total_s or self.ttft_s is None:
            return None
        gen_s = self.total_s - self.ttft_s
        return self.eval_count / gen_s if gen_s > 0 else None


@runtime_checkable
class ModelClient(Protocol):
    """Implementations: OllamaClient today; anything HTTP-shaped later."""

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[Chunk]:
        """Stream a response. Always an iterator, even for one-shot calls."""
        ...

    async def health(self) -> bool:
        """True if the endpoint answers. Gates the offline-fallback path."""
        ...

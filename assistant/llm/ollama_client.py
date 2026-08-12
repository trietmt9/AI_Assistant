"""Ollama implementation of ModelClient. The only module that knows Ollama exists.

Uses the native /api/chat endpoint rather than the OpenAI-compatible /v1 shim,
because /api/chat returns the timing fields (prompt_eval_duration, eval_count)
that the phase-0 bake-off in PLAN.md §11 needs.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict
from typing import Any

import httpx

from assistant.config import settings
from assistant.llm.base import Chunk, Message

NS_PER_S = 1_000_000_000


def _serialise(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.primary_model
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_s)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[Chunk]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [_serialise(m) for m in messages],
            "stream": True,
            "think": settings.enable_thinking if think is None else think,
            "options": {
                "num_ctx": settings.num_ctx,
                "temperature": settings.temperature,
            },
        }
        if tools:
            payload["tools"] = list(tools)

        started = time.perf_counter()
        ttft: float | None = None

        async with self._client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message") or {}
                text = msg.get("content") or ""
                calls = msg.get("tool_calls") or []

                if ttft is None and (text or calls):
                    ttft = time.perf_counter() - started

                if data.get("done"):
                    yield Chunk(
                        text=text,
                        tool_calls=calls,
                        done=True,
                        ttft_s=ttft,
                        total_s=time.perf_counter() - started,
                        eval_count=data.get("eval_count"),
                    )
                    return
                yield Chunk(text=text, tool_calls=calls)

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OllamaClient", "Chunk", "Message", "asdict"]

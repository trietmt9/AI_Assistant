"""Model construction — the other half of the network seam.

CLAUDE.md design rule 8: nothing outside `llm/` imports an Ollama client or
assumes the model is local. `ollama_client.py` satisfies that for raw streaming
(it is what the phase-0 bake-off measures with); this module satisfies it for the
`pydantic-ai` agent, which needs a `Model` object rather than a chat iterator.

The point is that phase 5 — moving the caller to the Jetson while the model stays
on the workstation — is a change to `EVELYN_OLLAMA_HOST` and nothing else. Import
`build_model()` from here; never construct a provider at a call site.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from assistant.config import settings


@lru_cache(maxsize=4)
def build_model(model_name: str | None = None, host: str | None = None) -> Model:
    """A `pydantic-ai` model pointed at wherever Ollama currently lives.

    Uses Ollama's OpenAI-compatible endpoint (`:11434/v1`, PLAN.md §4). The
    native `/api/chat` route that `ollama_client.py` uses returns richer timing
    fields, which is why that module exists separately — but `pydantic-ai` speaks
    the OpenAI shape, and tool calling is better tested on that path.
    """
    base = (host or settings.ollama_host).rstrip("/")
    return OpenAIChatModel(
        model_name or settings.primary_model,
        provider=OllamaProvider(base_url=f"{base}/v1"),
    )


def model_settings() -> dict:
    """Per-run settings shared by every agent.

    **`num_ctx` is deliberately absent, because it cannot be set from here.**
    Measured 2026-08-12: passing `extra_body={"options": {"num_ctx": 2048}}` has
    no effect — Ollama's OpenAI-compatible `/v1` route drops the `options` block,
    and `ollama ps` still reported a 32768 context. Shipping that call would be
    code that looks like configuration and is not.

    Ollama auto-sizes the context instead, and currently picks 32768 for this
    model, which comfortably exceeds the 8192 PLAN.md §3 asks for. Two things
    follow if you ever need to pin it:

    * bake it into a Modelfile (`PARAMETER num_ctx 8192`, then `ollama create`),
      which applies on every route; or
    * use `llm/ollama_client.py`, which talks to the native `/api/chat` endpoint
      where `options` *is* honoured — that is why it sets `num_ctx` and this
      does not.

    Worth watching: a 32k KV cache on a 27B is ~4 GB against the ~1 GB §3
    budgeted at 8k. It fits today (≈19.8 GB of 22.4 GiB usable) but leaves less
    headroom than planned.
    """
    return {"temperature": settings.temperature}


__all__ = ["build_model", "model_settings"]

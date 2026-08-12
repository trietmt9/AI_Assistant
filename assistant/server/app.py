"""The endpoint the laptop and phone talk to.

PLAN.md §2. Two routes, deliberately different, because the clients want
different things:

* **`/v1/chat/completions`** — OpenAI-compatible, so any existing Android client
  works without writing an app. Runs the full `pydantic-ai` agent: tools,
  citations, multi-step. Thinking is on (it cannot be turned off on that path,
  §5), so expect 15-35 s per turn. Acceptable when you are typing.

* **`/ws/voice`** — the laptop CLI's audio socket. Runs the **fast path**:
  pre-retrieve, one generation through `llm/ollama_client.py` with `think=False`,
  stream sentences into Piper as they complete. No tool loop. §5 measured
  thinking at 30x on TTFT, and 20 s of silence is not a conversation.

That split is the whole design. The text path optimises for capability, the voice
path for latency, and they share retrieval and the model underneath.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from assistant.config import settings
from assistant.server.auth import check_bearer, load_or_create_token

log = logging.getLogger(__name__)

# Populated at startup so no turn pays model-load latency.
_state: dict = {}


def use_cached_models_only() -> bool:
    """Skip HuggingFace's "is there a newer version?" check when models are local.

    `sentence-transformers` and `faster-whisper` each round-trip to huggingface.co
    on load even when fully cached. Measured 2026-08-12: **~4 s of the ~14 s
    startup**, plus the "sending unauthenticated requests to the HF Hub" warnings.

    Only enabled when the caches actually exist, so a first run can still
    download. An explicit `HF_HUB_OFFLINE` in the environment always wins.
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return False

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    needed = ["models--BAAI--bge-m3", f"models--Systran--faster-whisper-{settings.stt_model}"]
    if all((hub / name).exists() for name in needed):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm everything before accepting traffic.

    A cold Whisper load or Piper voice download in the middle of the first
    utterance is several seconds, and that is precisely when the latency budget
    gets judged. Measured warm-up, 2026-08-12:

        bge-m3     ~7 s   (568M model off disk; the dominant cost)
        whisper    ~1.5 s
        piper      ~1 s

    Set `EVELYN_WARM_ON_START=false` to skip it. The models then load on first
    use instead — good for iterating on server code, bad for measuring latency.
    """
    from assistant.retrieval.search import Retriever
    from assistant.voice import stt, tts

    _state["token"] = load_or_create_token()
    started = time.perf_counter()

    log.info("opening index...")
    _state["retriever"] = Retriever()

    if settings.warm_on_start:
        loop = asyncio.get_running_loop()
        log.info("warming embeddings (~7 s)...")
        await loop.run_in_executor(None, lambda: _state["retriever"].search("warm", k=1))
        log.info("warming speech (~3 s)...")
        await loop.run_in_executor(None, stt.warm)
        await loop.run_in_executor(None, tts.warm)
        _state["sample_rate"] = tts.sample_rate()
        log.info("warmed in %.1fs", time.perf_counter() - started)
    else:
        log.warning("warm-up skipped; the first request will be slow")

    log.info("ready on %s:%s", settings.server_host, settings.server_port)
    yield
    _state.clear()


app = FastAPI(title="Evelyn", lifespan=lifespan)


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    from fastapi import HTTPException

    try:
        check_bearer(request, _state.get("token", ""))
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code,
                            headers=exc.headers or {})
    return await call_next(request)


@app.get("/health")
async def health() -> dict:
    """Liveness only. Deliberately says nothing about the model or corpus."""
    return {"ok": True}


# --- OpenAI-compatible chat, for the Android app -------------------------


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """Some clients call this before chatting; give them one entry."""
    return {
        "object": "list",
        "data": [{"id": settings.primary_model, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible. Streaming and non-streaming both supported."""
    body = await request.json()
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))

    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content") or ""
            break
    if not user_text:
        return JSONResponse({"error": "no user message"}, status_code=400)

    cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    model = settings.primary_model

    from assistant.agent import Deps, build_agent
    from assistant.tracing import TraceStore, trace_turn

    agent = _state.setdefault("agent", build_agent(_state["retriever"]))
    deps = Deps(retriever=_state["retriever"])
    traces = _state.setdefault("traces", TraceStore())
    session = request.headers.get("x-session-id") or uuid.uuid4().hex

    if not stream:
        with trace_turn(traces, session, user_text, model=model, mode="api") as turn:
            result = await agent.run(user_text, deps=deps)
            turn.final_answer = result.output
            turn.tool_calls = list(deps.tool_calls)
            turn.retrieval = list(deps.retrieval_log)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.output},
                "finish_reason": "stop",
            }],
        }

    async def generate():
        from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPartDelta
        from pydantic_ai.messages import TextPart

        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(_ctx, events):
            async for event in events:
                text = None
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    text = event.part.content
                elif isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, TextPartDelta
                ):
                    text = event.delta.content_delta
                if text:
                    await queue.put(text)

        yield _sse(_chunk(cid, model, {"role": "assistant", "content": ""}))

        parts: list[str] = []

        async def run_agent():
            try:
                with trace_turn(traces, session, user_text, model=model, mode="api") as turn:
                    result = await agent.run(user_text, deps=deps, event_stream_handler=on_event)
                    turn.final_answer = result.output
                    turn.tool_calls = list(deps.tool_calls)
                    turn.retrieval = list(deps.retrieval_log)
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        while True:
            item = await queue.get()
            if item is None:
                break
            parts.append(item)
            yield _sse(_chunk(cid, model, {"content": item}))
        await task

        yield _sse(_chunk(cid, model, {}, finish="stop"))
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- voice socket, for the laptop CLI ------------------------------------


VOICE_PROMPT = """\
You are Evelyn, a local voice assistant. You are being spoken to and your reply
will be read aloud.

Answer from the passages below and name the source you used. Keep it to one or
two sentences — this is speech, not a document. If the passages do not answer the
question, say so plainly and say what you looked for. Never invent a detail about
the user's hardware or research.
"""


async def _voice_turn(ws: WebSocket, audio: np.ndarray) -> None:
    """One spoken turn: STT -> retrieve -> generate -> speak, all streaming."""
    from assistant.llm.base import Message
    from assistant.llm.ollama_client import OllamaClient
    from assistant.tools.knowledge import format_hits
    from assistant.voice import stt, tts

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    # Stage timings are reported to the client on every turn, not just when
    # debugging. §5's budget is the whole point of this path, and an aggregate
    # "12.8 s" tells you nothing about which stage to attack.
    timings: dict[str, float] = {}

    transcript = await loop.run_in_executor(None, stt.transcribe, audio)
    timings["stt_s"] = round(time.perf_counter() - t0, 3)
    if transcript.is_empty:
        await ws.send_json({"type": "transcript", "text": ""})
        await ws.send_json({"type": "done", "reason": "no speech"})
        return
    await ws.send_json({
        "type": "transcript", "text": transcript.text,
        "language": transcript.language, "elapsed_s": round(transcript.elapsed_s, 3),
    })

    mark = time.perf_counter()
    hits = await loop.run_in_executor(
        None,
        lambda: _state["retriever"].search(transcript.text, k=settings.voice_retrieval_k),
    )
    timings["retrieval_s"] = round(time.perf_counter() - mark, 3)

    # Shorter passages than the text path uses. §5: the text path can afford
    # context the voice path cannot, and this is that setting actually applied —
    # `format_hits` defaults to 1100 chars, which is a text-path number.
    if hits:
        passages = "\n\n---\n\n".join(
            f"[{i}] {h.render(settings.voice_max_chars_per_hit)}"
            for i, h in enumerate(hits, 1)
        )
    else:
        passages = "(no relevant passages found)"
    prompt = f"Passages:\n{passages}\n\nQuestion: {transcript.text}"
    timings["prompt_tokens_est"] = len(prompt) // 4

    # The fast path: native /api/chat, thinking off. This is the 30x (§5).
    client = OllamaClient()
    queue: asyncio.Queue = asyncio.Queue()

    llm_started = time.perf_counter()

    async def produce():
        first = True
        try:
            async for chunk in client.chat(
                [Message("system", VOICE_PROMPT), Message("user", prompt)],
                think=False,
            ):
                if chunk.text:
                    if first:
                        timings["llm_ttft_s"] = round(time.perf_counter() - llm_started, 3)
                        first = False
                    await queue.put(chunk.text)
        finally:
            timings["llm_total_s"] = round(time.perf_counter() - llm_started, 3)
            await queue.put(None)
            await client.aclose()

    async def text_stream():
        while (item := await queue.get()) is not None:
            await ws.send_json({"type": "delta", "text": item})
            yield item

    producer = asyncio.create_task(produce())
    first_audio_at: float | None = None
    spoken: list[str] = []

    async for speech in tts.stream_speech(text_stream()):
        if first_audio_at is None:
            first_audio_at = time.perf_counter() - t0
            await ws.send_json({
                "type": "speaking",
                "sample_rate": speech.sample_rate,
                "ttfa_s": round(first_audio_at, 3),
            })
        spoken.append(speech.text)
        await ws.send_bytes(speech.audio)

    await producer
    await ws.send_json({
        "type": "done",
        "text": " ".join(spoken),
        "citations": [h.citation for h in hits],
        "total_s": round(time.perf_counter() - t0, 3),
        "ttfa_s": round(first_audio_at, 3) if first_audio_at else None,
        "timings": timings,
    })


@app.websocket("/ws/voice")
async def voice_socket(ws: WebSocket) -> None:
    """Binary audio in, JSON events and binary audio out.

    Protocol, kept boring on purpose (§2 "keep the transport boring"):

        client -> {"type":"start"}            begin an utterance
        client -> <binary>                    int16 mono PCM @ 16 kHz
        client -> {"type":"end"}              finished speaking
        server -> {"type":"transcript",...}   what it heard
        server -> {"type":"delta",...}        text as it generates
        server -> {"type":"speaking",...}     first audio follows, with sample_rate
        server -> <binary>                    int16 mono PCM at that rate
        server -> {"type":"done",...}         timings and citations
    """
    token = _state.get("token", "")
    supplied = ws.query_params.get("token", "")
    if not supplied:
        header = ws.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    import secrets as _secrets

    if not supplied or not _secrets.compare_digest(supplied, token):
        await ws.close(code=4401)
        return

    await ws.accept()
    buffer = bytearray()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                buffer.extend(data)
                continue
            if (text := message.get("text")) is None:
                continue

            event = json.loads(text)
            kind = event.get("type")
            if kind == "start":
                buffer.clear()
            elif kind == "end":
                audio = np.frombuffer(bytes(buffer), dtype=np.int16)
                buffer.clear()
                if audio.size:
                    await _voice_turn(ws, audio)
                else:
                    await ws.send_json({"type": "done", "reason": "empty"})
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("voice socket error")
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass


__all__ = ["app"]

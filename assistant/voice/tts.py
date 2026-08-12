"""Text to speech — Piper, on the CPU, streaming.

PLAN.md §5, design rule 4: **everything streams.** Non-streaming here is not a
missed optimisation, it is a bug — synthesising a whole paragraph before playing
any of it adds seconds of dead air to every turn, and the assistant feels broken
long before it feels slow.

Two levels of streaming, both needed:

1. **Sentence chunking.** The LLM's token stream is split on sentence boundaries
   and each sentence is synthesised as soon as it completes, so speech starts on
   the first clause rather than the finished answer.
2. **Chunk streaming within a sentence.** Piper itself yields audio incrementally.

Runs on the CPU deliberately. Piper is small and fast enough there, and the GPU is
already holding the 27B plus Whisper.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from assistant.config import settings, DATA_DIR

log = logging.getLogger(__name__)

VOICE_DIR = DATA_DIR / "voices"

# Sentence boundary. Deliberately conservative: a false split mid-sentence is
# audible as an unnatural pause, while a missed split only delays speech a little.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
# Below this, wait for more text rather than speaking a fragment.
MIN_SPEAKABLE_CHARS = 24

# Stripped before speaking. Markdown that reads fine on a terminal sounds like
# noise aloud -- "star star PA4 star star".
_MD_STRIP = [
    (re.compile(r"```.*?```", re.DOTALL), " code block omitted "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),
    (re.compile(r"\*([^*]*)\*"), r"\1"),
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^\s*[-*]\s+", re.MULTILINE), ""),
    (re.compile(r"\s+"), " "),
]


@dataclass(slots=True)
class Speech:
    audio: bytes  # int16 mono PCM
    sample_rate: int
    text: str
    elapsed_s: float = 0.0


def speakable(text: str) -> str:
    """Strip markdown so it does not get read aloud literally."""
    out = text
    for pattern, repl in _MD_STRIP:
        out = pattern.sub(repl, out)
    return out.strip()


@lru_cache(maxsize=1)
def _voice():
    from piper import PiperVoice

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    name = settings.tts_voice
    onnx = VOICE_DIR / f"{name}.onnx"

    if not onnx.exists():
        from piper.download_voices import download_voice

        log.info("downloading piper voice %s (once)", name)
        download_voice(name, VOICE_DIR)

    log.info("loading piper voice %s", name)
    return PiperVoice.load(onnx, use_cuda=False)


def sample_rate() -> int:
    return int(_voice().config.sample_rate)


def synthesize(text: str) -> Speech:
    """Synthesise one chunk of text to int16 PCM."""
    from piper import SynthesisConfig

    text = speakable(text)
    if not text:
        return Speech(audio=b"", sample_rate=sample_rate(), text="")

    started = time.perf_counter()
    voice = _voice()
    cfg = SynthesisConfig(length_scale=1.0 / max(0.1, settings.tts_speed))
    audio = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text, cfg))
    return Speech(
        audio=audio,
        sample_rate=int(voice.config.sample_rate),
        text=text,
        elapsed_s=time.perf_counter() - started,
    )


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a growing buffer into complete sentences plus a remainder.

    Returns (sentences_ready_to_speak, text_still_accumulating). The remainder is
    fed back in on the next call — this is what lets speech begin on the first
    clause of a streaming answer.
    """
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    ready = [p.strip() for p in complete if p.strip()]
    return ready, remainder


async def stream_speech(text_stream: AsyncIterator[str]) -> AsyncIterator[Speech]:
    """Turn a stream of text deltas into a stream of audio chunks.

    This is the piece that makes the voice loop feel responsive: the first
    sentence is spoken while the model is still generating the third.

    Synthesis is pushed to a thread. `synthesize` is blocking CPU work, and
    calling it directly from the event loop stalls the entire server — including
    the very token stream feeding this generator — for the duration of every
    sentence. On an async server that is a correctness bug, not a tuning choice.
    """
    import asyncio

    loop = asyncio.get_running_loop()

    async def say(text: str) -> Speech:
        return await loop.run_in_executor(None, synthesize, text)

    buffer = ""
    async for delta in text_stream:
        buffer += delta
        ready, buffer = split_sentences(buffer)
        for sentence in ready:
            if len(sentence) < MIN_SPEAKABLE_CHARS and not buffer:
                continue
            speech = await say(sentence)
            if speech.audio:
                yield speech
    tail = buffer.strip()
    if tail:
        speech = await say(tail)
        if speech.audio:
            yield speech


def stream_speech_sync(text_chunks: Iterator[str]) -> Iterator[Speech]:
    """Synchronous variant, for tests and the daemon."""
    buffer = ""
    for delta in text_chunks:
        buffer += delta
        ready, buffer = split_sentences(buffer)
        for sentence in ready:
            speech = synthesize(sentence)
            if speech.audio:
                yield speech
    if buffer.strip():
        speech = synthesize(buffer.strip())
        if speech.audio:
            yield speech


def warm() -> None:
    """Download and load the voice. Call at server start, not on first turn."""
    synthesize("Ready.")


__all__ = [
    "synthesize",
    "stream_speech",
    "stream_speech_sync",
    "split_sentences",
    "speakable",
    "sample_rate",
    "warm",
    "Speech",
]

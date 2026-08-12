"""Speech to text — `faster-whisper` on the GPU.

PLAN.md §5. This runs on the **server**, not the client: a phone cannot run
Whisper at useful speed, and the workstation's GPU is idle between turns anyway.

The model is small (~500 MB at int8) and coexists with the 27B — measured
residency for the 27B is ~19.8 GB of 22.4 GiB usable, which leaves room for this
but not for much else. If VRAM gets tight, `EVELYN_STT_DEVICE=cpu` costs about
1.5x on a 5-second utterance and frees it entirely.

**Language is autodetected by default.** The corpus is English and Chinese
(§2 — the admin PDFs are Chinese), and Whisper handles both. Pin
`EVELYN_STT_LANGUAGE=en` only if autodetection starts mis-firing on short
utterances, which is its known weakness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from assistant.config import settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # what Whisper wants; clients must send this


@dataclass(slots=True)
class Transcript:
    text: str
    language: str = ""
    duration_s: float = 0.0
    elapsed_s: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _build(device: str, compute_type: str):
    from faster_whisper import WhisperModel

    log.info("loading whisper %s on %s (%s)", settings.stt_model, device, compute_type)
    return WhisperModel(settings.stt_model, device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def _model():
    """Load Whisper, falling back to CPU if CUDA is not actually usable.

    **The fallback probes with a real inference, not just construction.** This
    machine ships CUDA 13 (`libcublas.so.13`) while ctranslate2 is built against
    CUDA 12, and the resulting `libcublas.so.12 is not found` surfaces on the
    *first transcription*, not when the model object is created. An earlier
    version wrapped only the constructor and so failed at server startup with a
    fallback that never fired.
    """
    if settings.stt_device == "cpu":
        return _build("cpu", "int8")

    try:
        model = _build(settings.stt_device, settings.stt_compute_type)
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE // 2, dtype=np.float32), beam_size=1
        )
        list(segments)  # force the generator; this is where CUDA actually loads
        return model
    except Exception as exc:
        log.warning(
            "whisper on %s unusable (%s) -- falling back to CPU int8",
            settings.stt_device, exc,
        )
        return _build("cpu", "int8")


def transcribe(audio: np.ndarray, *, language: str | None = None) -> Transcript:
    """Transcribe mono float32 PCM at 16 kHz, range [-1, 1].

    Args:
        audio: 1-D float32 array. Int16 input is accepted and converted.
        language: ISO code, or None to autodetect.
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    audio = np.ascontiguousarray(audio, dtype=np.float32).flatten()

    if audio.size < SAMPLE_RATE // 10:  # <100 ms is a click, not speech
        return Transcript(text="")

    started = time.perf_counter()
    segments, info = _model().transcribe(
        audio,
        language=language or settings.stt_language,
        beam_size=1,  # greedy: this is a latency budget, not a benchmark
        vad_filter=True,
        condition_on_previous_text=False,  # stops one bad turn poisoning the next
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return Transcript(
        text=text,
        language=getattr(info, "language", "") or "",
        duration_s=getattr(info, "duration", 0.0) or 0.0,
        elapsed_s=time.perf_counter() - started,
    )


def warm() -> None:
    """Load the model and run one inference. Call at server start.

    Without this the first real utterance pays several seconds of model load,
    which is exactly the moment the latency budget is being judged.
    """
    transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))


__all__ = ["transcribe", "warm", "Transcript", "SAMPLE_RATE"]

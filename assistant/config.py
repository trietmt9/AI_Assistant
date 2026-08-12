"""Central settings. Secrets come from .env, never from code (CLAUDE.md safety rules)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EVELYN_", extra="ignore")

    # Where the model is served. Localhost during phases 0-4; the workstation's
    # LAN address once the Jetson takes over the front end (PLAN.md §2).
    ollama_host: str = "http://127.0.0.1:11434"

    # PLAN.md §3. Chosen by the phase-0 bake-off, measured 2026-08-04:
    # 44.8 tok/s vs 28.3 for qwen3.5:27b (MTP works on sm_75), tool calling tied
    # at 21/22. Changing this after traces are collected means retraining, since
    # a LoRA adapter is base-model-specific (TRAINING_PLAN.md §2).
    primary_model: str = "qwen3.6:27b-mtp-q4_K_M"
    fallback_model: str = "qwen3.5:9b"

    # Default context. PLAN.md §3: 4096 is too small for a system prompt plus
    # retrieved chunks plus a tool trace.
    num_ctx: int = 8192
    temperature: float = 0.3

    # Reasoning traces blow the voice budget (PLAN.md §5).
    enable_thinking: bool = False

    request_timeout_s: float = 120.0

    # --- server (PLAN.md §2) ------------------------------------------
    #
    # **Default bind is 127.0.0.1 and that default is load-bearing.** This
    # machine has a globally routable address (120.126.10.44/24, TANet, no NAT),
    # so binding 0.0.0.0 publishes the assistant -- and every document it can
    # read -- to the internet. Set `EVELYN_SERVER_HOST` to the Tailscale address
    # to reach it from the laptop and phone; never to 0.0.0.0.
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    # Bearer token for every route except /health. Generated on first `eve
    # serve` if unset and written to data/, so there is no unauthenticated mode
    # to forget to turn off.
    server_token: str = ""

    # --- speech (PLAN.md §5) ------------------------------------------
    #
    # **CPU, because CUDA does not work here** (measured 2026-08-12). This box
    # has CUDA 13 (`libcublas.so.13`); ctranslate2, which backs faster-whisper,
    # is built against CUDA 12 and fails with "libcublas.so.12 is not found" on
    # the first transcription. Measured CPU cost is 3.1x realtime — a 6.2 s
    # utterance transcribes in 2.0 s — which is usable but is a real term in the
    # §5 budget.
    #
    # To get the GPU back: `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
    # and put them on the loader path, then set `EVELYN_STT_DEVICE=cuda`. It
    # coexists with the CUDA 13 torch build (different directories), but it is a
    # second CUDA runtime on the machine — worth doing only if STT latency turns
    # out to matter more than the risk.
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str | None = None  # None = autodetect; the corpus is en + zh

    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0

    # Voice turns run through llm/ollama_client.py, not pydantic-ai: only the
    # native /api/chat route can disable thinking, which is worth 30x on TTFT
    # (PLAN.md §5). Fewer, shorter chunks too -- the text path can afford
    # context the voice path cannot.
    voice_retrieval_k: int = 3
    voice_max_chars_per_hit: int = 600

    # Retrieval (PLAN.md §4), settled by the phase-1 ablation on 2026-08-04.
    #
    # Reranking is OFF because it was measured to buy nothing: hit@5, hit@1 and
    # MRR were byte-identical with and without it, at 27.7 s per query versus
    # 91 ms. That is a 300x cost for zero gain on this corpus. Turn it back on
    # only if a larger golden set shows it separating configs -- 8 questions
    # cannot rule out that it helps on harder ones.
    rerank_enabled: bool = False
    rerank_device: str = "cpu"
    rerank_candidates: int = 20
    retrieval_k: int = 5

    # Derived paths — data/ is gitignored in full.
    index_dir: Path = DATA_DIR / "index"
    parsed_dir: Path = DATA_DIR / "parsed"
    # Incremental re-index bookkeeping: path + mtime + size + content hash
    # (PLAN.md §6). Separate from the index so the index can be rebuilt from
    # cached parses without re-reading the corpus.
    manifest_db: Path = DATA_DIR / "index_manifest.db"
    traces_db: Path = DATA_DIR / "traces.db"
    memory_db: Path = DATA_DIR / "memory.db"
    scratch_dir: Path = DATA_DIR / "scratch"


settings = Settings()

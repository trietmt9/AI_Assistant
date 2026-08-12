"""The laptop side. Talks to the workstation; runs no model, holds no index.

PLAN.md §2. Everything here must work with only the `client` extra installed —
`httpx`, `websockets`, `sounddevice`, `numpy`. **Do not import `assistant.agent`,
`assistant.retrieval` or anything that pulls torch from this module**, or
`uv tool install` on the laptop stops being a seconds-long operation.

That constraint is the reason config here is a small TOML file rather than the
`pydantic-settings` object the server uses: the laptop has no `.env`, no `data/`
directory, and no reason to know what a LanceDB index is.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("EVELYN_CLIENT_CONFIG", Path.home() / ".config" / "evelyn" / "config.toml")
)

MIC_RATE = 16_000  # what the server's Whisper expects
BLOCK = 1024


def _audio_device(sd) -> int | None:
    """Pick a device that will actually accept 16 kHz mono.

    PortAudio's idea of the default device on Linux is frequently a bare ALSA
    `hw:` node, which accepts only its card's native rate -- 48 kHz on the
    workstation -- and answers anything else with `Invalid sample rate
    [PaErrorCode -9997]`. It also picks per-direction, so the default *output*
    can land on HDMI while the user's speakers are on the analogue codec.

    The PulseAudio/PipeWire device has neither problem: it resamples
    transparently and follows whatever default source and sink the desktop is
    actually using. Prefer it, fall back to PortAudio's default, and let
    `EVELYN_AUDIO_DEVICE` override both.
    """
    override = os.environ.get("EVELYN_AUDIO_DEVICE")
    if override:
        return int(override) if override.isdigit() else sd.query_devices(override)["index"]
    for name in ("pipewire", "pulse"):
        try:
            return sd.query_devices(name)["index"]
        except ValueError:  # sounddevice's "no device matching ..."
            continue
    return None  # None means "PortAudio default" to sounddevice


@dataclass(slots=True)
class RemoteConfig:
    url: str
    token: str

    @property
    def ws_url(self) -> str:
        base = self.url.rstrip("/")
        ws = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws}/ws/voice?token={self.token}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def load_config(url: str | None = None, token: str | None = None) -> RemoteConfig:
    """Explicit args, then env, then `~/.config/evelyn/config.toml`."""
    data: dict = {}
    if CONFIG_PATH.exists():
        data = tomllib.loads(CONFIG_PATH.read_text())

    resolved_url = url or os.environ.get("EVELYN_SERVER_URL") or data.get("url", "")
    resolved_token = token or os.environ.get("EVELYN_SERVER_TOKEN") or data.get("token", "")

    if not resolved_url:
        raise SystemExit(
            f"No server configured. Write {CONFIG_PATH} with:\n\n"
            '  url = "https://workstation.your-tailnet.ts.net:8080"\n'
            '  token = "..."\n\n'
            "or set EVELYN_SERVER_URL and EVELYN_SERVER_TOKEN."
        )
    return RemoteConfig(url=resolved_url.rstrip("/"), token=resolved_token)


def write_config(url: str, token: str) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(f'url = "{url}"\ntoken = "{token}"\n')
    CONFIG_PATH.chmod(0o600)
    return CONFIG_PATH


# --- text ----------------------------------------------------------------


async def stream_chat(cfg: RemoteConfig, message: str, session: str = ""):
    """Stream a text turn over the OpenAI-compatible route.

    Uses the same endpoint the Android app does, so if this works the phone
    works — one fewer thing to debug twice.
    """
    import httpx

    payload = {
        "model": "evelyn",
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }
    headers = dict(cfg.headers)
    if session:
        headers["x-session-id"] = session

    async with httpx.AsyncClient(timeout=300.0) as http:
        async with http.stream(
            "POST", f"{cfg.url}/v1/chat/completions", json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                raise SystemExit(f"server returned {response.status_code}: {body[:300]}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if content := delta.get("content"):
                    yield content


async def health(cfg: RemoteConfig) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(f"{cfg.url}/health")
            return r.status_code == 200
    except Exception:
        return False


# --- voice ---------------------------------------------------------------


async def voice_turn(cfg: RemoteConfig, *, seconds: float = 0.0, on_event=None) -> dict:
    """Record from the microphone, send it up, play the reply.

    `seconds=0` means push-to-talk: recording runs until the caller's callback
    returns False. The laptop is a native process, so unlike a browser it can
    take the microphone directly — which is also what makes a real wake word
    possible later (§5).
    """
    try:
        import numpy as np
        import sounddevice as sd
        import websockets
    except ImportError as exc:
        # `uv tool install` builds its own isolated environment and does NOT
        # install optional extras -- so a numpy sitting in some project venv is
        # invisible here. Say that, rather than emitting a bare ImportError that
        # sends people hunting through the wrong virtualenv.
        raise SystemExit(
            f"Voice needs the `client` extra ({exc.name} is missing).\n\n"
            "Reinstall with it:\n"
            '  uv tool install --force "assistant[client] @ /path/to/AI_Assistant"\n\n'
            "Or add just the missing pieces to the existing tool env:\n"
            "  uv tool install --force --with numpy --with sounddevice "
            "--with websockets assistant\n\n"
            "`eve remote` (text) needs none of this and works already."
        ) from exc

    device = _audio_device(sd)
    events: dict = {}
    frames: list[bytes] = []

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    async with websockets.connect(cfg.ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))

        loop_done = False

        def callback(indata, _frames, _time, status):
            if status:
                emit({"type": "warn", "detail": str(status)})
            frames.append(bytes(indata))

        emit({"type": "recording"})
        with sd.RawInputStream(
            samplerate=MIC_RATE, blocksize=BLOCK, dtype="int16",
            channels=1, callback=callback, device=device,
        ):
            if seconds > 0:
                await _sleep(seconds)
            else:
                while not loop_done:
                    await _sleep(0.1)
                    if on_event and on_event({"type": "tick"}) is False:
                        loop_done = True

        for frame in frames:
            await ws.send(frame)
        await ws.send(json.dumps({"type": "end"}))
        emit({"type": "thinking"})

        stream = None
        sample_rate = 22050
        try:
            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    if stream is None:
                        stream = sd.RawOutputStream(
                            samplerate=sample_rate, channels=1, dtype="int16",
                            device=device,
                        )
                        stream.start()
                    stream.write(message)
                    continue

                event = json.loads(message)
                emit(event)
                if event.get("type") == "speaking":
                    sample_rate = int(event.get("sample_rate", sample_rate))
                elif event.get("type") in {"done", "error"}:
                    events = event
                    break
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

    return events


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


__all__ = [
    "RemoteConfig",
    "load_config",
    "write_config",
    "stream_chat",
    "voice_turn",
    "health",
    "CONFIG_PATH",
]

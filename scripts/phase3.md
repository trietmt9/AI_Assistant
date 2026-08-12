# Phase 3 runbook — the server, voice, and remote clients

`PLAN.md` §2 (topology decided 2026-08-12) and §5. The workstation is the server;
the laptop runs the `eve` CLI; the phone uses any OpenAI-compatible Android app.

---

## What got built

| Module | Does |
|---|---|
| `assistant/server/app.py` | FastAPI: `/v1/chat/completions`, `/ws/voice`, `/health` |
| `assistant/server/auth.py` | Bearer token, auto-generated, no unauthenticated mode |
| `assistant/voice/stt.py` | faster-whisper, with a real CUDA→CPU fallback |
| `assistant/voice/tts.py` | Piper, sentence-chunked streaming, markdown stripped |
| `assistant/client.py` | The laptop side — thin deps only |
| `eve serve / login / remote / talk` | CLI commands |

**Two routes, deliberately different:**

- **`/v1/chat/completions`** — full `pydantic-ai` agent, tools, multi-step. Runs
  with thinking on because `/v1` cannot disable it (§5), so 15–35 s per turn.
  Fine when typing.
- **`/ws/voice`** — the fast path: pre-retrieve → one generation through
  `llm/ollama_client.py` with `think=False` → stream sentences into Piper. No
  tool loop. This is where the 30× lives.

---

## Run it

```bash
# workstation
.venv/bin/python -m assistant.cli serve

# laptop, once
uv tool install /path/to/AI_Assistant           # or from git
eve login http://workstation:8080 --token "$(ssh workstation cat .../data/server_token)"
eve remote "what did my notes say about LOSO?"  # text
eve talk                                        # voice
```

The token is generated on first `serve` into `data/server_token` (mode 600).
There is no way to run without auth — see `server/auth.py` for why that is
deliberate.

---

## Measured, 2026-08-12

Voice turn, end to end, warm, over loopback:

| Question | TTFA | STT | Retrieval | LLM TTFT | Prompt |
|---|---|---|---|---|---|
| "Which STM32 pin is the ADS1298 chip select?" | **5.56 s** | 2.07 | 0.10 | 2.14 | 550 tok |
| "How is the reset pin wired on my FES board?" | **5.45 s** | 2.02 | 0.10 | 2.22 | 537 tok |
| "Hello, are you there?" | **3.85 s** | 1.98 | 0.09 | 1.57 | 376 tok |

Answers were correct and cited — PA4 and PB0 respectively.

**Against §5's <1 s target this is still 5×over.** But it is ~4× better than the
agentic path (19.6–31.4 s), and the budget breaks down usefully now:

| Stage | Cost | Fixable? |
|---|---|---|
| STT | ~2.0 s | **Yes** — GPU whisper is ~0.3 s; blocked on a CUDA 12/13 mismatch |
| Retrieval | 0.1 s | Already negligible |
| LLM TTFT | ~2.1 s | Partly — prefill of ~550 tokens |
| First sentence + TTS | ~1.3 s | Partly — TTS itself is 33× realtime, the wait is the model finishing a sentence |

The two worthwhile attacks, in order:

1. **Streaming STT.** Transcribe *while* the user speaks rather than after they
   stop. Removes most of 2 s from the critical path and needs no new hardware.
2. **GPU Whisper.** `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`, put
   them on the loader path, set `EVELYN_STT_DEVICE=cuda`. Worth ~1.7 s. It means
   a second CUDA runtime on the box, so weigh it.

---

## Things that bit, recorded so they do not bite twice

### faster-whisper cannot use this GPU

`libcublas.so.12 is not found`. The machine has CUDA 13 (`libcublas.so.13`);
ctranslate2 is built against CUDA 12. **The failure surfaces on the first
transcription, not at model construction**, so a `try/except` around
`WhisperModel(...)` never fires — the first version of `stt.py` had exactly that
bug and took the whole server down at startup. The fallback now probes with a
real inference.

CPU Whisper `small` int8 measures **3.1× realtime** (a 6.2 s utterance in 2.0 s),
which is usable, so `stt_device` defaults to `cpu`.

### Piper's first synthesis is 50× slower than its second

0.7× realtime cold, **33× realtime warm** — pure ONNX session warmup. That is
what `tts.warm()` at server start is for. Without it the first spoken reply of
every server lifetime looks catastrophically slow.

### `synthesize()` must not run on the event loop

It is blocking CPU work. Calling it directly from the async generator stalls the
entire server — including the token stream feeding it — for the duration of every
sentence. It runs in an executor now.

### Verify the server actually restarted

`pkill -f "assistant.cli serve"` **matches the shell running it** and kills your
own command. Worse, a failed restart leaves the *old* process serving, so edits
appear to have no effect — or, as happened here, an unrelated improvement gets
attributed to code that was never loaded. Check:

```bash
ps -eo pid,lstart,cmd | grep "cli serve" | grep -v grep
grep -c "address already in use" <server log>
```

---

## Security posture

`PLAN.md` §2: this host has a **globally routable IP with no NAT**
(`120.126.10.44/24`, TANet). Consequences baked into the code:

- `server_host` defaults to `127.0.0.1`, and `eve serve` prints a **red warning**
  if asked to bind `0.0.0.0`.
- Ollama stays on `127.0.0.1`; only this server is ever exposed, and it holds the
  token.
- `/health` is the only unauthenticated route and returns nothing but `{"ok":true}`
  — no model name, no document counts.

**Tailscale is still not installed.** Until it is, the server is loopback-only and
the laptop and phone cannot reach it. That is the correct default; the alternative
is publishing the assistant to the internet.

---

## Next

1. **Install Tailscale**, then set `EVELYN_SERVER_HOST` to the tailnet address.
   Nothing else works from another device until this happens.
2. **Point an Android app at it** — `/v1/chat/completions` and `/v1/models` are
   both live and tested with `curl`.
3. **Try `eve talk` with a real microphone.** Everything so far was tested by
   feeding synthesised speech back in, which is a fair pipeline test but says
   nothing about real-world VAD, accents or room noise.
4. **Streaming STT** is the biggest remaining latency win.

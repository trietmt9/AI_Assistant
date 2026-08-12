"""Phase-0 model bake-off (PLAN.md §11).

No published benchmark covers Turing/sm_75, so these numbers do not exist until
this script produces them. Measures, per candidate model:

  * TTFT on a RAG-shaped prompt   -- the §5 budget assumes 300 ms
  * tokens/sec at 8k context      -- must clear ~3-4 t/s of speech consumption
  * peak VRAM, thinking off       -- must leave room for a KV cache that grows
  * tool-call validity            -- the §10 metric that matters most here
  * over-calling rate             -- calling a tool when none was needed

Run 2026-08-04; `qwen3.6:27b-mtp-q4_K_M` won and is recorded in PLAN.md §3. Still
useful for re-measuring after a model or Ollama upgrade -- never ship a model
change on vibes.

Usage:
    .venv/bin/python eval/bakeoff.py qwen3.6:27b-mtp-q4_K_M qwen3.5:27b
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.llm.base import Message  # noqa: E402
from assistant.llm.ollama_client import OllamaClient  # noqa: E402

# A trimmed, realistic tool set. PLAN.md §9 caps a single prompt at ~8.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the user's personal document corpus (papers, textbooks, notes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "description": "Optional path filter"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate or manipulate a mathematical expression with SymPy. "
            "Use for ALL arithmetic, algebra, calculus and unit conversion.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list",
            "description": "List calendar events in a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO 8601 date"},
                    "end": {"type": "string", "description": "ISO 8601 date"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create",
            "description": "Create a calendar event. Requires confirmation before executing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "home_lights",
            "description": "Control smart lights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "state": {"type": "string", "enum": ["on", "off"]},
                    "brightness": {"type": "integer"},
                },
                "required": ["room", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Control media playback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "next", "previous"],
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_state",
            "description": "Report battery, CPU/GPU load, temperatures and network status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "training_status",
            "description": "Report GPU utilisation, VRAM, current step and loss of a training run.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
            },
        },
    },
]

SYSTEM = (
    "You are Evelyn, a local voice assistant. It is Tuesday 4 August 2026, 14:30. "
    "Route every computation through `calc` -- never do arithmetic yourself. "
    "Call a tool only when it is needed; answer directly otherwise."
)

# (utterance, expected tool name or None). None == answering without a tool is correct.
SCENARIOS: list[tuple[str, str | None]] = [
    ("What's the integral of x squared times sine of x?", "calc"),
    ("How much is 17% of 4,320?", "calc"),
    ("Convert 3.5 electron volts to joules.", "calc"),
    ("Solve x^2 - 5x + 6 = 0.", "calc"),
    ("What did my notes say about Noether's theorem?", "search_docs"),
    ("Find the paper where I wrote about tokamak confinement time.", "search_docs"),
    ("Look up my lecture notes on Hamiltonian mechanics.", "search_docs"),
    ("What's on my calendar tomorrow?", "calendar_list"),
    ("Am I free Friday afternoon?", "calendar_list"),
    ("What do I have on next week?", "calendar_list"),
    ("Book me a dentist appointment Thursday at 3pm.", "calendar_create"),
    ("Schedule a two-hour deep work block tomorrow morning.", "calendar_create"),
    ("Turn the kitchen lights off.", "home_lights"),
    ("Dim the bedroom to 20 percent.", "home_lights"),
    ("Skip this track.", "media_control"),
    ("Pause the music.", "media_control"),
    ("How hot is the GPU right now?", "system_state"),
    ("How's the training run going?", "training_status"),
    # Over-calling checks -- TRAINING_PLAN.md §3 flags this as a real failure mode.
    ("Hello Evelyn.", None),
    ("Thanks, that's all.", None),
    ("What's your name?", None),
    ("Explain in one sentence why the sky is blue.", None),
]

MATH_SPOT_CHECKS = [
    "A 2 kg block slides down a frictionless 30 degree incline. Set up the equations "
    "of motion. Do not compute numbers yourself -- call calc.",
    "State the Navier-Stokes momentum equation for an incompressible Newtonian fluid "
    "and name each term.",
    "Explain why the partition function factorises for non-interacting particles.",
]


def rag_prompt(target_tokens: int = 2500) -> str:
    """A RAG-shaped user turn: retrieved chunks plus a question, ~2-3k tokens."""
    chunk = (
        "[source: notes/statmech/ch4.md | section: Canonical Ensemble | page 87]\n"
        "The canonical partition function is $Z = \\sum_i e^{-\\beta E_i}$ where "
        "$\\beta = 1/k_B T$. For a system of $N$ non-interacting indistinguishable "
        "particles the partition function factorises as $Z_N = Z_1^N / N!$, the "
        "Gibbs correction resolving the entropy-of-mixing paradox. Free energy "
        "follows as $F = -k_B T \\ln Z$, and every thermodynamic quantity derives "
        "from $F$ by differentiation.\n\n"
    )
    # ~4 chars/token is a serviceable approximation for this purpose.
    reps = max(1, (target_tokens * 4) // len(chunk))
    return (
        "Here are retrieved passages:\n\n"
        + chunk * reps
        + "\nQuestion: using only the passages above, explain how the Gibbs "
        "correction changes the entropy of an ideal gas."
    )


def gpu_vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


@dataclass
class Result:
    model: str
    ttft_s: float | None = None
    tok_per_s: float | None = None
    load_s: float | None = None  # cold-start model load, measured separately
    peak_vram_mib: int = 0
    tool_correct: int = 0
    tool_malformed: int = 0
    over_called: int = 0
    under_called: int = 0
    total_scenarios: int = 0
    notes: list[str] = field(default_factory=list)


async def warmup(client: OllamaClient, model: str, res: Result) -> None:
    """Force the model into VRAM before timing anything.

    Without this, TTFT includes reading ~17 GB off disk and measures the wrong
    thing entirely -- the first smoke test reported 50 s TTFT for a 0.8B model
    purely because of cold load. PLAN.md §5 budgets *steady-state* TTFT, which is
    what OLLAMA_KEEP_ALIVE=30m exists to guarantee.

    The load time is worth keeping: it is what a KEEP_ALIVE expiry costs you.
    """
    started = time.perf_counter()
    async for _ in client.chat(
        [Message("user", "hi")], model=model, think=False
    ):
        pass
    res.load_s = time.perf_counter() - started


async def measure_latency(client: OllamaClient, model: str, res: Result) -> None:
    """TTFT and tok/s on a RAG-shaped prompt, thinking off. Model must be warm."""
    msgs = [Message("system", SYSTEM), Message("user", rag_prompt())]
    peak = 0
    stop = False

    async def poll() -> None:
        nonlocal peak
        while not stop:
            peak = max(peak, gpu_vram_mib())
            await asyncio.sleep(0.25)

    watcher = asyncio.create_task(poll())
    try:
        async for ch in client.chat(msgs, model=model, think=False):
            if ch.done:
                res.ttft_s = ch.ttft_s
                res.tok_per_s = ch.tokens_per_second
    finally:
        stop = True
        await watcher
    res.peak_vram_mib = peak


async def measure_tools(client: OllamaClient, model: str, res: Result) -> None:
    schema = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    res.total_scenarios = len(SCENARIOS)

    for utterance, expected in SCENARIOS:
        msgs = [Message("system", SYSTEM), Message("user", utterance)]
        calls: list[dict] = []
        async for ch in client.chat(msgs, model=model, tools=TOOLS, think=False):
            calls.extend(ch.tool_calls)

        if not calls:
            if expected is None:
                res.tool_correct += 1
            else:
                res.under_called += 1
                res.notes.append(f"no call, wanted {expected}: {utterance!r}")
            continue

        if expected is None:
            res.over_called += 1
            got = calls[0].get("function", {}).get("name")
            res.notes.append(f"over-called {got}: {utterance!r}")
            continue

        fn = calls[0].get("function", {})
        name = fn.get("name")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                res.tool_malformed += 1
                res.notes.append(f"unparseable args for {name}: {utterance!r}")
                continue

        if name != expected:
            res.notes.append(f"wrong tool {name}, wanted {expected}: {utterance!r}")
            continue

        required = schema.get(name, {}).get("required", [])
        missing = [k for k in required if k not in (raw_args or {})]
        if missing:
            res.tool_malformed += 1
            res.notes.append(f"{name} missing {missing}: {utterance!r}")
        else:
            res.tool_correct += 1


async def run(model: str) -> Result:
    client = OllamaClient(model=model)
    res = Result(model=model)
    if not await client.health():
        raise SystemExit(f"Ollama not reachable at {client.host} -- is the service running?")

    print(f"\n=== {model} ===", flush=True)
    print("  warmup (loading into VRAM)...", flush=True)
    await warmup(client, model, res)
    print(f"    loaded in {res.load_s:.1f}s", flush=True)
    print("  latency...", flush=True)
    await measure_latency(client, model, res)
    print("  tool calls...", flush=True)
    await measure_tools(client, model, res)
    await client.aclose()
    return res


def report(results: list[Result]) -> None:
    print("\n" + "=" * 82)
    print(
        f"{'model':<24}{'TTFT':>8}{'tok/s':>9}{'VRAM':>10}{'load':>8}"
        f"{'tools ok':>11}{'over':>7}{'bad':>6}"
    )
    print("-" * 82)
    for r in results:
        ttft = f"{r.ttft_s:.2f}s" if r.ttft_s else "n/a"
        tps = f"{r.tok_per_s:.1f}" if r.tok_per_s else "n/a"
        vram = f"{r.peak_vram_mib/1024:.1f}GB" if r.peak_vram_mib > 0 else "n/a"
        load = f"{r.load_s:.0f}s" if r.load_s else "n/a"
        ok = f"{r.tool_correct}/{r.total_scenarios}"
        print(
            f"{r.model:<24}{ttft:>8}{tps:>9}{vram:>10}{load:>8}"
            f"{ok:>11}{r.over_called:>7}{r.tool_malformed:>6}"
        )
    print("=" * 82)

    for r in results:
        if r.notes:
            print(f"\n{r.model} failures:")
            for n in r.notes:
                print(f"  - {n}")

    print("\nPass criteria (PLAN.md §5, §11):")
    print("  TTFT  < 0.30s  -- the voice budget; >1s means the voice loop needs rework")
    print("  tok/s  > 12    -- speech eats 3-4 t/s; anything less has no margin")
    print("  VRAM   < 22GB  -- must leave room for a KV cache that grows")

    out = Path(__file__).parent / "bakeoff_results.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    models = sys.argv[1:] or ["qwen3.6:27b-mtp-q4_K_M", "qwen3.5:27b"]
    report([asyncio.run(run(m)) for m in models])

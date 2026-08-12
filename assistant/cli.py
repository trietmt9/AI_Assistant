"""`eve` — the terminal REPL.

PLAN.md §4: typer + rich + prompt_toolkit. Phase 2's interface, and the one the
traces come from until the voice loop lands in phase 3.

**Output streams.** Design rule 4 makes non-streaming code in the voice path a
bug rather than an optimisation target, and while the CLI is not that path yet,
it shares the agent with it — so the streaming shape is established here where it
is cheap to get right.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from assistant.config import settings

app = typer.Typer(add_completion=False, help="Evelyn — local voice-driven assistant")
console = Console()
log = logging.getLogger(__name__)

BANNER = "[bold cyan]Evelyn[/] — local assistant.  /help for commands, Ctrl-D to exit."


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # These are chatty at INFO and say nothing useful during a chat.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _system_prompt_of(history) -> str:
    """Pull the instructions out of the message history.

    Logged explicitly because `TRAINING_PLAN.md` §3 requires the system prompt
    that was actually in force for a turn — it changes as the agent evolves, and
    a trace without it cannot be replayed.

    `@agent.instructions` output lands on `ModelRequest.instructions`, *not* in a
    `SystemPromptPart`. Both are checked: the part form is what
    `system_prompt=` produces, and either may appear depending on how the agent
    is configured.
    """
    for message in history:
        if instructions := getattr(message, "instructions", None):
            return str(instructions)
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "SystemPromptPart":
                return getattr(part, "content", "") or ""
    return ""


async def _run_turn(agent, deps, trace_store, session_id: str, text: str, history) -> list:
    """One agent turn: stream it, render it, log it."""
    from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPartDelta
    from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart

    from assistant.tracing import trace_turn

    deps.tool_calls.clear()
    deps.retrieval_log.clear()

    with trace_turn(
        trace_store, session_id, text, model=settings.primary_model, mode="cli"
    ) as turn:
        parts: list[str] = []
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:

            async def on_event(_ctx, events):
                """Accumulate text deltas across the *whole* run.

                Deliberately not `run_stream()`. That streams the first model
                response, and leaving its context manager abandons the rest of
                the tool loop -- a turn where the model says something before
                calling a tool ("...and for 12 factorial:") was being truncated
                at exactly that point, with the tool result never spoken. `run()`
                plus this handler streams every response and still runs the loop
                to completion.
                """
                async for event in events:
                    # A text part arrives as one PartStartEvent carrying its
                    # opening content, then PartDeltaEvents for the rest.
                    # Handling only the deltas silently eats the first fragment
                    # of every part -- "My name is Eve" rendered as "name is
                    # Eve", and "12 factorial" as "2 factorial".
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, TextPart
                    ):
                        if event.part.content:
                            parts.append(event.part.content)
                            live.update(Markdown("".join(parts)))
                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        parts.append(event.delta.content_delta)
                        live.update(Markdown("".join(parts)))

            result = await agent.run(
                text,
                deps=deps,
                message_history=history,
                event_stream_handler=on_event,
            )
            # Display keeps everything that streamed, including any narration
            # the model produced before calling a tool -- that is what the user
            # actually watched appear.
            streamed = "".join(parts).strip()
            live.update(Markdown(streamed or result.output))
            history = result.all_messages()

        # The *trace* keeps only the final response. Concatenating the narration
        # would store "My name is Eve. Let me compute that." immediately followed
        # by "My name is Eve, and 12 factorial is 479,001,600" -- duplicated text
        # that reads as a bad answer when TRAINING_PLAN.md §3 comes to build a
        # dataset from it. The narration is not lost; it is in `turn.messages`
        # as its own response, correctly separated.
        answer = result.output or streamed

        turn.final_answer = answer
        # pydantic-ai messages are dataclasses, not BaseModels -- they have no
        # `.model_dump()`. The type adapter is the supported way to round-trip
        # them, and round-tripping matters: TRAINING_PLAN.md §3 needs to rebuild
        # the exact conversation later.
        turn.messages = ModelMessagesTypeAdapter.dump_python(history, mode="json")
        turn.retrieval = list(deps.retrieval_log)
        turn.tool_calls = list(deps.tool_calls)
        turn.system_prompt = _system_prompt_of(history)

    if deps.tool_calls:
        names = ", ".join(tc.tool_name for tc in deps.tool_calls)
        console.print(f"[dim]tools: {names}[/]")
    return history


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO logs"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore prior history each turn"),
) -> None:
    """Start an interactive session."""
    _setup_logging(verbose)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    from assistant.agent import Deps, build_agent
    from assistant.retrieval.search import Retriever
    from assistant.retrieval.store import ChunkStore
    from assistant.tracing import TraceStore

    store = ChunkStore()
    n = store.count()
    if n == 0:
        console.print("[yellow]Index is empty — run the phase-1 ingest first "
                      "(scripts/phase1.md).[/]")

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(f"[dim]{n} chunks indexed · {settings.primary_model}[/]\n")

    retriever = Retriever(store)
    agent = build_agent(retriever)
    trace_store = TraceStore()
    deps = Deps(retriever=retriever)
    session_id = uuid.uuid4().hex
    history: list = []

    hist_file = settings.scratch_dir / "cli_history"
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    prompt = PromptSession(history=FileHistory(str(hist_file)))

    while True:
        try:
            text = prompt.prompt("› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break

        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/help":
            console.print(
                "[dim]/stats   trace statistics\n"
                "/clear   forget conversation history\n"
                "/exit    quit[/]"
            )
            continue
        if text == "/clear":
            history = []
            console.print("[dim]history cleared[/]")
            continue
        if text == "/stats":
            _print_stats(trace_store.stats())
            continue

        try:
            history = asyncio.run(
                _run_turn(agent, deps, trace_store, session_id, text,
                          [] if fresh else history)
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/]")
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/] {exc}")
            if verbose:
                console.print_exception()

    trace_store.close()


def _print_stats(stats: dict) -> None:
    table = Table(title="Traces", show_header=False, box=None)
    for key in ("turns", "tool_calls", "failed_tool_calls", "errors", "avg_elapsed_s"):
        table.add_row(key.replace("_", " "), str(stats.get(key, 0)))
    console.print(table)
    if stats.get("by_tool"):
        t2 = Table(title="By tool", show_header=False, box=None)
        for name, count in stats["by_tool"].items():
            t2.add_row(name, str(count))
        console.print(t2)


@app.command()
def ask(
    question: str = typer.Argument(..., help="One-shot question"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask a single question and exit. Useful for scripting and eval."""
    _setup_logging(verbose)

    from assistant.agent import Deps, build_agent
    from assistant.retrieval.search import Retriever
    from assistant.tracing import TraceStore

    retriever = Retriever()
    agent = build_agent(retriever)
    deps = Deps(retriever=retriever)
    trace_store = TraceStore()
    asyncio.run(
        _run_turn(agent, deps, trace_store, uuid.uuid4().hex, question, [])
    )
    trace_store.close()


def _tailscale_ip() -> str | None:
    """This machine's tailnet address, if Tailscale is up.

    Binding here rather than to `0.0.0.0` is the whole security posture: the
    tailnet is private and authenticated, while this host's real address is
    globally routable with no NAT (PLAN.md §2).
    """
    import shutil
    import subprocess

    if not shutil.which("tailscale"):
        return None
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    addr = out.stdout.strip().splitlines()
    return addr[0].strip() if addr and addr[0].strip() else None


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address. NEVER 0.0.0.0 on this host."),
    port: int = typer.Option(None, help="Port"),
    tailscale: bool = typer.Option(
        False, "--tailscale", help="Bind to this machine's tailnet address"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the server the laptop and phone connect to (PLAN.md §2)."""
    _setup_logging(True if verbose else False)
    import uvicorn

    from assistant.server.app import use_cached_models_only
    from assistant.server.auth import load_or_create_token, token_path

    offline = use_cached_models_only()

    bind = host or settings.server_host
    bind_port = port or settings.server_port

    if tailscale:
        ts = _tailscale_ip()
        if not ts:
            console.print(
                "[red]Tailscale is installed but not connected.[/]\n\n"
                "Run [bold]sudo tailscale up[/] and follow the login URL, then retry.\n"
                "[dim]`tailscale ip -4` should print a 100.x address when it is ready.[/]"
            )
            raise typer.Exit(1)
        bind = ts
        console.print(f"[green]binding to tailnet address {ts}[/]")

    # Preflight the bind. uvicorn's own failure is a bare `[Errno 98] address
    # already in use` after ~30 s of warming Whisper and Piper, which reads as a
    # crash rather than "you already have one running" -- and leaves the old
    # instance quietly serving stale code.
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((bind if bind != "localhost" else "127.0.0.1", bind_port))
    except OSError:
        console.print(
            f"[red]Port {bind_port} is already in use on {bind}.[/]\n\n"
            "Evelyn is probably already running. Find and stop it with:\n\n"
            f"  [bold]ss -ltnp | grep {bind_port}[/]\n"
            "  [bold]kill <pid>[/]\n\n"
            "[dim]Do not use `pkill -f \"assistant.cli serve\"` — the pattern "
            "matches the shell you run it from and kills that instead.[/]"
        )
        raise typer.Exit(1) from None
    finally:
        probe.close()

    token = load_or_create_token()

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(f"  bind   [bold]{bind}:{bind_port}[/]")
    console.print(f"  token  [dim]{token_path()}[/]")
    if offline:
        console.print("  [dim]models: local cache only (no HF round-trip)[/]")
    if settings.warm_on_start:
        console.print("  [dim]warming models, ~15s — first request will be fast[/]")

    if bind == "0.0.0.0":  # noqa: S104 — the whole point is to warn about it
        console.print(
            "\n[bold red]0.0.0.0 on this machine is the public internet.[/]\n"
            "[red]This host has a globally routable address (no NAT). Bind to "
            "127.0.0.1 or a Tailscale address instead.[/]\n"
        )
    elif bind not in {"127.0.0.1", "localhost", "::1"}:
        console.print(f"[yellow]  reachable from other machines at {bind}[/]")

    console.print()
    uvicorn.run(
        "assistant.server.app:app",
        host=bind,
        port=bind_port,
        log_level="info" if verbose else "warning",
    )


@app.command("prepare-models")
def prepare_models(
    force: bool = typer.Option(False, help="Rebuild even if it already exists"),
) -> None:
    """Convert bge-m3 to safetensors for fast startup (run once).

    The HuggingFace cache ships `pytorch_model.bin`, a 2.2 GB legacy pickle that
    must be fully deserialised on every load. A local safetensors copy is mmapped
    instead: **5.9 s -> 1.4 s**, measured. Costs 2.2 GB of disk.
    """
    _setup_logging(True)
    from assistant.retrieval.embed import make_local_copy

    with console.status("converting bge-m3 to safetensors..."):
        path = make_local_copy(force=force)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
    console.print(f"[green]ready[/] {path} ({size:.1f} GB)")
    console.print("[dim]startup will now load embeddings ~4x faster[/]")


@app.command()
def login(
    url: str = typer.Argument(..., help="e.g. http://workstation.tailnet.ts.net:8080"),
    token: str = typer.Option(..., prompt=True, hide_input=True, help="Server token"),
) -> None:
    """Save server details on this machine (run on the laptop)."""
    from assistant.client import load_config, health, write_config

    path = write_config(url.rstrip("/"), token)
    console.print(f"wrote [bold]{path}[/]")
    ok = asyncio.run(health(load_config()))
    console.print("[green]server reachable[/]" if ok else "[yellow]server not reachable yet[/]")


@app.command()
def remote(
    question: str = typer.Argument(None, help="One-shot question; omit for a REPL"),
    url: str = typer.Option(None, help="Override server URL"),
) -> None:
    """Talk to a remote Evelyn over text (run on the laptop)."""
    from assistant.client import load_config, stream_chat

    cfg = load_config(url=url)
    session = uuid.uuid4().hex

    async def one(text: str) -> None:
        parts: list[str] = []
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            async for delta in stream_chat(cfg, text, session):
                parts.append(delta)
                live.update(Markdown("".join(parts)))

    if question:
        asyncio.run(one(question))
        return

    from prompt_toolkit import PromptSession

    console.print(Panel(f"{BANNER}\n[dim]remote: {cfg.url}[/]", border_style="cyan"))
    prompt = PromptSession()
    while True:
        try:
            text = prompt.prompt("› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        try:
            asyncio.run(one(text))
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/] {exc}")


@app.command()
def talk(
    seconds: float = typer.Option(5.0, "--seconds", "-s", help="How long to record"),
    url: str = typer.Option(None, help="Override server URL"),
    once: bool = typer.Option(False, help="One turn, then exit"),
) -> None:
    """Speak to Evelyn and hear the reply (run on the laptop)."""
    from assistant.client import load_config, voice_turn

    cfg = load_config(url=url)
    console.print(Panel(f"{BANNER}\n[dim]voice · {cfg.url}[/]", border_style="cyan"))

    def on_event(event: dict):
        kind = event.get("type")
        if kind == "recording":
            console.print(f"[bold green]● listening {seconds:.0f}s...[/]")
        elif kind == "thinking":
            console.print("[dim]thinking...[/]")
        elif kind == "transcript":
            console.print(f"[cyan]you:[/] {event.get('text') or '(nothing heard)'}")
        elif kind == "speaking":
            console.print(f"[dim]first audio at {event.get('ttfa_s')}s[/]")
        elif kind == "error":
            console.print(f"[red]{event.get('detail')}[/]")

    while True:
        try:
            result = asyncio.run(voice_turn(cfg, seconds=seconds, on_event=on_event))
        except KeyboardInterrupt:
            console.print("\n[dim]bye[/]")
            break
        if text := result.get("text"):
            console.print(f"[bold]evelyn:[/] {text}")
        if cites := result.get("citations"):
            console.print(f"[dim]{'; '.join(cites[:3])}[/]")
        if total := result.get("total_s"):
            console.print(f"[dim]{total}s total, first audio {result.get('ttfa_s')}s[/]\n")
        if once:
            break
        try:
            input("press enter to speak again, Ctrl-C to quit ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break


@app.command()
def traces(limit: int = typer.Option(10, help="How many recent turns to show")) -> None:
    """Show recent logged turns — the phase-9 training data as it accumulates."""
    from assistant.tracing import TraceStore

    store = TraceStore()
    _print_stats(store.stats())
    table = Table(title=f"Last {limit} turns")
    table.add_column("when", style="dim")
    table.add_column("tools", justify="right")
    table.add_column("s", justify="right")
    table.add_column("question")
    import datetime as _dt

    for row in store.recent(limit):
        when = _dt.datetime.fromtimestamp(row["started_at"]).strftime("%m-%d %H:%M")
        table.add_row(
            when,
            str(row["n_tool_calls"]),
            f"{row['elapsed_s']:.1f}" if row["elapsed_s"] else "-",
            (row["user_input"] or "")[:60],
        )
    console.print(table)
    console.print(f"[dim]{Path(store.path)}[/]")
    store.close()


@app.command()
def index(
    tier: list[str] = typer.Option(["fast"], help="Tiers: fast, datasheets, books"),
    device: str = typer.Option("cuda", help="docling device"),
    embed_device: str = typer.Option("cpu", help="bge-m3 device"),
    force: bool = typer.Option(False, help="Re-index even if unchanged"),
) -> None:
    """Run the ingest pipeline (phase 1)."""
    _setup_logging(True)
    from assistant.ingest.index import build_index

    stats = build_index(
        tuple(tier), device=device, embed_device=embed_device, force=force
    )
    console.print(stats.summary())
    for failure in stats.failures[:20]:
        console.print(f"[yellow]  {failure}[/]")


if __name__ == "__main__":
    app()

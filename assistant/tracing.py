"""Every agent turn, logged to SQLite.

CLAUDE.md is explicit that this is **production code, not debug scaffolding**:
these traces are the fine-tuning dataset (`TRAINING_PLAN.md` §3, which ranks
logged real traces above every synthetic alternative). Two consequences shape
this module:

* **Never let logging break a turn.** A failed write is swallowed and warned
  about. Losing one trace is a rounding error; crashing the assistant because a
  disk filled is not.
* **Log the whole turn, not a summary.** System prompt, retrieved chunks, every
  tool call and result, the final answer. `TRAINING_PLAN.md` §3 needs the exact
  input distribution the model saw at deployment, and anything discarded here
  cannot be recovered later.

Schema note: turns and tool calls are separate tables rather than one blob, so
the dataset builder can filter on `tool_name` or `ok` without parsing JSON for
every row. The full message history is still kept verbatim in `turns.messages`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistant.config import settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    started_at    REAL NOT NULL,
    elapsed_s     REAL,
    model         TEXT,
    mode          TEXT,
    user_input    TEXT NOT NULL,
    final_answer  TEXT,
    system_prompt TEXT,
    messages      TEXT,         -- full pydantic-ai message history, JSON
    retrieval     TEXT,         -- citations + scores, JSON
    n_tool_calls  INTEGER DEFAULT 0,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    ttft_s        REAL,
    error         TEXT,
    corrected     INTEGER DEFAULT 0,   -- set by hand when fixing a bad trace
    exclude       INTEGER DEFAULT 0    -- set by hand to drop from the dataset
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id     TEXT NOT NULL REFERENCES turns(id),
    seq         INTEGER NOT NULL,
    tool_name   TEXT NOT NULL,
    args        TEXT,
    result      TEXT,
    ok          INTEGER NOT NULL DEFAULT 1,
    error       TEXT,
    elapsed_s   REAL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_started ON turns(started_at);
CREATE INDEX IF NOT EXISTS idx_tool_turn ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_calls(tool_name);
"""


def _dumps(value: Any) -> str | None:
    """JSON that never raises. Traces are worth more slightly lossy than absent."""
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover
        return json.dumps({"__unserialisable__": str(type(value))})


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    args: Any = None
    result: Any = None
    ok: bool = True
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass(slots=True)
class TurnRecord:
    session_id: str
    user_input: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    model: str = ""
    mode: str = "cli"
    final_answer: str = ""
    system_prompt: str = ""
    messages: Any = None
    retrieval: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None
    error: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class TraceStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.traces_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def record(self, turn: TurnRecord) -> None:
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO turns
                   (id, session_id, started_at, elapsed_s, model, mode, user_input,
                    final_answer, system_prompt, messages, retrieval, n_tool_calls,
                    input_tokens, output_tokens, ttft_s, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    turn.id,
                    turn.session_id,
                    turn.started_at,
                    turn.elapsed_s,
                    turn.model,
                    turn.mode,
                    turn.user_input,
                    turn.final_answer,
                    turn.system_prompt,
                    _dumps(turn.messages),
                    _dumps(turn.retrieval),
                    len(turn.tool_calls),
                    turn.input_tokens,
                    turn.output_tokens,
                    turn.ttft_s,
                    turn.error,
                ),
            )
            for seq, tc in enumerate(turn.tool_calls):
                self._conn.execute(
                    """INSERT INTO tool_calls
                       (turn_id, seq, tool_name, args, result, ok, error, elapsed_s)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        turn.id,
                        seq,
                        tc.tool_name,
                        _dumps(tc.args),
                        _dumps(tc.result),
                        int(tc.ok),
                        tc.error,
                        tc.elapsed_s,
                    ),
                )
            self._conn.commit()
        except Exception as exc:
            # Deliberately swallowed -- see the module docstring.
            log.warning("trace write failed (turn continues): %s", exc)

    # --- read side, used by the dataset builder and `eve traces` ------

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        return list(
            self._conn.execute(
                "SELECT * FROM turns ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        )

    def stats(self) -> dict[str, Any]:
        cur = self._conn.execute(
            """SELECT count(*), sum(n_tool_calls), avg(elapsed_s),
                      sum(error IS NOT NULL) FROM turns"""
        )
        turns, tools, avg_s, errors = cur.fetchone()
        by_tool = dict(
            self._conn.execute(
                "SELECT tool_name, count(*) FROM tool_calls GROUP BY 1 ORDER BY 2 DESC"
            )
        )
        failed = self._conn.execute(
            "SELECT count(*) FROM tool_calls WHERE ok = 0"
        ).fetchone()[0]
        return {
            "turns": turns or 0,
            "tool_calls": tools or 0,
            "failed_tool_calls": failed,
            "avg_elapsed_s": round(avg_s or 0.0, 2),
            "errors": errors or 0,
            "by_tool": by_tool,
        }

    def close(self) -> None:
        self._conn.close()


@contextmanager
def trace_turn(store: TraceStore, session_id: str, user_input: str, **kw):
    """Wrap one agent turn. The record is written even if the turn raises."""
    turn = TurnRecord(session_id=session_id, user_input=user_input, **kw)
    started = time.perf_counter()
    try:
        yield turn
    except Exception as exc:
        turn.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        turn.elapsed_s = time.perf_counter() - started
        store.record(turn)


__all__ = ["TraceStore", "TurnRecord", "ToolCallRecord", "trace_turn"]

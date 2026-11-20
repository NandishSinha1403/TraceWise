"""SQLite sink for per-call and per-request metrics (read by Grafana in M5)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.tracing.metrics import CallRecord, RequestRollup

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT, strategy TEXT, node TEXT, call_type TEXT, name TEXT,
    started_at TEXT, latency_ms REAL, wall_ms REAL, ttft_ms REAL,
    input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER,
    decode_tps REAL, cost_usd REAL, attempts INTEGER, backoff_ms REAL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_request ON llm_calls(request_id);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, finished_at TEXT, strategy TEXT,
    total_latency_ms REAL, total_cost_usd REAL, cost_complete INTEGER,
    n_llm_calls INTEGER, n_embed_calls INTEGER, n_tool_calls INTEGER,
    n_agent_hops INTEGER, n_critic_retries INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER,
    llm_latency_ms REAL, backoff_ms REAL, n_retried_calls INTEGER, errors TEXT
);
"""

CALL_COLUMNS = [
    "request_id", "strategy", "node", "call_type", "name", "started_at", "latency_ms", "wall_ms",
    "ttft_ms", "input_tokens", "output_tokens", "thinking_tokens", "decode_tps", "cost_usd",
    "attempts", "backoff_ms", "error",
]


class MetricsStore:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert_call(self, rec: CallRecord) -> None:
        d = rec.as_dict()
        placeholders = ",".join("?" for _ in CALL_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO llm_calls ({','.join(CALL_COLUMNS)}) VALUES ({placeholders})",
                [d[c] for c in CALL_COLUMNS],
            )
            self._conn.commit()

    def insert_request(self, roll: RequestRollup) -> None:
        d = roll.as_dict()
        d["errors"] = json.dumps(d["errors"])
        d["cost_complete"] = int(d["cost_complete"])
        d["finished_at"] = datetime.now(timezone.utc).isoformat()
        cols = list(d)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO requests ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                [d[c] for c in cols],
            )
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

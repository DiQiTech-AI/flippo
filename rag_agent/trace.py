from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import Database
from .ids import new_id


SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(api[_-]?key|authorization|cookie|password|secret|access[_-]?token|refresh[_-]?token)(?:$|[_-])",
    re.I,
)
SECRET_VALUE = re.compile(r"(?i)(bearer\s+)?(sk-|AIza)[A-Za-z0-9_.-]{8,}")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[REDACTED]", value)
    return value


class TraceRecorder:
    def __init__(self, db: Database, trace_dir: Path):
        self.db = db
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _lock(self, run_id: str) -> threading.Lock:
        with self._global_lock:
            return self._locks.setdefault(run_id, threading.Lock())

    def record(
        self,
        run_id: str,
        event_type: str,
        payload: Any,
        *,
        status: str = "success",
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        safe = redact(payload)
        now = datetime.now(UTC).isoformat()
        with self._lock(run_id):
            row = self.db.one("SELECT COALESCE(MAX(sequence), 0) AS n FROM trace_events WHERE run_id=?", (run_id,))
            sequence = int(row["n"]) + 1 if row else 1
            event = {
                "schema_version": "1.0",
                "event_id": new_id("evt"),
                "run_id": run_id,
                "sequence": sequence,
                "timestamp": now,
                "event_type": event_type,
                "duration_ms": duration_ms,
                "status": status,
                "payload": safe,
            }
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO trace_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event["event_id"], run_id, sequence, now, event_type,
                        duration_ms, status, json.dumps(safe, ensure_ascii=False),
                    ),
                )
            with (self.trace_dir / f"{run_id}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def list(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM trace_events WHERE run_id=? ORDER BY sequence", (run_id,))
        return [self.db.decode_row(row) or {} for row in rows]

    def jsonl(self, run_id: str) -> str:
        path = self.trace_dir / f"{run_id}.jsonl"
        return path.read_text(encoding="utf-8") if path.exists() else ""

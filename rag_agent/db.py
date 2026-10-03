from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .lexical import fts_index_text


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT NOT NULL,
  version_id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  filename TEXT NOT NULL,
  media_type TEXT NOT NULL,
  parser_name TEXT,
  parser_version TEXT,
  status TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'uploaded',
  page_count INTEGER,
  character_count INTEGER DEFAULT 0,
  chunk_count INTEGER DEFAULT 0,
  embedding_model TEXT,
  embedding_dimension INTEGER,
  file_path TEXT NOT NULL,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(sha256, embedding_model)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES documents(version_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  heading_path TEXT NOT NULL DEFAULT '[]',
  page_start INTEGER,
  page_end INTEGER,
  source_start INTEGER,
  source_end INTEGER,
  raw_text TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  UNIQUE(version_id, ordinal)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  version_id UNINDEXED,
  heading_text,
  normalized_text,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  version_id TEXT NOT NULL REFERENCES documents(version_id),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
  message_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  run_id TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  version_id TEXT NOT NULL REFERENCES documents(version_id),
  request_type TEXT NOT NULL,
  user_message TEXT NOT NULL,
  status TEXT NOT NULL,
  agent_mode TEXT NOT NULL,
  model TEXT,
  answer TEXT,
  report_json TEXT,
  citations_json TEXT NOT NULL DEFAULT '[]',
  stop_reason TEXT,
  tool_calls INTEGER NOT NULL DEFAULT 0,
  search_rounds INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  version_id TEXT NOT NULL REFERENCES documents(version_id),
  chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
  raw_text TEXT NOT NULL,
  context_text TEXT NOT NULL,
  location_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trace_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  event_type TEXT NOT NULL,
  duration_ms INTEGER,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS schema_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
        self._migrate_fts_cjk_features()

    def _migrate_fts_cjk_features(self) -> None:
        migration_key = "fts_cjk_bigrams_v1"
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            applied = conn.execute("SELECT 1 FROM schema_metadata WHERE key=?", (migration_key,)).fetchone()
            if applied:
                conn.commit()
                return
            conn.execute("DELETE FROM chunks_fts")
            rows = conn.execute(
                "SELECT chunk_id, version_id, heading_path, normalized_text FROM chunks ORDER BY version_id, ordinal"
            ).fetchall()
            for row in rows:
                try:
                    headings = json.loads(row["heading_path"] or "[]")
                except json.JSONDecodeError:
                    headings = []
                conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, version_id, heading_text, normalized_text) VALUES (?, ?, ?, ?)",
                    (
                        row["chunk_id"],
                        row["version_id"],
                        fts_index_text(" > ".join(str(item) for item in headings)),
                        fts_index_text(row["normalized_text"]),
                    ),
                )
            conn.execute("INSERT INTO schema_metadata(key, value) VALUES (?, '1')", (migration_key,))
            conn.commit()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    @staticmethod
    def decode_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return row
        result = dict(row)
        for key in ("heading_path", "location_json", "report_json", "citations_json", "payload_json"):
            if key in result and isinstance(result[key], str):
                try:
                    result[key.removesuffix("_json")] = json.loads(result[key])
                except json.JSONDecodeError:
                    pass
        return result

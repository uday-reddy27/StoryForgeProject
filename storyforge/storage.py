"""SQLite history of generated stories (standard library only - no server to install)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .text_utils import join_story

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    prompt        TEXT    NOT NULL,
    genre         TEXT,
    continuation  TEXT    NOT NULL,
    params        TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    duration_ms   INTEGER NOT NULL
);
"""


class StoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A new connection per call keeps this safe with Flask's threads.
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "prompt": row["prompt"],
            "genre": row["genre"],
            "continuation": row["continuation"],
            "story": join_story(row["prompt"], row["continuation"]),
            "params": json.loads(row["params"]),
            "model": row["model"],
            "duration_ms": row["duration_ms"],
        }

    def add(self, *, prompt: str, genre: Optional[str], continuation: str,
            params: dict, model: str, duration_ms: int) -> int:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:  # `with conn` commits
            cur = conn.execute(
                "INSERT INTO stories (created_at, prompt, genre, continuation, params, model, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (created, prompt, genre, continuation, json.dumps(params), model, duration_ms),
            )
            return int(cur.lastrowid)

    def list(self, limit: int = 30) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM stories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_dict(r) for r in rows]

    def get(self, story_id: int) -> Optional[dict]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        return self._to_dict(row) if row else None

    def delete(self, story_id: int) -> bool:
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM stories WHERE id = ?", (story_id,)).rowcount > 0

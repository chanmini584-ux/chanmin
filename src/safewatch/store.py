"""SQLite persistence for incidents (JSON documents) and operator actions (audit log)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, doc TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT,
                action TEXT, actor TEXT, detail TEXT, at TEXT);
            """)
            self.db.commit()

    def save(self, inc: dict) -> None:
        doc = json.dumps({k: v for k, v in inc.items() if not k.startswith("_")}, ensure_ascii=False)
        with self.lock:
            self.db.execute("INSERT INTO incidents(id, doc, updated) VALUES(?,?,?) "
                            "ON CONFLICT(id) DO UPDATE SET doc=excluded.doc, updated=excluded.updated",
                            (inc["id"], doc, datetime.now().astimezone().isoformat()))
            self.db.commit()

    def load_all(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT doc FROM incidents ORDER BY updated").fetchall()
        return [json.loads(r[0]) for r in rows]

    def audit(self, incident_id: str, action: str, actor: str = "", detail: dict | None = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO audit(incident_id, action, actor, detail, at) VALUES(?,?,?,?,?)",
                            (incident_id, action, actor, json.dumps(detail or {}, ensure_ascii=False),
                             datetime.now().astimezone().isoformat(timespec="milliseconds")))
            self.db.commit()

    def audit_log(self, incident_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT action, actor, detail, at FROM audit WHERE incident_id=? ORDER BY id",
                                   (incident_id,)).fetchall()
        return [{"action": a, "actor": b, "detail": json.loads(c), "at": d} for a, b, c, d in rows]

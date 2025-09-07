import os, sqlite3
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "app.db")

def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cls TEXT NOT NULL,
            score REAL NOT NULL,
            ok INTEGER NOT NULL,
            recipients TEXT,
            payload TEXT
        )""")
        conn.commit()

def insert_alert(ts: str, cls: str, score: float, ok: bool, recipients: str, payload: str):
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO alerts (ts, cls, score, ok, recipients, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, cls, score, 1 if ok else 0, recipients, payload),
        )
        conn.commit()

def recent_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT ts, cls, score, ok, recipients, payload FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    out = []
    for ts, cls, score, ok, rcpts, payload in rows:
        out.append({"ts": ts, "class": cls, "score": score, "ok": bool(ok), "recipients": rcpts, "payload": payload})
    return out

"""Idempotent event logging simulator (Days 10-11).

In production: events written append-only to Postgres with
	client event_id as unique key (idempotent against retries).
Here we test that property.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path("/tmp/experiments_test.db")


def init_db():
    """Create the events table with event_id as primary key."""
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            variant TEXT,
            user_id TEXT,
            metric TEXT,
            value REAL,
            ts TEXT
        )""")
        conn.commit()


def log_event(event_id, experiment_id, variant, user_id, metric, value, ts):
    """Idempotent: retries with same event_id are no-op."""
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        try:
            conn.execute(
                "INSERT INTO events (event_id, experiment_id, variant, user_id, metric, value, ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (event_id, experiment_id, variant, user_id, metric, value, ts),
            )
        except sqlite3.IntegrityError:
            # DB constraint violation -> idempotent retry detected; we silently ignore
            pass
        conn.commit()


def query_events():
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]


# Test the property
init_db()
log_event("e1", "exp1", "control", "u1", "thumbs_up", 1.0, "t1")
log_event("e1", "exp1", "control", "u1", "thumbs_up", 1.0, "t1")  # retry, idempotent
log_event("e2", "exp1", "treatment", "u2", "thumbs_up", 0.0, "t2")

assert query_events() == 2, "idempotent logging should only store 2 unique events"
print("Idempotent logging: 2 events stored, retry of e1 ignored [PASS]")
print("Day 11 idempotent logging validated")

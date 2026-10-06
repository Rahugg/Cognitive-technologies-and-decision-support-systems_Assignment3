from collections import defaultdict, deque
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row


class Memory:
    """Session deque plus PostgreSQL metadata history; image pixels are never stored."""

    def __init__(self, database_url, window=5):
        self.database_url = database_url
        self.sessions = defaultdict(lambda: deque(maxlen=window))
        self._initialize()

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    quality DOUBLE PRECISION NOT NULL,
                    relevance DOUBLE PRECISION NOT NULL,
                    accepted BOOLEAN NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY,
                    value DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

    def previous(self, session_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT emotion, confidence, created_at FROM predictions "
                "WHERE session_id = %s AND accepted = TRUE ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row and row["created_at"]:
            row["created_at"] = row["created_at"].isoformat()
        return row

    def recent(self, session_id):
        return list(self.sessions[session_id])

    def record(self, session_id, perception, attention, outcome):
        record = {
            "emotion": perception["emotion"],
            "confidence": float(perception["confidence"]),
            "quality": float(perception["quality_score"]),
            "relevance": float(attention["relevance"]),
            "accepted": bool(attention["accepted"]),
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.sessions[session_id].append(record)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(session_id, emotion, confidence, quality, relevance, accepted) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, record["emotion"], record["confidence"], record["quality"],
                 record["relevance"], record["accepted"]),
            )
        return record

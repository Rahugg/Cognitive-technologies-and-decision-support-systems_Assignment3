from collections import defaultdict, deque
from datetime import datetime, timezone

import copy
import logging

import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

from .config import UTILITY

log = logging.getLogger(__name__)


class Memory:
    """Session deque plus PostgreSQL metadata history; image pixels are never stored."""

    def __init__(self, database_url, window=5):
        self.database_url = database_url
        self.sessions = defaultdict(lambda: deque(maxlen=window))
        self.utility = copy.deepcopy(UTILITY)  # matrix M; Learning may update it
        self.decisions = []  # in-memory copy of decision records (always kept)
        try:
            self._initialize()
            self.db_ok = True
        except psycopg.Error as error:
            self.db_ok = False
            log.warning("PostgreSQL unavailable (%s); using in-memory storage only.", error)

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
                CREATE TABLE IF NOT EXISTS decisions (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    recommended TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    reliability DOUBLE PRECISION NOT NULL,
                    trend TEXT NOT NULL,
                    accepted BOOLEAN NOT NULL,
                    ranking JSONB NOT NULL,
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
        if not self.db_ok:
            accepted = [r for r in self.sessions[session_id] if r["accepted"]]
            return ({"emotion": accepted[-1]["emotion"], "confidence": accepted[-1]["confidence"],
                     "created_at": accepted[-1]["timestamp"]} if accepted else None)
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

    def smooth_prediction(self, session_id, perception, accepted):
        """Average the last accepted frame probabilities for a short video context."""
        if not accepted:
            return {key: perception[key] for key in ("emotion", "confidence", "probabilities")}

        history = [item["probabilities"] for item in self.sessions[session_id]
                   if item["accepted"] and "probabilities" in item]
        history.append(perception["probabilities"])
        history = history[-self.sessions[session_id].maxlen:]
        count = len(history)
        probabilities = {
            emotion: sum(frame[emotion] for frame in history) / count
            for emotion in perception["probabilities"]
        }
        emotion = max(probabilities, key=probabilities.get)
        return {"emotion": emotion, "confidence": probabilities[emotion],
                "probabilities": probabilities}

    def record(self, session_id, perception, attention, outcome):
        record = {
            "emotion": perception["emotion"],
            "confidence": float(perception["confidence"]),
            "quality": float(perception["quality_score"]),
            "relevance": float(attention["relevance"]),
            "accepted": bool(attention["accepted"]),
            "probabilities": perception["probabilities"],
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.sessions[session_id].append(record)
        if not self.db_ok:
            return record
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(session_id, emotion, confidence, quality, relevance, accepted) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, record["emotion"], record["confidence"], record["quality"],
                 record["relevance"], record["accepted"]),
            )
        return record

    def save_decision(self, session_id, record):
        """Store one decision record (metadata only, no images)."""
        self.decisions.append({"session_id": session_id, **record})
        if not self.db_ok:
            return False
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO decisions (session_id, recommended, emotion, reliability, "
                "trend, accepted, ranking) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (session_id, record["recommended"], record["emotion"], record["reliability"],
                 record["trend"], record["accepted"], Jsonb(record["ranking"])),
            )
        return True

from collections import defaultdict, deque
from datetime import datetime, timezone

import copy
import logging

import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

from .config import ALTERNATIVES, EMOTIONS, UTILITY

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc).isoformat()


class Memory:
    """Session deque plus PostgreSQL metadata history; image pixels are never stored."""

    def __init__(self, database_url, window=5):
        self.database_url = database_url
        self.sessions = defaultdict(lambda: deque(maxlen=window))
        self.utility = copy.deepcopy(UTILITY)  # matrix M; Learning may update it
        self.decisions = []  # in-memory copy of decision records (always kept)
        self.feedback = []  # in-memory copy of tutor feedback (always kept)
        self.utility_history = []  # in-memory copy of matrix versions (always kept)
        self.last_decision_id = None  # id of the most recent save_decision()
        self.db_ok = False
        if not self.database_url:
            return
        try:
            self._initialize()
            self._load_latest_utility()
            self.db_ok = True
        except psycopg.Error as error:
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
            # --- Learning / feedback (Role B) ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id BIGSERIAL PRIMARY KEY,
                    decision_id BIGINT REFERENCES decisions(id),
                    session_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    recommended TEXT NOT NULL,
                    selected TEXT NOT NULL,
                    emotion_correct BOOLEAN NOT NULL,
                    processed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS utility_versions (
                    id BIGSERIAL PRIMARY KEY,
                    matrix JSONB NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

    def _load_latest_utility(self):
        """Restore the most recently saved matrix M, if Learning has saved one."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT matrix FROM utility_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            self.utility = row["matrix"]

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
            "timestamp": _now(),
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
        """Store one decision record (metadata only, no images).

        Returns True if written to PostgreSQL, False if kept in memory only.
        The id of the new decision is available as self.last_decision_id.
        """
        entry = {"session_id": session_id, **record}
        if not self.db_ok:
            entry["id"] = len(self.decisions) + 1
            self.decisions.append(entry)
            self.last_decision_id = entry["id"]
            return False
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO decisions (session_id, recommended, emotion, reliability, "
                "trend, accepted, ranking) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (session_id, record["recommended"], record["emotion"] or "unavailable", record["reliability"],
                 record["trend"], record["accepted"], Jsonb(record["ranking"])),
            ).fetchone()
        entry["id"] = row["id"]
        self.decisions.append(entry)
        self.last_decision_id = row["id"]
        return True

    # ------------------------------------------------------------------
    # Tutor feedback (Role B)
    # ------------------------------------------------------------------
    def save_feedback(self, session_id, emotion, recommended, selected,
                      emotion_correct, decision_id=None):
        """Store one piece of tutor feedback and return it as a dict.

        emotion: expression estimated by the system, or None for a rejected frame.
        recommended: action the system recommended.
        selected: action the tutor actually chose.
        emotion_correct: did the tutor agree with the recognized emotion?
        """
        if emotion not in EMOTIONS and (emotion is not None or emotion_correct):
            raise ValueError(f"Unknown emotion: {emotion!r}")
        if recommended not in ALTERNATIVES:
            raise ValueError(f"Unknown recommended action: {recommended!r}")
        if selected not in ALTERNATIVES:
            raise ValueError(f"Unknown selected action: {selected!r}")

        record = {
            "decision_id": decision_id,
            "session_id": session_id,
            "emotion": emotion,
            "recommended": recommended,
            "selected": selected,
            "emotion_correct": bool(emotion_correct),
            "processed": False,
            "created_at": _now(),
        }
        if self.db_ok:
            with self._connect() as conn:
                row = conn.execute(
                    "INSERT INTO feedback (decision_id, session_id, emotion, recommended, "
                    "selected, emotion_correct) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (decision_id, session_id, emotion or "unavailable", recommended, selected,
                     record["emotion_correct"]),
                ).fetchone()
            record["id"] = row["id"]
        else:
            record["id"] = len(self.feedback) + 1
        self.feedback.append(record)
        return record

    def pending_feedback(self):
        """Feedback not yet used by Learning, oldest first."""
        if not self.db_ok:
            return [dict(r) for r in self.feedback if not r["processed"]]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, decision_id, session_id, emotion, recommended, selected, "
                "emotion_correct, processed, created_at FROM feedback "
                "WHERE processed = FALSE ORDER BY id"
            ).fetchall()
        for row in rows:
            row["created_at"] = row["created_at"].isoformat()
            if row["emotion"] == "unavailable":
                row["emotion"] = None
        return rows

    def complete_feedback_batch(self, feedback_ids, matrix=None, reason=""):
        """Persist the new utility version and processed flags in the same transaction."""
        ids = list(feedback_ids)
        if not ids:
            return None
        version = None
        if self.db_ok:
            with self._connect() as conn:
                if matrix is not None:
                    version = self._insert_utility_version(conn, matrix, reason)
                conn.execute("UPDATE feedback SET processed = TRUE WHERE id = ANY(%s)", (ids,))
        if matrix is not None:
            self.utility = matrix
            version = self._record_utility_version(matrix, reason, version)
        for item in self.feedback:
            if item["id"] in ids:
                item["processed"] = True
        return version

    # ------------------------------------------------------------------
    # Utility matrix versions (Role B)
    # ------------------------------------------------------------------
    def save_utility_version(self, reason=""):
        """Snapshot the current matrix M and return its version number."""
        version = None
        if self.db_ok:
            with self._connect() as conn:
                version = self._insert_utility_version(conn, self.utility, reason)
        return self._record_utility_version(self.utility, reason, version)

    @staticmethod
    def _insert_utility_version(conn, matrix, reason):
        return conn.execute(
            "INSERT INTO utility_versions (matrix, reason) VALUES (%s, %s) RETURNING id",
            (Jsonb(matrix), reason),
        ).fetchone()["id"]

    def _record_utility_version(self, matrix, reason, version):
        if version is None:
            version = len(self.utility_history) + 1
        self.utility_history.append(
            {"version": version, "matrix": copy.deepcopy(matrix), "reason": reason, "created_at": _now()})
        return version

    def reset_utility(self, reason="reset to expert matrix"):
        """Restore the original expert utility matrix and save a version."""
        self.utility = copy.deepcopy(UTILITY)
        return self.save_utility_version(reason)

    def utility_versions(self):
        """All saved versions of matrix M, oldest first."""
        if not self.db_ok:
            return [dict(v) for v in self.utility_history]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, matrix, reason, created_at FROM utility_versions ORDER BY id"
            ).fetchall()
        return [{"version": r["id"], "matrix": r["matrix"], "reason": r["reason"],
                 "created_at": r["created_at"].isoformat()} for r in rows]

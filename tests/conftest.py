import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.emotion_system.config import EMOTIONS, UTILITY
from src.emotion_system.state import CogState


def peaked(emotion: str, p: float = 0.9) -> dict[str, float]:
    rest = (1 - p) / (len(EMOTIONS) - 1)
    return {e: p if e == emotion else rest for e in EMOTIONS}


class FakeMemory:
    """Stands in for Memory: short-term frames and decision records, no database."""

    def __init__(self, frames=()):
        self.frames = list(frames)
        self.utility = copy.deepcopy(UTILITY)
        self.saved = []

    def recent(self, session_id):
        return self.frames

    def save_decision(self, session_id, record):
        self.saved.append((session_id, record))
        return True


def frame(emotion: str, accepted: bool = True, p: float = 0.9) -> dict:
    return {"accepted": accepted, "probabilities": peaked(emotion, p), "emotion": emotion}


@pytest.fixture
def make_state():
    def _make(emotion="happy", p=0.9, quality=0.8, accepted=True, previous=None):
        probs = peaked(emotion, p)
        return CogState(session_id="t", emotion=emotion, probs=probs, confidence=p, quality=quality,
                        face_found=True, accepted=accepted, reject_reason="" if accepted else "low quality",
                        previous_emotion=previous)
    return _make

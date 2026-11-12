"""Shared state passed through the seven cognitive functions."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CogState:
    session_id: str
    image: Any = None  # transient input frame; never persisted
    # Perception
    emotion: str = ""
    probs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    quality: float = 0.0
    face_found: bool = False
    brightness: float = 0.0
    sharpness: float = 0.0
    # Attention
    accepted: bool = False
    reject_reason: str = ""
    relevance: float = 0.0
    # Memory
    previous_emotion: str | None = None
    # Reasoning
    smoothed_probs: dict[str, float] = field(default_factory=dict)
    smoothed_confidence: float = 0.0
    trend: str = "insufficient_history"
    reasoning_steps: list[str] = field(default_factory=list)
    label: str = ""  # R1/R3/R4 outcome: insufficient_data | uncertain | <emotion>
    rules: list[str] = field(default_factory=list)
    # Decision
    alternatives: list[str] = field(default_factory=list)
    criteria_matrix: dict[str, Any] = field(default_factory=dict)
    ranking: list[dict[str, Any]] = field(default_factory=list)
    recommended: str = ""
    # Action
    action: str = ""  # recommendation text for the tutor
    report: dict[str, Any] | None = None
    # Shared trace: every module appends one short reason
    explanation: list[str] = field(default_factory=list)

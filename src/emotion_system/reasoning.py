"""Reasoning (STUB): copies perception into the smoothed fields."""
from .state import CogState


def run(state: CogState, memory) -> CogState:
    state.smoothed_probs = dict(state.probs)
    state.smoothed_confidence = state.confidence
    state.label = state.emotion if state.accepted else "insufficient_data"
    state.explanation.append("Reasoning: stub, no smoothing yet.")
    return state

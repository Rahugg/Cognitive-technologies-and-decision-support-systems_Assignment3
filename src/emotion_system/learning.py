"""Learning (STUB, Role B): will update utility matrix M from tutor feedback."""
from .state import CogState


def run(state: CogState, memory) -> CogState:
    state.explanation.append("Learning: no feedback recorded yet (stub).")
    return state

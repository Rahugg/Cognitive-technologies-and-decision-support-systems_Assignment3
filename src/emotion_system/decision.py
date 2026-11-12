"""Decision (STUB): recommends 'ask_directly' when rejected, else 'continue'."""
from .state import CogState


def run(state: CogState, memory) -> CogState:
    state.recommended = "continue" if state.accepted else "ask_directly"
    state.explanation.append(f"Decision: stub -> {state.recommended}.")
    return state

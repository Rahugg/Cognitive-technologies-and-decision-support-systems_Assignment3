"""Action (STUB): text only, no database record yet."""
from .state import CogState


def run(state: CogState, memory) -> CogState:
    state.action = f"Suggested tutor action: {state.recommended}"
    state.report = {"session_id": state.session_id, "recommended": state.recommended}
    state.explanation.append("Action: stub recommendation.")
    return state

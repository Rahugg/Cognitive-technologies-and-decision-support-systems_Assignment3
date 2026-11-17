"""Decision: builds the 5x4 criteria matrix and ranks alternatives with the DSS model."""
from dss.model import rank

from .config import (ALTERNATIVES, CRITERIA, CRITERIA_TYPES, DEFAULT_WEIGHTS, DISRUPTION,
                     EMOTIONS, TIME_COST_MIN, UNSTABLE_RELIABILITY_FACTOR)
from .state import CogState


def needs_check_in(state: CogState) -> bool:
    return not state.accepted or state.label == "uncertain"


def reliability(state: CogState) -> float:
    """Zero without a face. Otherwise, rejected frames use the weaker of the smoothed and the
    current confidence; an unstable trend scales r by UNSTABLE_RELIABILITY_FACTOR."""
    if not state.face_found:
        return 0.0
    confidence = state.smoothed_confidence
    if not state.accepted:
        confidence = min(confidence, state.confidence)
    r = confidence * state.quality
    if state.trend == "changing":
        r *= UNSTABLE_RELIABILITY_FACTOR
    return max(0.0, min(1.0, r))


def run(state: CogState, memory) -> CogState:
    r = reliability(state)
    matrix = []
    for j, alt in enumerate(ALTERNATIVES):
        fit = sum(state.smoothed_probs[e] * memory.utility[e][j] for e in EMOTIONS)  # C1
        rel = 1 - r if alt == "ask_directly" else r  # C2
        matrix.append([fit, rel, DISRUPTION[alt], TIME_COST_MIN[alt]])  # C3, C4
    result = rank(matrix, DEFAULT_WEIGHTS, CRITERIA_TYPES)
    state.alternatives = list(ALTERNATIVES)
    state.criteria_matrix = {"criteria": CRITERIA, "types": CRITERIA_TYPES,
                             "weights": list(DEFAULT_WEIGHTS), "reliability": r,
                             "values": dict(zip(ALTERNATIVES, matrix))}
    state.ranking = [{"alternative": ALTERNATIVES[i], "rank": result["ranks"][i],
                      "score": result["scores"][i],
                      "contributions": dict(zip(CRITERIA, result["contributions"][i]))}
                     for i in result["order"]]
    first = state.ranking[0]["alternative"]
    state.recommended = "ask_directly" if needs_check_in(state) else first
    state.explanation.append(f"Decision: WSM ranks {first} first. Guidance is {state.recommended} "
                             f"(reliability {r:.2f}, trend {state.trend}).")
    return state

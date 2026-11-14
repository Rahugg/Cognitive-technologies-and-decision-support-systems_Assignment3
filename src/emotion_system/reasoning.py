"""Reasoning: recency-weighted smoothing, R1-R4 rules and trend, with visible steps."""
from .config import (EMOTIONS, R2_BONUS, RECENCY_DECAY, TREND_L1_THRESHOLD,
                     UNCERTAIN_CONFIDENCE)
from .knowledge import Knowledge
from .state import CogState

_GRAPH_ACTIONS = Knowledge().actions  # R4 mapping from the knowledge graph


def _smooth(frames: list[dict[str, float]]) -> tuple[dict[str, float], list[float]]:
    raw = [RECENCY_DECAY ** (len(frames) - 1 - i) for i in range(len(frames))]  # newest = 1
    weights = [w / sum(raw) for w in raw]
    return {e: sum(w * f[e] for w, f in zip(weights, frames)) for e in EMOTIONS}, weights


def run(state: CogState, memory) -> CogState:
    recent = memory.recent(state.session_id)
    # Only Attention-accepted frames are evidence; Memory never overrides a rejection.
    frames = [r["probabilities"] for r in recent if r["accepted"]]
    steps = [f"1. Window: {len(frames)} accepted of {len(recent)} stored frame(s)."]
    if frames:
        smoothed, weights = _smooth(frames)
        steps.append("2. Recency weights (old->new): " + ", ".join(f"{w:.2f}" for w in weights))
    else:
        smoothed = {e: 1 / len(EMOTIONS) for e in EMOTIONS}
        steps.append("2. No accepted history: uniform probabilities (nothing to trust).")

    top = max(smoothed, key=smoothed.get)
    if state.accepted and state.previous_emotion == top:  # R2 on smoothed probs
        old = smoothed[top]
        new = min(1.0, old + R2_BONUS)
        scale = (1 - new) / (1 - old) if old < 1 else 0.0
        smoothed = {e: new if e == top else p * scale for e, p in smoothed.items()}
        steps.append(f"3. R2: previous accepted emotion '{top}' repeats -> {old:.2f} + {R2_BONUS} = {new:.2f}.")
        state.rules.append("R2: repeated emotion -> +0.05 on smoothed probability")
    else:
        steps.append("3. R2: not applied (no repetition or frame rejected).")

    if len(frames) < 2:
        state.trend = "insufficient_history"
        steps.append("4. Trend: insufficient_history (need >= 2 accepted frames).")
    else:
        l1 = sum(abs(frames[-1][e] - sum(f[e] for f in frames) / len(frames)) for e in EMOTIONS)
        state.trend = "changing" if l1 > TREND_L1_THRESHOLD else "stable"
        steps.append(f"4. Trend: newest frame vs window mean L1 = {l1:.2f} "
                     f"(threshold {TREND_L1_THRESHOLD}) -> {state.trend}.")

    state.smoothed_probs, state.smoothed_confidence = smoothed, smoothed[top]
    if not state.accepted:
        state.label = "insufficient_data"
        state.rules.append("R1: rejected by Attention -> request_more_data")
    elif state.smoothed_confidence < UNCERTAIN_CONFIDENCE:
        state.label = "uncertain"
        state.rules.append(f"R3: smoothed confidence {state.smoothed_confidence:.2f} < "
                           f"{UNCERTAIN_CONFIDENCE} -> uncertain")
    else:
        state.label = top
        state.rules.append(f"R4: accepted '{top}' -> graph action {_GRAPH_ACTIONS[top]}")
    steps.append(f"5. Outcome: {state.rules[-1]}")
    state.reasoning_steps = steps
    state.explanation.append(f"Reasoning: {state.label}, trend {state.trend}, "
                             f"smoothed {top} {state.smoothed_confidence:.0%}.")
    return state

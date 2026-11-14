"""Action: tutor recommendation text, decision record in PostgreSQL, session report."""
from datetime import datetime, timezone

from .state import CogState

LIMITATION = ("The system classifies facial expressions, not internal states. "
              "It is decision support: the tutor makes the final call.")


def run(state: CogState, memory) -> CogState:
    best = state.ranking[0]
    top = sorted(best["contributions"].items(), key=lambda kv: -kv[1])[:2]
    why = ", ".join(f"{name} (+{value:.2f})" for name, value in top)
    reliability = state.criteria_matrix["reliability"]
    text = f"Recommended: {best['alternative']}. Main reasons: {why}."
    if not state.accepted:
        ask = next(r["rank"] for r in state.ranking if r["alternative"] == "ask_directly")
        text = (f"Current frame is unreliable ({state.reject_reason}), so the ranking leans on "
                f"earlier frames. " + text + f" Asking the student directly is ranked #{ask}.")
    elif state.label == "uncertain":
        text = "The expression is unclear; confirm with the student. " + text
    state.action = text
    record = {"recommended": best["alternative"], "reliability": reliability,
              "emotion": max(state.smoothed_probs, key=state.smoothed_probs.get),
              "trend": state.trend, "accepted": state.accepted,
              "ranking": [{"alternative": r["alternative"], "score": round(r["score"], 4)}
                          for r in state.ranking]}
    written = memory.save_decision(state.session_id, record)
    state.explanation.append(f"Action: {'recorded in PostgreSQL' if written else 'kept in memory only (no DB)'}.")
    state.report = {**record, "session_id": state.session_id, "text": text, "label": state.label,
                    "rules": state.rules, "reasoning_steps": state.reasoning_steps,
                    "explanation": state.explanation, "db_written": written,
                    "limitation": LIMITATION,
                    "created_at": datetime.now(timezone.utc).isoformat()}
    return state

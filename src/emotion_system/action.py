"""Action: tutor recommendation text, decision record in PostgreSQL, session report."""
from datetime import datetime, timezone

from .decision import needs_check_in
from .state import CogState

LIMITATION = ("The system classifies facial expressions, not internal states. "
              "It is decision support: the tutor makes the final call.")


def run(state: CogState, memory) -> CogState:
    recommended = next(row for row in state.ranking if row["alternative"] == state.recommended)
    reliability = state.criteria_matrix["reliability"]
    text = f"Recommended: {state.recommended}."
    if needs_check_in(state):
        reason = (f"The frame is unreliable ({state.reject_reason})." if not state.accepted
                  else "The smoothed expression is unclear.")
        text += f" {reason} Asking directly is ranked #{recommended['rank']} by WSM."
    else:
        top = sorted(recommended["contributions"].items(), key=lambda kv: -kv[1])[:2]
        why = ", ".join(f"{name} (+{value:.2f})" for name, value in top)
        text += f" Main reasons: {why}."
    state.action = text
    record = {"recommended": state.recommended, "reliability": reliability,
              "emotion": max(state.smoothed_probs, key=state.smoothed_probs.get) if state.accepted else None,
              "trend": state.trend, "accepted": state.accepted,
              "ranking": [{"alternative": r["alternative"], "score": round(r["score"], 4)}
                          for r in state.ranking]}
    written = memory.save_decision(state.session_id, record)
    decision_id = getattr(memory, "last_decision_id", None)
    state.explanation.append(f"Action: {'recorded in PostgreSQL' if written else 'kept in memory only (no DB)'}.")
    state.report = {**record, "session_id": state.session_id, "text": text, "label": state.label,
                    "rules": state.rules, "reasoning_steps": state.reasoning_steps,
                    "explanation": state.explanation, "db_written": written, "decision_id": decision_id,
                    "limitation": LIMITATION,
                    "created_at": datetime.now(timezone.utc).isoformat()}
    return state

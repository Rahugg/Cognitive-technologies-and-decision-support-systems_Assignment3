
"""Learning (Role B): updates utility matrix M from tutor feedback."""

from .config import ALTERNATIVES
from .state import CogState

LEARNING_RATE = 0.05


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply_feedback(memory, learning_rate: float = LEARNING_RATE) -> dict:
    """Consume pending tutor feedback and update utility matrix M."""
    pending = memory.pending_feedback()
    updated = skipped = 0

    for item in pending:
        if not item["emotion_correct"]:
            skipped += 1
            continue

        row = memory.utility[item["emotion"]]

        chosen = ALTERNATIVES.index(item["selected"])
        row[chosen] = round(_clamp(row[chosen] + learning_rate), 4)

        if item["recommended"] != item["selected"]:
            rejected = ALTERNATIVES.index(item["recommended"])
            row[rejected] = round(
                _clamp(row[rejected] - learning_rate), 4
            )

        updated += 1

    version = None

    if pending:
        memory.mark_feedback_processed(
            [item["id"] for item in pending]
        )

    if updated:
        version = memory.save_utility_version(
            f"{updated} tutor feedback record(s)"
        )

    return {
        "processed": len(pending),
        "updated": updated,
        "skipped": skipped,
        "version": version,
    }


def submit_feedback(
    memory,
    session_id,
    emotion,
    recommended,
    selected,
    emotion_correct,
    decision_id=None,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """Save tutor feedback and apply its learning update immediately."""

    record = memory.save_feedback(
        session_id,
        emotion,
        recommended,
        selected,
        emotion_correct,
        decision_id=decision_id,
    )

    result = apply_feedback(memory, learning_rate)

    return {
        "feedback": record,
        **result,
    }


def run(state: CogState, memory) -> CogState:
    if not hasattr(memory, "pending_feedback"):
        state.explanation.append(
            "Learning: memory has no feedback storage."
        )
        return state

    result = apply_feedback(memory)

    if result["processed"] == 0:
        state.explanation.append("Learning: no new feedback.")
    else:
        state.explanation.append(
            f"Learning: {result['updated']} update(s) applied, "
            f"{result['skipped']} skipped "
            f"(emotion marked incorrect); "
            f"matrix version {result['version']}."
        )

    return state

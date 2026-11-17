"""Learning (Role B): updates utility matrix M from tutor feedback."""
from .config import ALTERNATIVES
from .state import CogState

LEARNING_RATE = 0.05


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply_feedback(memory, learning_rate: float = LEARNING_RATE) -> dict:
    """Consume all pending tutor feedback and update memory.utility in place.

    Returns a summary: processed / updated / skipped counts and the new matrix version.
    """
    pending = memory.pending_feedback()
    updated = skipped = 0
    for item in pending:
        if not item["emotion_correct"]:
            skipped += 1  # the emotion was misrecognized, so don't learn emotion->action from it
            continue
        row = memory.utility[item["emotion"]]
        chosen = ALTERNATIVES.index(item["selected"])
        row[chosen] = round(_clamp(row[chosen] + learning_rate), 4)
        if item["recommended"] != item["selected"]:
            rejected = ALTERNATIVES.index(item["recommended"])
            row[rejected] = round(_clamp(row[rejected] - learning_rate), 4)
        updated += 1

    version = None
    if pending:
        memory.mark_feedback_processed([item["id"] for item in pending])
    if updated:
        version = memory.save_utility_version(f"{updated} tutor feedback record(s)")
    return {"processed": len(pending), "updated": updated, "skipped": skipped, "version": version}


def run(state: CogState, memory) -> CogState:
    if not hasattr(memory, "pending_feedback"):
        state.explanation.append("Learning: memory has no feedback storage.")
        return state
    result = apply_feedback(memory)
    if result["processed"] == 0:
        state.explanation.append("Learning: no new feedback.")
    else:
        state.explanation.append(
            f"Learning: {result['updated']} update(s) applied, {result['skipped']} skipped "
            f"(emotion marked incorrect); matrix version {result['version']}.")
    return state

import pytest

from src.emotion_system.config import ALTERNATIVES, UTILITY
from src.emotion_system.learning import (
    LEARNING_RATE,
    apply_feedback,
    submit_feedback,
)
from src.emotion_system.memory import Memory


@pytest.fixture
def memory():
    m = Memory("postgresql://nobody:nobody@localhost:1/none")
    # Unreachable database -> in-memory fallback
    assert not m.db_ok
    return m


def idx(action):
    return ALTERNATIVES.index(action)


def test_selected_up_and_rejected_down(memory):
    memory.save_feedback("s", "angry", "continue", "pause", True)

    result = apply_feedback(memory)

    assert result["updated"] == 1

    row = memory.utility["angry"]

    assert row[idx("pause")] == pytest.approx(
        UTILITY["angry"][idx("pause")] + LEARNING_RATE
    )
    assert row[idx("continue")] == pytest.approx(
        UTILITY["angry"][idx("continue")] - LEARNING_RATE
    )


def test_agreement_only_reinforces_selected(memory):
    memory.save_feedback("s", "happy", "continue", "continue", True)

    apply_feedback(memory)

    row = memory.utility["happy"]

    assert row[idx("continue")] == 0.95
    assert row[idx("re_explain")] == UTILITY["happy"][idx("re_explain")]


def test_values_stay_in_bounds(memory):
    memory.utility["happy"][idx("continue")] = 0.98
    memory.utility["happy"][idx("pause")] = 0.02

    memory.save_feedback("s", "happy", "pause", "continue", True)

    apply_feedback(memory)

    assert memory.utility["happy"][idx("continue")] == 1.0
    assert memory.utility["happy"][idx("pause")] == 0.0


def test_incorrect_emotion_does_not_change_matrix(memory):
    memory.save_feedback("s", "sad", "pause", "continue", False)

    result = apply_feedback(memory)

    assert result["skipped"] == 1
    assert result["updated"] == 0
    assert result["version"] is None

    assert memory.utility["sad"] == UTILITY["sad"]


def test_feedback_is_processed_only_once(memory):
    memory.save_feedback("s", "angry", "continue", "pause", True)

    apply_feedback(memory)

    assert memory.pending_feedback() == []

    result = apply_feedback(memory)

    assert result["processed"] == 0


def test_version_saved(memory):
    memory.save_feedback("s", "angry", "continue", "pause", True)

    result = apply_feedback(memory)

    versions = memory.utility_versions()

    assert result["version"] == versions[-1]["version"]

    assert versions[-1]["matrix"]["angry"][idx("pause")] == pytest.approx(
        0.85
    )


def test_invalid_feedback_rejected(memory):
    with pytest.raises(ValueError):
        memory.save_feedback("s", "angry", "continue", "dance", True)

    with pytest.raises(ValueError):
        memory.save_feedback("s", "confused", "continue", "pause", True)


def test_original_config_not_modified(memory):
    before = {key: list(value) for key, value in UTILITY.items()}

    memory.save_feedback("s", "angry", "continue", "pause", True)

    apply_feedback(memory)

    assert UTILITY == before


def test_submit_feedback_updates_matrix_immediately(memory):
    result = submit_feedback(
        memory,
        session_id="s",
        emotion="angry",
        recommended="continue",
        selected="pause",
        emotion_correct=True,
        decision_id=123,
    )

    assert result["updated"] == 1
    assert result["processed"] == 1
    assert result["feedback"]["decision_id"] == 123

    assert memory.utility["angry"][idx("pause")] == pytest.approx(0.85)
    assert memory.utility["angry"][idx("continue")] == pytest.approx(0.05)

    assert memory.pending_feedback() == []


def test_reset_utility_restores_original_matrix(memory):
    memory.save_feedback("s", "angry", "continue", "pause", True)

    apply_feedback(memory)

    version = memory.reset_utility()

    assert memory.utility == UTILITY

    versions = memory.utility_versions()

    assert versions[-1]["version"] == version
    assert versions[-1]["matrix"] == UTILITY

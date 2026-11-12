from conftest import FakeMemory, frame

from src.emotion_system import action, decision, reasoning


def run(state, frames):
    memory = FakeMemory(frames)
    state = action.run(decision.run(reasoning.run(state, memory), memory), memory)
    return state, memory


def test_record_saved_without_image(make_state):
    state, memory = run(make_state(), [frame("happy")])
    session, record = memory.saved[0]
    assert session == "t" and record["recommended"] == state.recommended
    assert not any(k in record for k in ("image", "face_crop", "pixels"))
    assert state.report["db_written"] and "limitation" in state.report


def test_text_mentions_top_criteria(make_state):
    state, _ = run(make_state(), [frame("happy")])
    assert state.recommended in state.action and "fit" in state.action


def test_rejected_frame_text(make_state):
    state, _ = run(make_state(accepted=False), [frame("happy"), frame("angry", accepted=False)])
    assert "unreliable" in state.action and "ranked #" in state.action

from conftest import FakeMemory, frame

from src.emotion_system import reasoning


def test_weights_favour_recent_frames(make_state):
    memory = FakeMemory([frame("sad"), frame("happy")])
    state = reasoning.run(make_state("happy"), memory)
    assert state.smoothed_probs["happy"] > state.smoothed_probs["sad"]
    assert abs(sum(state.smoothed_probs.values()) - 1) < 1e-9


def test_rejected_frames_are_not_evidence(make_state):
    memory = FakeMemory([frame("happy"), frame("angry", accepted=False)])
    state = reasoning.run(make_state("angry", accepted=False), memory)
    assert max(state.smoothed_probs, key=state.smoothed_probs.get) == "happy"
    assert state.label == "insufficient_data" and any(r.startswith("R1") for r in state.rules)


def test_rejected_without_history_is_uniform(make_state):
    state = reasoning.run(make_state(accepted=False), FakeMemory([frame("angry", accepted=False)]))
    assert len(set(round(v, 6) for v in state.smoothed_probs.values())) == 1
    assert state.trend == "insufficient_history"


def test_r2_bonus_on_repeated_emotion(make_state):
    memory = FakeMemory([frame("happy", p=0.7)])
    plain = reasoning.run(make_state("happy", p=0.7), memory)
    bonus = reasoning.run(make_state("happy", p=0.7, previous="happy"), memory)
    assert abs(bonus.smoothed_probs["happy"] - plain.smoothed_probs["happy"] - 0.05) < 1e-9
    assert abs(sum(bonus.smoothed_probs.values()) - 1) < 1e-9


def test_trend_values(make_state):
    one = reasoning.run(make_state(), FakeMemory([frame("happy")]))
    stable = reasoning.run(make_state(), FakeMemory([frame("happy"), frame("happy")]))
    changing = reasoning.run(make_state(), FakeMemory([frame("sad"), frame("happy")]))
    assert (one.trend, stable.trend, changing.trend) == ("insufficient_history", "stable", "changing")


def test_r3_uncertain_and_steps_present(make_state):
    state = reasoning.run(make_state("happy", p=0.4), FakeMemory([frame("happy", p=0.4)]))
    assert state.label == "uncertain" and len(state.reasoning_steps) == 5
    assert state.explanation[-1].startswith("Reasoning")

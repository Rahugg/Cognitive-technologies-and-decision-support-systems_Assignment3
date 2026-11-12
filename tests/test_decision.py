from conftest import FakeMemory, frame

from src.emotion_system import decision, reasoning


def run(state, frames):
    memory = FakeMemory(frames)
    return decision.run(reasoning.run(state, memory), memory)


def test_matrix_shape_and_ranking(make_state):
    state = run(make_state("happy", 0.95, 0.9), [frame("happy", p=0.95)])
    assert len(state.alternatives) == 5
    assert all(len(row) == 4 for row in state.criteria_matrix["values"].values())
    assert [r["rank"] for r in state.ranking] == [1, 2, 3, 4, 5]
    assert state.recommended == state.ranking[0]["alternative"] == "continue"


def test_no_history_rejected_prefers_ask_directly(make_state):
    state = run(make_state(accepted=False, quality=0.03, p=0.3), [frame("angry", accepted=False, p=0.3)])
    assert state.recommended == "ask_directly" and state.label == "insufficient_data"


def test_low_reliability_raises_ask_directly(make_state):
    good = run(make_state("sad", 0.9, 0.9), [frame("sad")])
    poor = run(make_state("sad", 0.9, 0.1), [frame("sad")])
    rank = lambda s: next(r["rank"] for r in s.ranking if r["alternative"] == "ask_directly")
    assert rank(poor) <= rank(good)


def test_changing_trend_lowers_reliability(make_state):
    stable = run(make_state("happy"), [frame("happy"), frame("happy")])
    changing = run(make_state("happy"), [frame("sad"), frame("happy")])
    assert changing.trend == "changing"
    assert changing.criteria_matrix["reliability"] < stable.criteria_matrix["reliability"]


def test_learning_updated_matrix_is_used(make_state):
    memory = FakeMemory([frame("happy")])
    state = reasoning.run(make_state("happy"), memory)
    memory.utility["happy"][2] = 5.0  # pause becomes very useful for happy
    assert decision.run(state, memory).criteria_matrix["values"]["pause"][0] > 4

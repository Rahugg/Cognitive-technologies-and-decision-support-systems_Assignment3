import pytest

from dss.model import rank


def test_wsm_prefers_benefit_and_low_cost():
    result = rank([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5], ["benefit", "cost"])
    assert result["order"][0] == 0 and result["ranks"] == [1, 2]
    assert abs(sum(map(sum, result["contributions"])) - sum(result["scores"])) < 1e-9


def test_wsm_explains_margin_with_signed_contributions():
    result = rank([[1, 5, 8], [0, 0, 8], [0, 5, 8]], [2, 1, 1],
                  ["benefit", "cost", "cost"])
    assert result["scores"] == [0.75, 0.5, 0.25]
    assert result["comparison"] == {
        "winner": 0, "runner_up": 1, "margin": 0.25, "contributions": [0.5, -0.25, 0.0],
    }
    assert result["weights"] == [0.5, 0.25, 0.25]


def test_tie_uses_input_order():
    result = rank([[1], [1]], [1], ["benefit"])
    assert result["order"] == [0, 1]
    assert result["comparison"]["margin"] == 0


@pytest.mark.parametrize("matrix,weights,types", [
    ([[1], [0]], [0], ["benefit"]),
    ([[1], [0]], [-1], ["benefit"]),
    ([[1], [0]], [1], ["other"]),
    ([[1], [0, 1]], [1], ["benefit"]),
    ([[float("nan")], [0]], [1], ["benefit"]),
    ([[1], [0]], [float("inf")], ["benefit"]),
    ([[1], [0]], [1, 2], ["benefit"]),
    ([], [1], ["benefit"]),
])
def test_invalid_inputs_raise_value_error(matrix, weights, types):
    with pytest.raises(ValueError):
        rank(matrix, weights, types)

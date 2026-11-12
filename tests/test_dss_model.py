from dss.model import rank


def test_wsm_prefers_benefit_and_low_cost():
    result = rank([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5], ["benefit", "cost"])
    assert result["order"][0] == 0 and result["ranks"] == [1, 2]
    assert abs(sum(map(sum, result["contributions"])) - sum(result["scores"])) < 1e-9

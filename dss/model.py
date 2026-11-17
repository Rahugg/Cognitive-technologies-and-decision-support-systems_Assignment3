"""Weighted Sum Model for ranking the tutor's alternatives."""
from math import isfinite
from typing import Sequence


def rank(matrix: Sequence[Sequence[float]], weights: Sequence[float],
         types: Sequence[str], method: str = "wsm") -> dict:
    """Rank rows by the weighted sum of their min-max normalized criteria.

    Cost criteria reward smaller values. A constant column gives every row the
    same contribution. Equal scores keep the original alternative order.
    """
    if method != "wsm":
        raise ValueError(f"Unsupported method: {method}")
    if len(matrix) == 0 or len(weights) == 0 or len(types) != len(weights):
        raise ValueError("Provide alternatives and one type and weight per criterion.")
    if any(len(row) != len(weights) for row in matrix):
        raise ValueError("Each alternative must have a value for every criterion.")
    if any(kind not in ("benefit", "cost") for kind in types):
        raise ValueError("Criterion types must be 'benefit' or 'cost'.")
    if any(not isfinite(value) for row in matrix for value in row):
        raise ValueError("Criterion values must be finite.")
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Weights must be finite and nonnegative.")
    total = sum(weights)
    if not isfinite(total) or total <= 0:
        raise ValueError("Weights must have a finite, positive sum.")

    normalized_weights = [weight / total for weight in weights]
    contributions = [[0.0] * len(weights) for _ in matrix]
    for j, (column, weight, kind) in enumerate(zip(zip(*matrix), normalized_weights, types)):
        low, high = min(column), max(column)
        for i, value in enumerate(column):
            if high == low:
                normalized = 1.0
            elif kind == "cost":
                normalized = (high - value) / (high - low)
            else:
                normalized = (value - low) / (high - low)
            contributions[i][j] = weight * normalized

    scores = [sum(row) for row in contributions]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(order)
    for position, alternative in enumerate(order, start=1):
        ranks[alternative] = position

    comparison = None
    if len(order) > 1:
        winner, runner_up = order[:2]
        comparison = {
            "winner": winner,
            "runner_up": runner_up,
            "margin": scores[winner] - scores[runner_up],
            "contributions": [first - second for first, second in
                              zip(contributions[winner], contributions[runner_up])],
        }

    return {"scores": scores, "ranks": ranks, "order": order, "contributions": contributions,
            "weights": normalized_weights, "comparison": comparison, "method": method}

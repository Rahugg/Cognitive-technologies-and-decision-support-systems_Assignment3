"""DSS ranking model. STUB (Role C replaces/extends): WSM with min-max normalisation."""
from typing import Sequence


def rank(matrix: Sequence[Sequence[float]], weights: Sequence[float],
         types: Sequence[str], method: str = "wsm") -> dict:
    """Rank alternatives (rows) over criteria (columns).

    Returns {"scores": [...], "ranks": [...1 = best], "order": [best..worst indices],
    "contributions": [[weight * normalised value per criterion]], "method": "wsm"}.
    """
    if method != "wsm":
        raise ValueError(f"Unsupported method: {method}")
    total = sum(weights) or 1.0
    w = [x / total for x in weights]
    cols = list(zip(*matrix))
    contributions = [[0.0] * len(w) for _ in matrix]
    for j, col in enumerate(cols):
        lo, hi = min(col), max(col)
        for i, value in enumerate(col):
            norm = 1.0 if hi == lo else (value - lo) / (hi - lo)
            norm = norm if types[j] == "benefit" or hi == lo else 1.0 - norm
            contributions[i][j] = w[j] * norm
    scores = [sum(row) for row in contributions]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [order.index(i) + 1 for i in range(len(scores))]
    return {"scores": scores, "ranks": ranks, "order": order,
            "contributions": contributions, "method": method}

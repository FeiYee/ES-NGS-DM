"""L2-regularized Bradley-Terry aggregation of within-group pairwise preferences."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

Pair = Tuple[int, int]  # winner, loser, 0-indexed


def pairs_from_group_scores(candidate_ids: Sequence[int], scores: Sequence[float]) -> list[Pair]:
    ids = [int(c) - 1 for c in candidate_ids]
    scores = np.asarray(scores, dtype=np.float64)
    pairs: list[Pair] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if not (np.isfinite(scores[i]) and np.isfinite(scores[j])):
                continue
            if scores[i] > scores[j] + 1e-12:
                pairs.append((ids[i], ids[j]))
            elif scores[j] > scores[i] + 1e-12:
                pairs.append((ids[j], ids[i]))
    return pairs


def fit_bradley_terry(
    pairs: Iterable[Pair],
    n_items: int,
    lam: float = 0.1,
    steps: int = 500,
    lr: float = 0.08,
) -> np.ndarray:
    """Return mean-zero utilities. Deterministic given the pair list."""
    pairs = list(pairs)
    utilities = np.zeros(n_items, dtype=np.float64)
    if not pairs:
        return utilities
    winners = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    losers = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    for _ in range(steps):
        diff = utilities[winners] - utilities[losers]
        # sigmoid(loser - winner) = 1 - sigmoid(winner - loser)
        grad_scale = 1.0 / (1.0 + np.exp(np.clip(diff, -40.0, 40.0)))
        gradient = -lam * utilities
        np.add.at(gradient, winners, grad_scale)
        np.add.at(gradient, losers, -grad_scale)
        utilities += lr * gradient
        utilities -= utilities.mean()
    if not np.isfinite(utilities).all():
        raise ValueError("Bradley-Terry utilities are not finite")
    return utilities

"""Deterministic overlapping 8-candidate groups. Independent of exact labels."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def overlapping_groups(
    n_candidates: int = 48,
    group_size: int = 8,
    start_stride: int = 3,
) -> List[Tuple[int, ...]]:
    """Circular windows. Candidate ids are 1..n_candidates.

    Starts every `start_stride` positions. With n=48, size=8, stride=3 this
    yields 16 groups and each candidate appears at least twice. Windows share
    members, so the co-occurrence graph is connected. The construction uses
    only n, group size, and stride.
    """
    if n_candidates < group_size or group_size < 2:
        raise ValueError("invalid group size")
    if start_stride < 1 or start_stride >= n_candidates:
        raise ValueError("invalid stride")
    ids = np.arange(1, n_candidates + 1)
    groups: List[Tuple[int, ...]] = []
    for start in range(0, n_candidates, start_stride):
        members = tuple(int(ids[(start + offset) % n_candidates]) for offset in range(group_size))
        groups.append(members)
    return groups


def coverage_counts(groups: Sequence[Sequence[int]], n_candidates: int) -> np.ndarray:
    counts = np.zeros(n_candidates, dtype=np.int64)
    for group in groups:
        for cid in group:
            counts[int(cid) - 1] += 1
    return counts


def comparison_graph_connected(groups: Sequence[Sequence[int]], n_candidates: int) -> bool:
    parent = list(range(n_candidates + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        anchor = int(group[0])
        for cid in group[1:]:
            union(anchor, int(cid))
    roots = {find(i) for i in range(1, n_candidates + 1)}
    return len(roots) == 1


def assert_overlap_invariants(groups: Iterable[Sequence[int]], n_candidates: int = 48) -> dict:
    groups = list(groups)
    counts = coverage_counts(groups, n_candidates)
    connected = comparison_graph_connected(groups, n_candidates)
    if int(counts.min()) < 2:
        raise AssertionError(f"coverage below 2: min={int(counts.min())}")
    if not connected:
        raise AssertionError("comparison graph is disconnected")
    return {
        "n_groups": len(groups),
        "min_coverage": int(counts.min()),
        "max_coverage": int(counts.max()),
        "mean_coverage": float(counts.mean()),
        "connected": connected,
    }

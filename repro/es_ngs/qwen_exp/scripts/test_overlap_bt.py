"""Local smoke for overlapping groups and Bradley-Terry aggregation."""
from __future__ import annotations

import numpy as np

from overlap_groups import assert_overlap_invariants, overlapping_groups
from bradley_terry import fit_bradley_terry, pairs_from_group_scores


def main() -> None:
    groups = overlapping_groups()
    info = assert_overlap_invariants(groups, 48)
    again = overlapping_groups()
    if groups != again:
        raise SystemExit("groups not deterministic")
    print("GROUPS", info)

    # A strict total order 48>47>...>1 should be recovered up to affine shift.
    truth = np.arange(48, 0, -1, dtype=np.float64)
    pairs = []
    for group in groups:
        scores = [truth[cid - 1] for cid in group]
        pairs.extend(pairs_from_group_scores(group, scores))
    u1 = fit_bradley_terry(pairs, 48)
    u2 = fit_bradley_terry(pairs, 48)
    if not np.allclose(u1, u2):
        raise SystemExit("BT not deterministic")
    order = np.argsort(-u1)
    expected = np.argsort(-truth)
    if not np.array_equal(order, expected):
        raise SystemExit(f"BT order mismatch {order[:8]} vs {expected[:8]}")
    print("BT_OK", float(u1.max()), float(u1.min()))
    print("OVERLAP_BT_SMOKE_PASS")


if __name__ == "__main__":
    main()

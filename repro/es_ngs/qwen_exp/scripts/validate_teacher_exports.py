"""Fail fast when exported teacher preferences are incomplete or uninformative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(
    path: Path,
    min_valid_rate: float,
    min_nonconstant_rate: float,
    min_ndcg_margin: float,
    min_informative_rate: float,
    expected_candidates: int,
    expected_group_size: int,
) -> dict:
    df = pd.read_csv(path)
    required = {
        "teacher",
        "split",
        "state_id",
        "candidate_id",
        "teacher_group",
        "teacher_score",
        "teacher_valid",
        "parse_valid",
        "nonconstant",
        "ndcg8_group",
        "position_baseline_ndcg8",
        "random_baseline_ndcg8",
        "target_informative",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df.duplicated(["teacher", "state_id", "candidate_id"]).any():
        raise ValueError(f"{path}: duplicate teacher/state/candidate rows")

    state_coverage = df.groupby(["teacher", "state_id"], sort=False).agg(
        rows=("candidate_id", "size"),
        unique_candidates=("candidate_id", "nunique"),
        groups=("teacher_group", "nunique"),
    )
    expected_groups = int(np.ceil(expected_candidates / expected_group_size))
    bad_coverage = state_coverage[
        (state_coverage["rows"] != expected_candidates)
        | (state_coverage["unique_candidates"] != expected_candidates)
        | (state_coverage["groups"] != expected_groups)
    ]
    if len(bad_coverage):
        raise ValueError(
            f"{path}: {len(bad_coverage)} states do not have exactly "
            f"{expected_candidates} candidates in {expected_groups} disjoint groups"
        )
    for (_, _), state_rows in df.groupby(["teacher", "state_id"], sort=False):
        ids = sorted(pd.to_numeric(state_rows["candidate_id"], errors="raise").astype(int).tolist())
        if ids != list(range(1, expected_candidates + 1)):
            raise ValueError(f"{path}: candidate ids are not 1..{expected_candidates}")

    val_df = df[df["split"].astype(str).str.lower() == "val"].copy()
    if val_df.empty:
        raise ValueError(f"{path}: no validation rows; teacher gate must not use training groups")

    groups = val_df.groupby(["teacher", "state_id", "teacher_group"], sort=False)
    group_audit = groups.agg(
        parse_valid=("parse_valid", "min"),
        teacher_valid=("teacher_valid", "min"),
        nonconstant=("nonconstant", "min"),
        target_informative=("target_informative", "min"),
        group_size=("candidate_id", "size"),
        finite_scores=("teacher_score", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
    ).reset_index()
    if not bool((group_audit["group_size"] == expected_group_size).all()):
        raise ValueError(f"{path}: validation teacher groups are not all size {expected_group_size}")
    valid_rate = float(group_audit["teacher_valid"].mean())
    parse_rate = float(group_audit["parse_valid"].mean())
    informative = group_audit["target_informative"].astype(bool)
    informative_rate = float(informative.mean())
    if informative_rate < min_informative_rate:
        raise ValueError(
            f"{path}: informative validation-group rate {informative_rate:.3f} "
            f"< {min_informative_rate:.3f}"
        )
    nonconstant_rate = float(group_audit.loc[informative, "nonconstant"].mean())
    finite_rate = float(
        np.isfinite(pd.to_numeric(val_df["teacher_score"], errors="coerce")).mean()
    )
    group_scores = groups[
        ["ndcg8_group", "position_baseline_ndcg8", "random_baseline_ndcg8", "target_informative"]
    ].first()
    group_scores = group_scores[group_scores["target_informative"].astype(bool)]
    teacher_ndcg = float(pd.to_numeric(group_scores["ndcg8_group"], errors="coerce").mean())
    position_ndcg = float(pd.to_numeric(group_scores["position_baseline_ndcg8"], errors="coerce").mean())
    random_ndcg = float(pd.to_numeric(group_scores["random_baseline_ndcg8"], errors="coerce").mean())
    ndcg_margin = teacher_ndcg - max(position_ndcg, random_ndcg)
    if valid_rate < min_valid_rate:
        raise ValueError(f"{path}: teacher-valid rate {valid_rate:.3f} < {min_valid_rate:.3f}")
    if parse_rate < min_valid_rate:
        raise ValueError(f"{path}: parse-valid rate {parse_rate:.3f} < {min_valid_rate:.3f}")
    if finite_rate < min_valid_rate:
        raise ValueError(f"{path}: finite-score coverage {finite_rate:.3f} < {min_valid_rate:.3f}")
    if nonconstant_rate < min_nonconstant_rate:
        raise ValueError(
            f"{path}: nonconstant preference rate {nonconstant_rate:.3f} < {min_nonconstant_rate:.3f}"
        )
    if not np.isfinite(ndcg_margin) or ndcg_margin < min_ndcg_margin:
        raise ValueError(
            f"{path}: teacher NDCG margin {ndcg_margin:.3f} < {min_ndcg_margin:.3f} "
            f"(teacher={teacher_ndcg:.3f}, position={position_ndcg:.3f}, random={random_ndcg:.3f})"
        )
    return {
        "path": str(path),
        "teachers": sorted(df["teacher"].unique().tolist()),
        "states_total": int(df["state_id"].nunique()),
        "states_validation": int(val_df["state_id"].nunique()),
        "groups": int(len(group_audit)),
        "rows": int(len(df)),
        "parse_valid_rate": parse_rate,
        "teacher_valid_rate": valid_rate,
        "finite_score_coverage": finite_rate,
        "nonconstant_rate": nonconstant_rate,
        "informative_group_rate": informative_rate,
        "teacher_ndcg8": teacher_ndcg,
        "position_baseline_ndcg8": position_ndcg,
        "random_baseline_ndcg8": random_ndcg,
        "ndcg_margin": ndcg_margin,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--min-valid-rate", type=float, default=0.95)
    ap.add_argument("--min-nonconstant-rate", type=float, default=0.80)
    ap.add_argument("--min-ndcg-margin", type=float, default=0.01)
    ap.add_argument("--min-informative-rate", type=float, default=0.20)
    ap.add_argument("--expected-candidates", type=int, default=48)
    ap.add_argument("--expected-group-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    reports = [
        audit(
            path,
            args.min_valid_rate,
            args.min_nonconstant_rate,
            args.min_ndcg_margin,
            args.min_informative_rate,
            args.expected_candidates,
            args.expected_group_size,
        )
        for path in args.paths
    ]
    payload = {"status": "ok", "reports": reports}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

"""Aggregate overlapping group scores into one utility per candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bradley_terry import fit_bradley_terry, pairs_from_group_scores
from overlap_groups import comparison_graph_connected, coverage_counts


def aggregate_state(frame: pd.DataFrame, n_candidates: int) -> pd.DataFrame | None:
    pairs = []
    finite_groups = 0
    nonconstant_groups = 0
    for _, group in frame.groupby("teacher_group"):
        scores = pd.to_numeric(group["teacher_score"], errors="coerce").to_numpy()
        ids = group["candidate_id"].astype(int).tolist()
        if not np.isfinite(scores).all():
            continue
        finite_groups += 1
        if float(np.ptp(scores)) <= 1e-12:
            continue
        nonconstant_groups += 1
        pairs.extend(pairs_from_group_scores(ids, scores))
    present = sorted(set(int(x) for x in frame["candidate_id"]))
    connected = comparison_graph_connected(
        [tuple(g["candidate_id"].astype(int).tolist()) for _, g in frame.groupby("teacher_group") if np.isfinite(pd.to_numeric(g["teacher_score"], errors="coerce")).all()],
        n_candidates,
    )
    counts = coverage_counts(
        [tuple(g["candidate_id"].astype(int).tolist()) for _, g in frame.groupby("teacher_group")],
        n_candidates,
    )
    if not connected or finite_groups == 0 or present != list(range(1, n_candidates + 1)):
        return None
    utilities = fit_bradley_terry(pairs, n_candidates)
    # Confidence: share of a candidate's group appearances that were nonconstant and finite.
    conf = np.zeros(n_candidates, dtype=np.float64)
    for cid in range(1, n_candidates + 1):
        sub = frame[frame["candidate_id"] == cid]
        scores = pd.to_numeric(sub["teacher_score"], errors="coerce")
        good = np.isfinite(scores.to_numpy())
        conf[cid - 1] = float(good.mean()) if len(sub) else 0.0
    out = frame.drop_duplicates("candidate_id").copy()
    out = out.sort_values("candidate_id")
    out["teacher_utility"] = utilities
    out["teacher_confidence"] = conf
    out["coverage_count"] = counts
    out["comparison_graph_connected"] = True
    out["finite_groups"] = finite_groups
    out["nonconstant_groups"] = nonconstant_groups
    out["teacher_valid"] = 1
    out["teacher_score"] = utilities
    out["teacher_group"] = 0
    return out[
        [
            "teacher",
            "state_id",
            "instance_id",
            "split",
            "candidate_id",
            "teacher_utility",
            "teacher_score",
            "teacher_group",
            "teacher_confidence",
            "coverage_count",
            "comparison_graph_connected",
            "teacher_valid",
            "finite_groups",
            "nonconstant_groups",
        ]
    ]


def ensemble(math_df: pd.DataFrame, general_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_id", "candidate_id"]
    a = math_df.set_index(keys)
    b = general_df.set_index(keys)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    rows = []
    for sid, part_a in a.groupby(level=0):
        part_b = b.xs(sid, level=0)
        ua = part_a["teacher_utility"].to_numpy()
        ub = part_b.loc[part_a.index.get_level_values(1), "teacher_utility"].to_numpy()
        za = (ua - ua.mean()) / (ua.std() + 1e-8)
        zb = (ub - ub.mean()) / (ub.std() + 1e-8)
        mixed = 0.5 * (za + zb)
        agree = 1.0 - np.abs(za - zb) / (np.abs(za - zb).mean() + 1e-8)
        agree = np.clip(agree, 0.0, 1.0)
        for cid, util, conf, cov_a, cov_b in zip(
            part_a.index.get_level_values(1),
            mixed,
            0.5 * (part_a["teacher_confidence"].to_numpy() + part_b["teacher_confidence"].to_numpy()) * agree,
            part_a["coverage_count"].to_numpy(),
            part_b["coverage_count"].to_numpy(),
        ):
            meta = part_a.iloc[0]
            rows.append(
                {
                    "teacher": "qwen_ensemble",
                    "state_id": int(sid),
                    "instance_id": meta["instance_id"],
                    "split": meta["split"],
                    "candidate_id": int(cid),
                    "teacher_utility": float(util),
                    "teacher_score": float(util),
                    "teacher_group": 0,
                    "teacher_confidence": float(conf),
                    "coverage_count": int(min(cov_a, cov_b)),
                    "comparison_graph_connected": True,
                    "teacher_valid": 1,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument(
        "--n-candidates",
        type=int,
        default=48,
        help="If >0 and a state has exactly this many unique candidates, use it; "
        "otherwise use per-state max candidate_id. Pass 0 to always use per-state max.",
    )
    ap.add_argument("--ensemble-out", type=Path, default=None)
    args = ap.parse_args()
    frames = []
    failed = []
    for path in args.group_csv:
        raw = pd.read_csv(path)
        for (teacher, sid), state in raw.groupby(["teacher", "state_id"], sort=False):
            n_cands = int(state["candidate_id"].astype(int).max())
            if args.n_candidates and args.n_candidates > 0:
                # Keep CLI default as a floor hint only when all states share it;
                # otherwise trust per-state max candidate id (48 vs 64 packs).
                if int(state["candidate_id"].astype(int).nunique()) == args.n_candidates:
                    n_cands = args.n_candidates
            agg = aggregate_state(state, n_cands)
            if agg is None:
                failed.append({"teacher": teacher, "state_id": int(sid), "n_cands": n_cands})
                continue
            frames.append(agg)
    if not frames:
        raise SystemExit("no states aggregated")
    out = pd.concat(frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    report = {
        "rows": int(len(out)),
        "states": int(out["state_id"].nunique()),
        "failed_states": failed,
        "failed_count": len(failed),
    }
    args.out_csv.with_suffix(".aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if args.ensemble_out and {"qwen_math", "qwen_general"} <= set(out["teacher"].unique()):
        ens = ensemble(out[out["teacher"] == "qwen_math"], out[out["teacher"] == "qwen_general"])
        both = pd.concat([out, ens], ignore_index=True)
        both.to_csv(args.ensemble_out, index=False)
        print("ENSEMBLE_ROWS", len(ens))


if __name__ == "__main__":
    main()

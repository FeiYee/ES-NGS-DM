"""State-level validation for aggregated global teacher utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def ndcg_at_k(scores: np.ndarray, targets: np.ndarray, k: int = 8) -> float:
    if len(scores) < 2 or not np.isfinite(targets).any():
        return float("nan")
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    gains = np.maximum(targets[order], 0.0)
    dcg = float(np.sum(gains / np.log2(np.arange(2, k + 2))))
    ideal = np.sort(np.maximum(targets, 0.0))[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 1e-9 else float("nan")


def pairwise_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    n = 0
    hits = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if not (np.isfinite(targets[i]) and np.isfinite(targets[j])):
                continue
            if abs(targets[i] - targets[j]) <= 1e-12:
                continue
            n += 1
            pred = scores[i] - scores[j]
            truth = targets[i] - targets[j]
            hits += int((pred > 0) == (truth > 0))
    return hits / n if n else float("nan")


def top8_metrics(scores: np.ndarray, targets: np.ndarray, feasible: np.ndarray) -> dict:
    k = min(8, len(scores))
    pred = set(np.argsort(-scores)[:k].tolist())
    exact = np.where(feasible & np.isfinite(targets), targets, -np.inf)
    oracle = set(np.argsort(-exact)[:k].tolist())
    recall = len(pred & oracle) / k
    best_exact = float(np.max(exact[list(pred)])) if pred else float("nan")
    oracle_best = float(np.max(exact[list(oracle)])) if oracle else float("nan")
    mean8 = float(np.mean(exact[list(pred)])) if pred else float("nan")
    oracle_mean8 = float(np.mean(exact[list(oracle)])) if oracle else float("nan")
    regret = oracle_mean8 - mean8 if np.isfinite(oracle_mean8) and np.isfinite(mean8) else float("nan")
    return {
        "top8_recall": recall,
        "top8_best_exact_gain": best_exact,
        "oracle_best_exact_gain": oracle_best,
        "regret_at_8": regret,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-csv", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.teacher_csv)
    reports = []
    for teacher, tdf in df.groupby("teacher"):
        val = tdf[tdf["split"].astype(str).str.lower() == "val"].copy()
        if val.empty:
            raise SystemExit(f"no validation rows for {teacher}")
        rows = []
        for sid, state in val.groupby("state_id"):
            state = state.sort_values("candidate_id")
            d = args.label_dir / "states" / f"s{int(sid)}"
            y = np.load(d / "y.npy").astype(np.float64)
            feas = np.load(d / "feas.npy") > 0.5
            scores = pd.to_numeric(state["teacher_utility"], errors="coerce").to_numpy()
            targets = np.asarray([y[int(c) - 1] if feas[int(c) - 1] else np.nan for c in state["candidate_id"]], dtype=np.float64)
            metrics = {
                "state_id": int(sid),
                "ndcg8": ndcg_at_k(scores, np.nan_to_num(targets, nan=-1e9)),
                "pairwise_accuracy": pairwise_accuracy(scores, targets),
                "connected": bool(state["comparison_graph_connected"].astype(bool).all()),
                "mean_confidence": float(pd.to_numeric(state["teacher_confidence"], errors="coerce").mean()),
                "mean_coverage": float(pd.to_numeric(state["coverage_count"], errors="coerce").mean()),
                "nonconstant": int(float(np.ptp(scores[np.isfinite(scores)])) > 1e-12),
                "parse_finite": int(np.isfinite(scores).all()),
            }
            metrics.update(top8_metrics(scores, y, feas))
            rows.append(metrics)
        rdf = pd.DataFrame(rows)
        reports.append(
            {
                "teacher": teacher,
                "n_val_states": int(len(rdf)),
                "state_ndcg8": float(rdf["ndcg8"].mean()),
                "pairwise_accuracy": float(rdf["pairwise_accuracy"].mean()),
                "top8_recall": float(rdf["top8_recall"].mean()),
                "top8_best_exact_gain": float(rdf["top8_best_exact_gain"].mean()),
                "regret_at_8": float(rdf["regret_at_8"].mean()),
                "connected_rate": float(rdf["connected"].mean()),
                "nonconstant_rate": float(rdf["nonconstant"].mean()),
                "finite_rate": float(rdf["parse_finite"].mean()),
                "mean_confidence": float(rdf["mean_confidence"].mean()),
                "mean_coverage": float(rdf["mean_coverage"].mean()),
            }
        )
    # teacher agreement on validation states present for both math and general
    teachers = set(df["teacher"].unique())
    if {"qwen_math", "qwen_general"} <= teachers:
        a = df[(df["teacher"] == "qwen_math") & (df["split"] == "val")]
        b = df[(df["teacher"] == "qwen_general") & (df["split"] == "val")]
        kts, sps = [], []
        for sid in sorted(set(a["state_id"]) & set(b["state_id"])):
            sa = a[a["state_id"] == sid].sort_values("candidate_id")["teacher_utility"].to_numpy()
            sb = b[b["state_id"] == sid].sort_values("candidate_id")["teacher_utility"].to_numpy()
            kts.append(float(kendalltau(sa, sb).correlation))
            sps.append(float(spearmanr(sa, sb).correlation))
        agreement = {
            "kendall": float(np.nanmean(kts)),
            "spearman": float(np.nanmean(sps)),
        }
    else:
        agreement = {}
    payload = {"status": "ok", "reports": reports, "teacher_agreement": agreement}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

"""Build leakage-free Qwen prompts and an instance-wise split from labels_v2."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from decision_prompt import build_decision_prompt, summarize_action, summarize_global

ROOT = Path(__file__).resolve().parents[1]


def ndcg_at_k(scores: np.ndarray, targets: np.ndarray, valid: np.ndarray, k: int = 8) -> float:
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return float("nan")
    s, t = scores[idx], targets[idx]
    k = min(k, len(idx))
    order = np.argsort(-s)[:k]
    gains = np.maximum(t[order], 0.0)
    dcg = float(np.sum(gains / np.log2(np.arange(2, k + 2))))
    ideal = np.sort(np.maximum(t, 0.0))[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 1e-9 else float("nan")


def soft_rank_targets(y: np.ndarray, feasible: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(y, dtype=np.float64).copy()
    valid = np.asarray(feasible, dtype=bool) & np.isfinite(z)
    z[~valid] = -1e9
    if not valid.any():
        return np.zeros_like(z)
    z = z / max(temperature, 1e-6)
    z = z - z[valid].max()
    e = np.zeros_like(z)
    e[valid] = np.exp(z[valid])
    return e / max(e.sum(), 1e-12)


def build_target(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    y = np.array([c["y_pct"] for c in cands], dtype=np.float64)
    feas = np.array([c["feasible"] for c in cands], dtype=bool)
    valid = feas & np.isfinite(y)
    if valid.any():
        span = max(float(np.ptp(y[valid])), 1.0)
        infeasible_floor = float(np.min(y[valid]) - span)
    else:
        infeasible_floor = -1.0
    # Exact-repair failures are consistently below every feasible action.
    order_scores = np.where(valid, y, infeasible_floor)
    choice = [int(cands[i]["cid"]) for i in np.argsort(-order_scores)]
    score = [round(float(order_scores[i]), 6) for i in range(len(cands))]
    prob = [1.0 if (c["feasible"] and c["y_pct"] > 0) else 0.0 for c in cands]
    return {"choice": choice, "score": score, "probability": prob}


def load_state(label_dir: Path, sid: int, rows: pd.DataFrame) -> Dict[str, Any]:
    d = label_dir / "states" / f"s{sid}"
    action = np.load(d / "action.npy")
    glob = np.load(d / "global.npy")
    y = np.load(d / "y.npy").astype(np.float64)
    feas = np.load(d / "feas.npy") > 0.5
    # rows are 1-indexed candidate_id
    r0 = rows.iloc[0]
    meta = {
        "state_id": sid,
        "instance_id": str(r0["instance_id"]),
        "n": int(r0["n"]),
        "family": str(r0["family"]),
        "phase": str(r0["phase"]),
        "before_makespan_h": float(r0["before_makespan_h"]),
        "before_T_h": float(r0["before_makespan_h"]),
        "seed": int(r0["seed"]),
        "global": summarize_global(glob[0] if glob.ndim == 2 else glob),
    }
    C = min(len(y), action.shape[0], len(feas))
    cands = []
    for i in range(C):
        yi = float(y[i]) if np.isfinite(y[i]) else float("nan")
        cands.append(
            {
                "cid": i + 1,
                "feasible": bool(feas[i]),
                "y_pct": yi if np.isfinite(yi) else 0.0,
                "action": summarize_action(action[i]),
            }
        )
    return {"meta": meta, "cands": cands, "y": y[:C], "feas": feas[:C]}


def sample_candidate_groups(
    cands: List[Dict[str, Any]],
    rng: random.Random,
    group_size: int,
    n_groups: int,
) -> List[List[Dict[str, Any]]]:
    """Create label-free random groups, covering candidates before resampling."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    groups: List[List[Dict[str, Any]]] = []
    while len(groups) < n_groups:
        shuffled = list(cands)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), group_size):
            group = shuffled[start : start + group_size]
            if len(group) >= 2:
                groups.append(group)
            if len(groups) == n_groups:
                break
    return groups


def split_instances(instance_ids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = sorted(set(instance_ids))
    rng.shuffle(ids)
    n_val = max(1, int(round(val_frac * len(ids))))
    val = ids[:n_val]
    train = ids[n_val:]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples-per-state", type=int, default=6)
    ap.add_argument("--candidate-group-size", type=int, default=8)
    args = ap.parse_args()

    label_dir = args.label_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(label_dir / "LABELS.csv")
    by_state = {sid: g for sid, g in df.groupby("state_id")}
    states = sorted(by_state.keys())

    records = []
    soft_rows = []
    for sid in states:
        st = load_state(label_dir, int(sid), by_state[sid])
        records.append(st)
        soft = soft_rank_targets(st["y"], st["feas"])
        for i, c in enumerate(st["cands"]):
            soft_rows.append(
                {
                    "state_id": sid,
                    "instance_id": st["meta"]["instance_id"],
                    "candidate_id": c["cid"],
                    "y_pct": c["y_pct"],
                    "feasible": int(c["feasible"]),
                    "soft_rank": float(soft[i]) if i < len(soft) else 0.0,
                    "positive_gain": int(c["feasible"] and c["y_pct"] > 0),
                }
            )

    inst_ids = [r["meta"]["instance_id"] for r in records]
    train_inst, val_inst = split_instances(inst_ids, args.val_frac, args.seed)
    train_set, val_set = set(train_inst), set(val_inst)
    if train_set & val_set:
        raise ValueError("instance leakage between train and validation splits")

    rng = random.Random(args.seed)
    prompts_train, prompts_val = [], []
    for st in records:
        split = "train" if st["meta"]["instance_id"] in train_set else "val"
        # Exact-once partition: #groups must match n_cands / group_size (48→6, 64→8).
        n_cands = len(st["cands"])
        if n_cands % args.candidate_group_size != 0:
            raise ValueError(
                f"state {st['meta']['state_id']} has {n_cands} cands not divisible by "
                f"{args.candidate_group_size}"
            )
        n_groups = max(args.samples_per_state, n_cands // args.candidate_group_size)
        # Prefer exact-once coverage when samples_per_state does not match n_cands.
        if n_groups * args.candidate_group_size != n_cands:
            n_groups = n_cands // args.candidate_group_size
        groups = sample_candidate_groups(
            st["cands"], rng, args.candidate_group_size, n_groups
        )
        covered_ids = [int(candidate["cid"]) for group in groups for candidate in group]
        expected_ids = sorted(int(candidate["cid"]) for candidate in st["cands"])
        if sorted(covered_ids) != expected_ids:
            raise ValueError(
                f"state {st['meta']['state_id']} is not covered exactly once by label-free groups "
                f"(n_cands={n_cands}, n_groups={n_groups}, covered={len(covered_ids)})"
            )
        for group_id, candidate_group in enumerate(groups):
            prompt = build_decision_prompt(st["meta"], candidate_group)
            target = build_target(candidate_group)
            row = {
                "state_id": st["meta"]["state_id"],
                "instance_id": st["meta"]["instance_id"],
                "split": split,
                "teacher_group": group_id,
                "prompt": prompt,
                "target": target,
                "candidate_ids": [c["cid"] for c in candidate_group],
                "sampling_policy": "label_free_random_partition",
            }
            (prompts_train if split == "train" else prompts_val).append(row)

    (out_dir / "split.json").write_text(
        json.dumps({"train_instances": train_inst, "val_instances": val_inst, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "prompts_train.jsonl").open("w", encoding="utf-8") as f:
        for r in prompts_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "prompts_val.jsonl").open("w", encoding="utf-8") as f:
        for r in prompts_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(soft_rows).to_csv(out_dir / "exact_soft_labels.csv", index=False)

    # state-level exact ranking dump for student KD
    state_index = []
    for st in records:
        state_index.append(
            {
                "state_id": st["meta"]["state_id"],
                "instance_id": st["meta"]["instance_id"],
                "split": "train" if st["meta"]["instance_id"] in train_set else "val",
                "n_cands": len(st["cands"]),
            }
        )
    pd.DataFrame(state_index).to_csv(out_dir / "state_index.csv", index=False)
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "candidate_group_size": args.candidate_group_size,
                "samples_per_state": args.samples_per_state,
                "sampling_policy": "label_free_random_partition",
                "prompt_contains_exact_feasibility": False,
                "soft_targets_mask_infeasible": True,
                "prompt_feature_schema": "interpretable_energy_sync_v2",
                "exact_candidate_coverage_once": True,
                "train_validation_split_by_instance": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "states": len(records),
                "train_instances": len(train_inst),
                "val_instances": len(val_inst),
                "prompts_train": len(prompts_train),
                "prompts_val": len(prompts_val),
            }
        )
    )


if __name__ == "__main__":
    main()

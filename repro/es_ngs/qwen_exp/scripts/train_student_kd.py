"""Train ES-NGS critic variants: Exact-only / Exact Soft-label / Qwen-* KD."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --es-ngs-root points to the directory containing model.py.
def _add_paths(es_ngs_root: Path, pydeps: Optional[Path] = None) -> None:
    if str(es_ngs_root) not in sys.path:
        sys.path.insert(0, str(es_ngs_root))
    if pydeps and str(pydeps) not in sys.path:
        sys.path.insert(0, str(pydeps))


class StateDataset(Dataset):
    def __init__(
        self,
        state_ids: List[int],
        label_dir: Path,
        soft_csv: Optional[Path] = None,
        teacher_csv: Optional[Path] = None,
        teacher_name: Optional[str] = None,
        confidence_threshold: float = 0.0,
    ):
        self.state_ids = state_ids
        self.label_dir = label_dir
        self.soft = None
        self.teacher = None
        for supplied in (soft_csv, teacher_csv):
            if supplied is not None and not supplied.is_file():
                raise FileNotFoundError(f"supervision file not found: {supplied}")
        if soft_csv and soft_csv.exists():
            sdf = pd.read_csv(soft_csv)
            self.soft = {(int(r.state_id), int(r.candidate_id)): float(r.soft_rank) for r in sdf.itertuples()}
        if teacher_csv and teacher_csv.exists():
            tdf = pd.read_csv(teacher_csv)
            if teacher_name:
                tdf = tdf[tdf["teacher"] == teacher_name]
            duplicate = tdf.duplicated(["state_id", "candidate_id"], keep=False)
            if duplicate.any():
                raise ValueError(
                    "teacher CSV has duplicate state/candidate rows; export disjoint candidate groups"
                )
            self.teacher = {}
            has_group = "teacher_group" in tdf.columns
            score_attr = "teacher_utility" if "teacher_utility" in tdf.columns else "teacher_score"
            global_teacher = score_attr == "teacher_utility"
            for r in tdf.itertuples():
                score = float(getattr(r, score_attr))
                conf = float(getattr(r, "teacher_confidence", 1.0))
                valid = bool(getattr(r, "teacher_valid", 1)) and math.isfinite(score) and conf >= confidence_threshold
                group = 0 if global_teacher else (int(getattr(r, "teacher_group", 0)) if has_group else 0)
                self.teacher[(int(r.state_id), int(r.candidate_id))] = (score, group, valid)

    def __len__(self):
        return len(self.state_ids)

    def __getitem__(self, i):
        sid = self.state_ids[i]
        d = self.label_dir / "states" / f"s{sid}"

        def nan0(a):
            a = a.astype(np.float32)
            a[~np.isfinite(a)] = 0.0
            return a

        X = np.load(d / "X.npy")
        A = np.load(d / "A.npy")
        action = np.load(d / "action.npy")
        glob = np.load(d / "global.npy")
        mask = np.load(d / "mask.npy")
        y = nan0(np.load(d / "y.npy"))
        feas = np.load(d / "feas.npy") > 0.5
        yd = nan0(np.load(d / "y_drive.npy"))
        ys = nan0(np.load(d / "y_swap.npy"))
        ysyn = nan0(np.load(d / "y_sync.npy"))
        C = y.shape[0]
        soft = np.zeros(C, dtype=np.float32)
        teach = np.zeros(C, dtype=np.float32)
        teacher_mask = np.zeros(C, dtype=np.bool_)
        teacher_group = np.full(C, -1, dtype=np.int64)
        for c in range(C):
            soft[c] = self.soft.get((sid, c + 1), 0.0) if self.soft else 0.0
            if self.teacher and (sid, c + 1) in self.teacher:
                score, group, valid = self.teacher[(sid, c + 1)]
                if valid:
                    teach[c] = score
                    teacher_mask[c] = True
                    teacher_group[c] = group
        return {
            "X": torch.from_numpy(X.astype(np.float32)),
            "A": torch.from_numpy(A.astype(np.int64)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "global": torch.from_numpy(glob.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "y": torch.from_numpy(y),
            "y_drive": torch.from_numpy(yd),
            "y_swap": torch.from_numpy(ys),
            "y_sync": torch.from_numpy(ysyn),
            "soft": torch.from_numpy(soft),
            "teacher": torch.from_numpy(teach),
            "teacher_mask": torch.from_numpy(teacher_mask),
            "teacher_group": torch.from_numpy(teacher_group),
            "feasible": torch.from_numpy(feas.astype(np.bool_)),
            "sid": sid,
        }


def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    Cmax = max(b["X"].shape[0] for b in batch)
    B = len(batch)
    M, Fdim = batch[0]["X"].shape[1], batch[0]["X"].shape[2]
    T = batch[0]["A"].shape[1]
    Aact, G = batch[0]["action"].shape[1], batch[0]["global"].shape[1]
    out = {
        "X": torch.zeros(B, Cmax, M, Fdim),
        "A": torch.zeros(B, Cmax, T, T, dtype=torch.long),
        "action": torch.zeros(B, Cmax, Aact),
        "global": torch.zeros(B, Cmax, G),
        "mask": torch.zeros(B, Cmax, T),
        "y": torch.zeros(B, Cmax),
        "y_drive": torch.zeros(B, Cmax),
        "y_swap": torch.zeros(B, Cmax),
        "y_sync": torch.zeros(B, Cmax),
        "soft": torch.zeros(B, Cmax),
        "teacher": torch.zeros(B, Cmax),
        "teacher_mask": torch.zeros(B, Cmax, dtype=torch.bool),
        "teacher_group": torch.full((B, Cmax), -1, dtype=torch.long),
        "feasible": torch.zeros(B, Cmax, dtype=torch.bool),
        "cand_mask": torch.zeros(B, Cmax, dtype=torch.bool),
    }
    # Keep the CLS token valid even for padded candidate slots so the
    # Transformer never receives an all-masked sequence and emits NaNs.
    out["mask"][:, :, 0] = 1.0
    for i, b in enumerate(batch):
        C = b["X"].shape[0]
        for k in (
            "X", "A", "action", "global", "mask", "y", "y_drive", "y_swap",
            "y_sync", "soft", "teacher", "teacher_mask", "teacher_group", "feasible",
        ):
            out[k][i, :C] = b[k]
        out["cand_mask"][i, :C] = True
    return out


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


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    for value in np.unique(scores):
        tied = np.where(scores == value)[0]
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def eval_loader(model, loader, device, online_feas_weight: float = 1.0) -> Dict[str, float]:
    model.eval()
    ndcgs, top1, top8_feasible, top8_recall, top8_regret = [], [], [], [], []
    feas_scores, feas_targets = [], []
    for batch in loader:
        B, C = batch["y"].shape
        flat = {
            "X": batch["X"].reshape(B * C, *batch["X"].shape[2:]).to(device),
            "A": batch["A"].reshape(B * C, *batch["A"].shape[2:]).to(device),
            "action": batch["action"].reshape(B * C, -1).to(device),
            "global": batch["global"].reshape(B * C, -1).to(device),
            "mask": batch["mask"].reshape(B * C, -1).to(device),
        }
        pred = model(flat)
        gain = pred["gain"].reshape(B, C).cpu().numpy()
        feas_logit = pred["feas"].reshape(B, C).cpu().numpy()
        feas_prob = 1.0 / (1.0 + np.exp(-np.clip(feas_logit, -30.0, 30.0)))
        online_score = gain + online_feas_weight * np.log(np.clip(feas_prob, 1e-6, 1.0))
        y = batch["y"].numpy()
        candidate = batch["cand_mask"].numpy()
        feasible = batch["feasible"].numpy()
        for i in range(B):
            v = candidate[i]
            if v.sum() < 2 or not (feasible[i] & v).any():
                continue
            relevance = np.where(feasible[i], np.maximum(y[i], 0.0), 0.0)
            nd = ndcg_at_k(online_score[i], relevance, v, 8)
            if math.isfinite(nd):
                ndcgs.append(nd)
            gi = online_score[i].copy()
            gi[~v] = -1e9
            yi = y[i].copy()
            yi[~(feasible[i] & v)] = -1e9
            top1.append(1.0 if int(np.argmax(gi)) == int(np.argmax(yi)) else 0.0)
            k = min(8, int(v.sum()))
            selected = np.argsort(-gi)[:k]
            top8_feasible.append(float(feasible[i, selected].mean()))
            best_idx = int(np.argmax(yi))
            top8_recall.append(float(best_idx in selected))
            selected_best = float(np.max(yi[selected]))
            top8_regret.append(max(0.0, float(yi[best_idx] - selected_best)))
            feas_scores.extend(feas_prob[i, v].tolist())
            feas_targets.extend(feasible[i, v].astype(np.int64).tolist())
    auc = binary_auc(np.asarray(feas_scores), np.asarray(feas_targets)) if feas_scores else float("nan")
    return {
        "ndcg8": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top8_feasible_rate": float(np.mean(top8_feasible)) if top8_feasible else float("nan"),
        "top8_recall": float(np.mean(top8_recall)) if top8_recall else float("nan"),
        "top8_regret": float(np.mean(top8_regret)) if top8_regret else float("nan"),
        "feasibility_auc": auc,
        "n_states": len(ndcgs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=[
            "exact_only",
            "exact_softlabel",
            "qwen_general_kd",
            "qwen_math_kd",
            "overlap_math_kd",
            "overlap_general_kd",
            "overlap_ensemble_kd",
            "full_v3",
        ],
        required=True,
    )
    ap.add_argument("--es-ngs-root", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--state-index", type=Path, required=True)
    ap.add_argument("--soft-csv", type=Path, default=None)
    ap.add_argument("--teacher-csv", type=Path, default=None)
    ap.add_argument("--out-ckpt", type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank-w", type=float, default=1.0)
    ap.add_argument("--kd-w", type=float, default=0.5)
    ap.add_argument("--soft-w", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--feas-w", type=float, default=0.5)
    ap.add_argument("--online-feas-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--final-all-data",
        action="store_true",
        help="Train the deployment checkpoint on every indexed state and save the final epoch.",
    )
    ap.add_argument("--pydeps", type=Path, default=None)
    ap.add_argument(
        "--teacher-name",
        type=str,
        default="",
        help="Override teacher column filter (e.g. exact_self for Replay self-training).",
    )
    args = ap.parse_args()

    _add_paths(args.es_ngs_root, args.pydeps)
    from model import (  # noqa: WPS433
        ESNGSCritic,
        count_params,
        listwise_probability_loss,
        listwise_ranking_loss,
        three_way_loss,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    idx = pd.read_csv(args.state_index)
    if args.final_all_data:
        train_ids = [int(x) for x in idx["state_id"].drop_duplicates().tolist()]
        # This evaluation is a training diagnostic only. Reported predictive
        # performance always comes from the five grouped held-out folds.
        val_ids = list(train_ids)
    else:
        train_ids = [int(x) for x in idx.loc[idx["split"] == "train", "state_id"].tolist()]
        val_ids = [int(x) for x in idx.loc[idx["split"] == "val", "state_id"].tolist()]

    teacher_name = None
    soft_csv = None
    teacher_csv = None
    if args.variant == "exact_softlabel":
        soft_csv = args.soft_csv
    elif args.variant == "qwen_general_kd":
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_general"
    elif args.variant == "qwen_math_kd":
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_math"
    elif args.variant == "overlap_math_kd":
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_math"
    elif args.variant == "overlap_general_kd":
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_general"
    elif args.variant == "overlap_ensemble_kd":
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_ensemble"
    elif args.variant == "full_v3":
        soft_csv = args.soft_csv
        teacher_csv = args.teacher_csv
        teacher_name = "qwen_ensemble"
    if args.teacher_name:
        teacher_name = args.teacher_name

    train_ds = StateDataset(
        train_ids, args.label_dir, soft_csv, teacher_csv, teacher_name, args.confidence_threshold
    )
    val_ds = StateDataset(
        val_ids, args.label_dir, soft_csv, teacher_csv, teacher_name, args.confidence_threshold
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESNGSCritic().to(device)
    if args.init_ckpt:
        init_blob = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(init_blob["model"], strict=True)
        print("initialized", args.init_ckpt, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print("params", count_params(model), "device", device, "variant", args.variant)

    initial = eval_loader(model, val_loader, device, args.online_feas_weight)
    best = float(initial["ndcg8"])
    best_key = (float(initial["top8_recall"]), -float(initial["top8_regret"]), best)
    history = [{"epoch": 0, "train_loss": None, **initial}]
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "meta": history[0], "args": vars(args), "variant": args.variant},
        args.out_ckpt,
    )
    print(json.dumps(history[0]), flush=True)
    print("saved epoch0", args.out_ckpt, "ndcg8", best, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            B, C = batch["y"].shape
            flat = {
                "X": batch["X"].reshape(B * C, *batch["X"].shape[2:]).to(device),
                "A": batch["A"].reshape(B * C, *batch["A"].shape[2:]).to(device),
                "action": batch["action"].reshape(B * C, -1).to(device),
                "global": batch["global"].reshape(B * C, -1).to(device),
                "mask": batch["mask"].reshape(B * C, -1).to(device),
            }
            pred = model(flat)
            for k in pred:
                pred[k] = pred[k].reshape(B, C)
            y = batch["y"].to(device)
            yd = batch["y_drive"].to(device)
            ys = batch["y_swap"].to(device)
            ysyn = batch["y_sync"].to(device)
            soft = batch["soft"].to(device)
            teach = batch["teacher"].to(device)
            feasible = batch["feasible"].to(device)
            candidate = batch["cand_mask"].to(device)
            teacher_mask = batch["teacher_mask"].to(device)
            teacher_group = batch["teacher_group"].to(device)

            mse_sum = flat["X"].new_zeros(())
            rank_sum = flat["X"].new_zeros(())
            kd_sum = flat["X"].new_zeros(())
            soft_sum = flat["X"].new_zeros(())
            mse_terms = 0
            rank_terms = 0
            kd_terms = 0
            soft_terms = 0
            for i in range(B):
                v_feasible = feasible[i] & candidate[i]
                v_all = candidate[i]
                pi = {k: pred[k][i] for k in pred}
                if v_feasible.sum() >= 2:
                    mse_i, _ = three_way_loss(pi, y[i], yd[i], ys[i], ysyn[i], v_feasible)
                    mse_sum = mse_sum + mse_i
                    mse_terms += 1
                if v_all.sum() >= 2 and v_feasible.any():
                    rank_target = y[i].clone()
                    feasible_values = rank_target[v_feasible]
                    span = (feasible_values.max() - feasible_values.min()).abs().clamp_min(1.0)
                    floor = feasible_values.min() - span
                    rank_target[v_all & ~feasible[i]] = floor
                    rank_sum = rank_sum + listwise_ranking_loss(
                        pred["gain"][i], rank_target, v_all
                    )
                    rank_terms += 1
                use_global = args.variant in (
                    "overlap_math_kd",
                    "overlap_general_kd",
                    "overlap_ensemble_kd",
                    "full_v3",
                )
                if args.variant == "exact_softlabel" and v_all.sum() >= 2:
                    kd_i = listwise_probability_loss(pred["gain"][i], soft[i], v_all)
                    if torch.isfinite(kd_i):
                        kd_sum = kd_sum + kd_i
                        kd_terms += 1
                if args.variant == "full_v3" and v_all.sum() >= 2:
                    kd_i = listwise_probability_loss(pred["gain"][i], soft[i], v_all)
                    if torch.isfinite(kd_i):
                        soft_sum = soft_sum + kd_i
                        soft_terms += 1
                if use_global:
                    v_teacher = teacher_mask[i] & v_all
                    if v_teacher.sum() >= 2:
                        teacher_values = teach[i][v_teacher] / max(args.kd_temp, 1e-6)
                        if (teacher_values.max() - teacher_values.min()).abs() > 1e-6:
                            kd_sum = kd_sum + listwise_ranking_loss(
                                pred["gain"][i], teach[i] / max(args.kd_temp, 1e-6), v_teacher
                            )
                            kd_terms += 1
                elif args.variant in ("qwen_general_kd", "qwen_math_kd"):
                    groups = torch.unique(teacher_group[i][teacher_mask[i] & v_all])
                    for group_id in groups:
                        v_teacher = teacher_mask[i] & v_all & (teacher_group[i] == group_id)
                        if v_teacher.sum() < 2:
                            continue
                        teacher_values = teach[i][v_teacher]
                        if (teacher_values.max() - teacher_values.min()).abs() <= 1e-6:
                            continue
                        kd_sum = kd_sum + listwise_ranking_loss(
                            pred["gain"][i], teach[i], v_teacher
                        )
                        kd_terms += 1
            if rank_terms == 0:
                continue
            feas_logits = pred["feas"][candidate]
            feas_labels = feasible[candidate].float()
            feas_loss = F.binary_cross_entropy_with_logits(feas_logits, feas_labels)
            loss = args.rank_w * (rank_sum / rank_terms) + args.feas_w * feas_loss
            if mse_terms:
                loss = loss + mse_sum / mse_terms
            if args.variant != "exact_only" and kd_terms:
                loss = loss + args.kd_w * (kd_sum / kd_terms)
            if soft_terms:
                loss = loss + args.soft_w * (soft_sum / soft_terms)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))

        metrics = eval_loader(model, val_loader, device, args.online_feas_weight)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, **metrics}
        history.append(row)
        print(json.dumps(row))
        metric_key = (
            float(metrics["top8_recall"]),
            -float(metrics["top8_regret"]),
            float(metrics["ndcg8"]),
        )
        if args.final_all_data or (math.isfinite(metrics["ndcg8"]) and metric_key >= best_key):
            best = metrics["ndcg8"]
            best_key = metric_key
            torch.save(
                {"model": model.state_dict(), "meta": row, "args": vars(args), "variant": args.variant},
                args.out_ckpt,
            )
            print("saved", args.out_ckpt, "ndcg8", best)

    hist_path = args.out_ckpt.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("DONE", args.variant, "best_ndcg8", best)


if __name__ == "__main__":
    main()

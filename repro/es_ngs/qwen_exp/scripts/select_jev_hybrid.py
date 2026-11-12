"""Select one validation-only blend between structural and Qwen decision scores."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def ndcg8(scores, gains, feasible):
    relevance = np.where(feasible, np.maximum(gains, 0.0), 0.0)
    ideal = np.sort(relevance)[::-1][:8]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    if idcg <= 1e-12:
        return float("nan")
    order = np.argsort(-scores)[:8]
    return float(np.sum(relevance[order] / np.log2(np.arange(2, len(order) + 2))) / idcg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--prior-critic", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--esngs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scales", default="0,0.01,0.025,0.05,0.1,0.2,0.5")
    args = parser.parse_args()
    sys.path.insert(0, str(args.esngs_root))
    from jev_decision import build_decision_input, forward_decisions, load_jev_decision
    from model import ESNGSCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, backbone, heads, blob = load_jev_decision(args.qwen_manifest, device)
    prior_blob = torch.load(args.prior_critic, map_location=device, weights_only=False)
    prior = ESNGSCritic().to(device)
    prior.load_state_dict(prior_blob["model"], strict=True); prior.eval()
    prior_feas_weight = float((prior_blob.get("args") or {}).get("online_feas_weight", 1.0))
    ids_frame = pd.read_csv(args.state_index)
    state_ids = ids_frame.loc[ids_frame["split"] == "val", "state_id"].astype(int).tolist()
    records = []
    with torch.inference_mode():
        for sid in state_ids:
            root = args.label_dir / "states" / f"s{sid}"
            loadf = lambda name: np.nan_to_num(np.load(root / f"{name}.npy").astype(np.float32))
            action, glob, X, mask = loadf("action"), loadf("global"), loadf("X"), loadf("mask")
            A = np.load(root / "A.npy").astype(np.int64)
            y, feasible = loadf("y"), loadf("feas") > 0.5
            token_ids, positions, order = build_decision_input(
                tokenizer, action, glob, None, int(blob.get("max_length", 4096)),
                str(blob.get("position_mode", "candidate")),
            )
            numeric = torch.tensor(np.concatenate([action[order], glob[order]], axis=1), device=device)
            qwen = forward_decisions(
                backbone, heads, token_ids.unsqueeze(0).to(device), positions.to(device), numeric
            )["choice"].float().cpu().numpy()
            p = prior({
                "X": torch.tensor(X, device=device), "A": torch.tensor(A, device=device),
                "action": torch.tensor(action, device=device), "global": torch.tensor(glob, device=device),
                "mask": torch.tensor(mask, device=device),
            })
            prior_score = p["gain"]
            if prior_feas_weight:
                prior_score = prior_score + prior_feas_weight * torch.log(
                    torch.sigmoid(p["feas"]).clamp_min(1e-6)
                )
            records.append((y[order], feasible[order], prior_score.float().cpu().numpy()[order], qwen))

    curve = []
    for scale in [float(x) for x in args.scales.split(",")]:
        ndcgs, recalls, regrets = [], [], []
        for gains, feasible, structural, qwen in records:
            scores = structural + scale * qwen
            metric = ndcg8(scores, gains, feasible)
            if math.isfinite(metric): ndcgs.append(metric)
            valid_gain = np.where(feasible, gains, -np.inf)
            best = int(np.argmax(valid_gain)); selected = np.argsort(-scores)[:8]
            recalls.append(float(best in selected))
            regrets.append(max(0.0, float(valid_gain[best] - np.max(valid_gain[selected]))))
        curve.append({
            "scale": scale, "ndcg8": float(np.mean(ndcgs)),
            "top8_recall": float(np.mean(recalls)), "regret8": float(np.mean(regrets)),
        })
    selected = sorted(curve, key=lambda row: (-row["ndcg8"], row["scale"]))[0]
    scale = selected["scale"]
    with torch.no_grad():
        inverse = -20.0 if scale <= 0 else math.log(math.expm1(scale))
        heads.residual_log_scale.fill_(inverse)
    blob["heads"] = {key: value.detach().cpu() for key, value in heads.state_dict().items()}
    blob["prior_critic"] = str(args.prior_critic)
    blob["selected_residual_scale"] = scale
    blob["metrics"] = {**selected, "selection": "max validation NDCG@8; ties use smaller scale"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "jev_qwen_math.pt"
    torch.save(blob, output)
    report = {"validation_states": len(records), "curve": curve, "selected": selected, "manifest": str(output)}
    (args.out_dir / "blend_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Export direct Jev-style decision utilities for compiling into a fast scorer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--esngs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.esngs_root))
    from jev_decision import build_decision_input, forward_decisions, load_jev_decision
    from model import ESNGSCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, backbone, heads, blob = load_jev_decision(args.manifest, device)
    prior_path = Path(str(blob["prior_critic"]))
    prior_blob = torch.load(prior_path, map_location=device, weights_only=False)
    prior = ESNGSCritic().to(device)
    prior.load_state_dict(prior_blob["model"], strict=True)
    prior.eval()
    prior_feas_weight = float((prior_blob.get("args") or {}).get("online_feas_weight", 1.0))

    index = pd.read_csv(args.state_index)
    rows = []
    split_counts: dict[str, int] = {}
    with torch.inference_mode():
        for pos, item in enumerate(index.itertuples(index=False), start=1):
            sid, split = int(item.state_id), str(item.split)
            root = args.label_dir / "states" / f"s{sid}"
            loadf = lambda name: np.nan_to_num(np.load(root / f"{name}.npy").astype(np.float32))
            action, glob, X, mask = loadf("action"), loadf("global"), loadf("X"), loadf("mask")
            A = np.load(root / "A.npy").astype(np.int64)
            token_ids, positions, order = build_decision_input(
                tokenizer, action, glob, None, int(blob.get("max_length", 4096)),
                str(blob.get("position_mode", "candidate")),
            )
            numeric = torch.tensor(np.concatenate([action[order], glob[order]], axis=1), device=device)
            qwen = forward_decisions(
                backbone, heads, token_ids.unsqueeze(0).to(device), positions.to(device), numeric
            )["choice"].float().cpu().numpy()
            pred = prior({
                "X": torch.tensor(X, device=device), "A": torch.tensor(A, device=device),
                "action": torch.tensor(action, device=device), "global": torch.tensor(glob, device=device),
                "mask": torch.tensor(mask, device=device),
            })
            structural = pred["gain"]
            if prior_feas_weight:
                structural = structural + prior_feas_weight * torch.log(
                    torch.sigmoid(pred["feas"]).clamp_min(1e-6)
                )
            structural = structural.float().cpu().numpy()[order]
            scale = float(blob.get("selected_residual_scale", 0.0))
            utility = structural + scale * qwen
            for rank_pos, candidate_zero in enumerate(order):
                rows.append({
                    "state_id": sid, "candidate_id": int(candidate_zero) + 1,
                    "teacher": "jev_direct", "teacher_utility": float(utility[rank_pos]),
                    "teacher_confidence": 1.0, "teacher_valid": 1, "split": split,
                })
            split_counts[split] = split_counts.get(split, 0) + 1
            if pos % 25 == 0:
                print(f"exported {pos}/{len(index)} states", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "states": len(index), "rows": len(rows), "splits": split_counts,
        "source_manifest": str(args.manifest),
        "residual_scale": float(blob.get("selected_residual_scale", 0.0)),
        "role": "direct decision utilities; no natural-language teacher labels",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

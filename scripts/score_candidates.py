"""Rank a saved pre-repair candidate pool with the released deployment model."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro/es_ngs"))
from model import ESNGSCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/training/states/s1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/es_ngs_dm.pt")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ESNGSCritic().eval()
    model.load_state_dict(blob["model"], strict=True)
    batch = {}
    for name in ("X", "A", "action", "global", "mask"):
        dtype = np.int64 if name == "A" else np.float32
        batch[name] = torch.from_numpy(np.load(args.features / f"{name}.npy").astype(dtype))
    with torch.inference_mode():
        pred = model(batch)
        weight = float(blob.get("args", {}).get("online_feas_weight", 1.0))
        score = pred["gain"] + weight * torch.log(torch.sigmoid(pred["feas"]).clamp_min(1e-6))
    values = score.numpy()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite scores")
    order = np.argsort(-values, kind="stable")[:args.top_k]
    result = [{"candidate_id": int(i) + 1, "score": float(values[i]),
               "predicted_gain_pct": float(pred["gain"][i]),
               "repairability": float(torch.sigmoid(pred["feas"][i]))} for i in order]
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

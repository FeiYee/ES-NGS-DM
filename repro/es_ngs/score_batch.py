"""Score candidate features in a directory; write scores.npy (gain)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import ESNGSCritic  # noqa: E402

DEFAULT_CKPT = ROOT / "experiments" / "checkpoints" / "critic_best.pt"


def load_model(ckpt: Path, device: torch.device):
    model = ESNGSCritic()
    has_feas = False
    online_feas_weight = 0.0
    if ckpt.is_file():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        state = blob["model"]
        has_feas = any(str(k).startswith("head_feas.") for k in state)
        model.load_state_dict(state, strict=has_feas)
        online_feas_weight = float((blob.get("args") or {}).get("online_feas_weight", 1.0)) if has_feas else 0.0
    model.to(device).eval()
    return model, has_feas, online_feas_weight


@torch.no_grad()
def score_dir(feat_dir: Path, ckpt: Path, out_path: Path | None = None) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, has_feas, online_feas_weight = load_model(ckpt, device)
    X = torch.from_numpy(np.load(feat_dir / "X.npy").astype(np.float32))
    A = torch.from_numpy(np.load(feat_dir / "A.npy").astype(np.int64))
    action = torch.from_numpy(np.load(feat_dir / "action.npy").astype(np.float32))
    glob = torch.from_numpy(np.load(feat_dir / "global.npy").astype(np.float32))
    mask = torch.from_numpy(np.load(feat_dir / "mask.npy").astype(np.float32))
    batch = {"X": X.to(device), "A": A.to(device), "action": action.to(device), "global": glob.to(device), "mask": mask.to(device)}
    pred = model(batch)
    ranking_score = pred["gain"]
    if has_feas:
        ranking_score = ranking_score + online_feas_weight * torch.log(
            torch.sigmoid(pred["feas"]).clamp_min(1e-6)
        )
    gain = ranking_score.cpu().numpy().astype(np.float64)
    out = out_path or (feat_dir / "scores.npy")
    np.save(out, gain)
    return gain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--out", default="")
    args = p.parse_args()
    out = Path(args.out) if args.out else None
    g = score_dir(Path(args.feat_dir), Path(args.ckpt), out)
    print("scored", len(g), "max", float(np.nanmax(g)) if len(g) else None)


if __name__ == "__main__":
    main()

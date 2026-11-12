"""Persistent critic scorer with optional Energy/Sync feature masks.

Protocol (stdin/stdout, one line each):
  LOAD <ckpt_path>
  SCORE <feat_dir> <out_scores_npy> [none|energy|sync]
  QUIT
Responses: READY / OK / ERR <msg>

Env:
  ESNGS_FORCE_CPU=1  — hide CUDA before importing torch (avoids driver init hangs)
  ESNGS_TORCH_THREADS — CPU thread count (default 1)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Force CPU *before* importing torch — CUDA driver init under SSH/multi-process
# was hanging LOAD on the eval host.
if os.environ.get("ESNGS_FORCE_CPU", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from model import ESNGSCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_KIND = "critic"
JEV_TOKENIZER = None
JEV_HEADS = None
JEV_MAX_LENGTH = 4096
JEV_POSITION_MODE = "query"
JEV_PRIOR = None
JEV_PRIOR_FEAS_WEIGHT = 0.0
MODEL_HAS_FEAS = False
ONLINE_FEAS_WEIGHT = 0.0
torch.set_num_threads(max(1, int(os.environ.get("ESNGS_TORCH_THREADS", "1"))))

# Channel conventions matching features.jl (0-indexed here)
# Energy: X[:, :, 4:8], action[:, 16:20], global[:, [3,4,5,10]]
# Sync:   X[:, :, 8:12], action[:, 24:28], global[:, [2,7,8,9]]


def _say(msg: str) -> None:
    print(msg, flush=True)


def load_ckpt(path: str):
    global MODEL, MODEL_KIND, JEV_TOKENIZER, JEV_HEADS, JEV_MAX_LENGTH, JEV_POSITION_MODE
    global JEV_PRIOR, JEV_PRIOR_FEAS_WEIGHT
    global MODEL_HAS_FEAS, ONLINE_FEAS_WEIGHT
    _say(f"LOADING {path} device={DEVICE}")
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    if blob.get("kind") == "jev_qwen_decision":
        from jev_decision import load_jev_decision

        JEV_TOKENIZER, MODEL, JEV_HEADS, blob = load_jev_decision(path, DEVICE)
        MODEL_KIND = "jev_qwen_decision"
        JEV_MAX_LENGTH = int(blob.get("max_length", 4096))
        JEV_POSITION_MODE = str(blob.get("position_mode", "query"))
        prior_path = str(blob.get("prior_critic", "") or "")
        JEV_PRIOR = None
        JEV_PRIOR_FEAS_WEIGHT = 0.0
        if prior_path:
            prior_blob = torch.load(prior_path, map_location=DEVICE, weights_only=False)
            JEV_PRIOR = ESNGSCritic().to(DEVICE)
            JEV_PRIOR.load_state_dict(prior_blob["model"], strict=True)
            JEV_PRIOR.eval()
            JEV_PRIOR_FEAS_WEIGHT = float(
                (prior_blob.get("args") or {}).get("online_feas_weight", 1.0)
            )
        MODEL_HAS_FEAS = True
        ONLINE_FEAS_WEIGHT = float(blob.get("online_feas_weight", 0.0))
        return True
    MODEL_KIND = "critic"
    JEV_TOKENIZER = None
    JEV_HEADS = None
    MODEL = ESNGSCritic().to(DEVICE)
    state = blob["model"]
    MODEL_HAS_FEAS = any(str(k).startswith("head_feas.") for k in state)
    MODEL.load_state_dict(state, strict=MODEL_HAS_FEAS)
    args = blob.get("args") or {}
    ONLINE_FEAS_WEIGHT = float(args.get("online_feas_weight", 1.0)) if MODEL_HAS_FEAS else 0.0
    MODEL.eval()
    return True


def apply_mask(X, action, glob, mask_mode: str):
    m = (mask_mode or "none").lower()
    if m not in ("energy", "wo_energy", "no_energy", "sync", "wo_sync", "no_sync"):
        return X, action, glob
    X = X.clone()
    action = action.clone()
    glob = glob.clone()
    if m in ("energy", "wo_energy", "no_energy"):
        X[:, :, 4:8] = 0
        action[:, 16:20] = 0
        glob[:, [3, 4, 5, 10]] = 0
    elif m in ("sync", "wo_sync", "no_sync"):
        X[:, :, 8:12] = 0
        action[:, 24:28] = 0
        glob[:, [2, 7, 8, 9]] = 0
    return X, action, glob


@torch.inference_mode()
def score(feat_dir: str, out_path: str, mask_mode: str = "none"):
    d = Path(feat_dir)
    X = torch.from_numpy(np.load(d / "X.npy").astype(np.float32)).to(DEVICE)
    A = torch.from_numpy(np.load(d / "A.npy").astype(np.int64)).to(DEVICE)
    action = torch.from_numpy(np.load(d / "action.npy").astype(np.float32)).to(DEVICE)
    glob = torch.from_numpy(np.load(d / "global.npy").astype(np.float32)).to(DEVICE)
    mask = torch.from_numpy(np.load(d / "mask.npy").astype(np.float32)).to(DEVICE)
    X, action, glob = apply_mask(X, action, glob, mask_mode)
    if MODEL_KIND == "jev_qwen_decision":
        from jev_decision import build_decision_input, forward_decisions

        ids, query_positions, _ = build_decision_input(
            JEV_TOKENIZER,
            action.cpu().numpy(),
            glob.cpu().numpy(),
            None,
            JEV_MAX_LENGTH,
            JEV_POSITION_MODE,
        )
        prior_choice = None
        if JEV_PRIOR is not None:
            prior = JEV_PRIOR({"X": X, "A": A, "action": action, "global": glob, "mask": mask})
            prior_choice = prior["gain"]
            if JEV_PRIOR_FEAS_WEIGHT:
                prior_choice = prior_choice + JEV_PRIOR_FEAS_WEIGHT * torch.log(
                    torch.sigmoid(prior["feas"]).clamp_min(1e-6)
                )
        pred = forward_decisions(
            MODEL,
            JEV_HEADS,
            ids.unsqueeze(0).to(DEVICE),
            query_positions.to(DEVICE),
            torch.cat([action, glob], dim=1),
            prior_choice,
        )
        ranking_score = pred["choice"]
        if ONLINE_FEAS_WEIGHT:
            feasible_prob = torch.sigmoid(pred["feasible"]).clamp_min(1e-6)
            ranking_score = ranking_score + ONLINE_FEAS_WEIGHT * torch.log(feasible_prob)
    else:
        pred = MODEL({"X": X, "A": A, "action": action, "global": glob, "mask": mask})
        ranking_score = pred["gain"]
        if MODEL_HAS_FEAS:
            feasible_prob = torch.sigmoid(pred["feas"]).clamp_min(1e-6)
            ranking_score = ranking_score + ONLINE_FEAS_WEIGHT * torch.log(feasible_prob)
    result = ranking_score.cpu().numpy().astype(np.float64)
    np.save(out_path, result)
    return len(result)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _say(f"READY device={DEVICE}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Tab-separated arguments preserve paths containing spaces. Legacy
        # whitespace commands remain supported for the original smoke script.
        parts = line.split("\t") if "\t" in line else line.split()
        cmd = parts[0].upper()
        try:
            if cmd == "LOAD":
                load_ckpt(parts[1])
                _say("OK")
            elif cmd == "SCORE":
                if MODEL is None:
                    _say("ERR model_not_loaded")
                    continue
                mask_mode = parts[3] if len(parts) > 3 else "none"
                n = score(parts[1], parts[2], mask_mode)
                _say(f"OK {n}")
            elif cmd == "QUIT":
                _say("OK")
                break
            else:
                _say(f"ERR unknown_cmd {cmd}")
        except Exception as e:
            _say(f"ERR {type(e).__name__}:{e}".replace("\n", " "))


if __name__ == "__main__":
    main()

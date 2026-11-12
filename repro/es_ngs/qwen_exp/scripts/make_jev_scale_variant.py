"""Create a validation-selected Qwen residual-scale variant without retraining."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    blob = torch.load(args.input, map_location="cpu", weights_only=False)
    if blob.get("kind") != "jev_qwen_decision":
        raise ValueError("input is not a Jev decision manifest")
    inverse = -20.0 if args.scale <= 0 else math.log(math.expm1(args.scale))
    state = dict(blob["heads"])
    state["residual_log_scale"] = torch.tensor(inverse, dtype=torch.float32)
    blob["heads"] = state
    blob["selected_residual_scale"] = float(args.scale)
    blob["scale_selection_basis"] = "validation top-8 shortlist regret/recall"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(f"WROTE {args.output} scale={args.scale}", flush=True)


if __name__ == "__main__":
    main()

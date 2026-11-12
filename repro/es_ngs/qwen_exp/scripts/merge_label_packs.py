"""Merge labels_v2 + labels_v2_replay into one pack with remapped state_ids."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def copy_states(src: Path, dst: Path, offset: int) -> int:
    n = 0
    for d in sorted(src.glob("s*"), key=lambda p: int(p.name[1:])):
        sid = int(d.name[1:])
        new_id = sid + offset
        dest = dst / f"s{new_id}"
        if dest.exists():
            raise FileExistsError(f"refusing to overwrite state directory: {dest}")
        shutil.copytree(d, dest)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="labels_v2")
    ap.add_argument("--replay", type=Path, required=True, help="labels_v2_replay")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("--out must be a new or empty directory")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "states").mkdir(exist_ok=True)
    (args.out / "instances").mkdir(exist_ok=True)

    base_lab = pd.read_csv(args.base / "LABELS.csv")
    base_max = int(base_lab["state_id"].max())
    # copy base states as-is
    n_base = copy_states(args.base / "states", args.out / "states", 0)
    # copy instances
    for p in (args.base / "instances").glob("*.jls"):
        shutil.copy2(p, args.out / "instances" / p.name)
    for p in (args.replay / "instances").glob("*.jls"):
        shutil.copy2(p, args.out / "instances" / p.name)

    replay_lab = pd.read_csv(args.replay / "LABELS.csv").copy()
    replay_lab["state_id"] = replay_lab["state_id"].astype(int) + base_max
    replay_lab["feat_dir"] = replay_lab["state_id"].map(lambda s: f"states/s{s}")
    n_rep = copy_states(args.replay / "states", args.out / "states", base_max)

    merged = pd.concat([base_lab, replay_lab], ignore_index=True)
    merged.to_csv(args.out / "LABELS.csv", index=False)
    meta = [
        f"total_labels={len(merged)}",
        f"n_states={int(merged['state_id'].nunique())}",
        f"n_base_states={n_base}",
        f"n_replay_states={n_rep}",
        f"base_max_state_id={base_max}",
        "feature_version=2",
        "merged_from=labels_v2+labels_v2_replay",
    ]
    (args.out / "META.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    print("MERGED", args.out, "states", n_base + n_rep, "base_max", base_max, flush=True)


if __name__ == "__main__":
    main()

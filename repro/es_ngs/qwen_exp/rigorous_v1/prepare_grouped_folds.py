from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.index)
    required = {"state_id", "instance_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")

    instances = sorted(frame["instance_id"].astype(str).unique())
    buckets: list[list[str]] = [[] for _ in range(5)]
    # Stable grouped split.  Sorting the SHA256 values avoids dependence on
    # filesystem/CSV row order, and keeps every instance wholly in one fold.
    ranked = sorted(instances, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    for position, instance_id in enumerate(ranked):
        buckets[position % 5].append(instance_id)

    manifest = {"source_index": str(args.index), "folds": {}}
    for fold, val_instances in enumerate(buckets, start=1):
        out = frame.copy()
        out["split"] = out["instance_id"].astype(str).isin(val_instances).map(
            {True: "val", False: "train"}
        )
        path = args.out / f"state_index_fold{fold}.csv"
        out.to_csv(path, index=False)
        train_instances = sorted(set(instances) - set(val_instances))
        manifest["folds"][str(fold)] = {
            "train_instances": train_instances,
            "val_instances": val_instances,
            "train_states": int((out["split"] == "train").sum()),
            "val_states": int((out["split"] == "val").sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.out / "FOLD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Aggregate seed means first, then report instance-level mean and sample SD."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not paths:
        p.error("no CSV records found")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "status" in frame:
        frame = frame.loc[frame["status"] == "ok"].copy()
    frame = frame.rename(columns={"makespan_h":"completion_h", "charged_online_s":"online_s"})
    if frame.duplicated(["case_id", "seed"]).any():
        raise ValueError("duplicate case/seed records; select one run per case and seed")
    keys = [k for k in ("completion_h", "truck_km", "online_s", "exact_evals") if k in frame]
    values = frame.groupby("case_id")[keys].mean()
    result = {"instances":len(values), "runs":len(frame),
              "seed_counts":frame.groupby("case_id")["seed"].nunique().to_dict(),
              "metrics":{k:{"mean":float(values[k].mean()), "sample_sd":float(values[k].std(ddof=1))} for k in keys}}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

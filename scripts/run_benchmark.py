"""Run ES-NGS-DM serially over the fixed paper evaluation lists."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["synthetic", "lade"], default="synthetic")
    p.add_argument("--julia", default="julia")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/benchmark")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--budget", type=int, default=60)
    args = p.parse_args()
    with (ROOT / "data/splits/paper_evaluation.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["benchmark"] == args.benchmark]
    if not rows:
        raise ValueError("empty evaluation list")
    for row in rows:
        for seed in args.seeds:
            command = [args.julia, f"--project={ROOT}", str(ROOT / "scripts/run_case.jl"),
                       "--benchmark", args.benchmark, "--case", row["case_id"],
                       "--seed", str(seed), "--budget", str(args.budget),
                       "--python", args.python, "--out", str(args.out.resolve())]
            print(f"RUN {row['case_id']} seed={seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

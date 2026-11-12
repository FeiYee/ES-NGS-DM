"""Portable fixed-epoch training on the released supplementary decision states."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "repro/es_ngs"
SCRIPTS = CODE / "qwen_exp/scripts"
DATA = ROOT / "data/training"


def launch(script, **kwargs):
    command = [sys.executable, str(SCRIPTS / script)]
    for key, value in kwargs.items():
        option = "--" + key.replace("_", "-")
        if value is True:
            command.append(option)
        elif value is not None and value is not False:
            command += [option, str(value)]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["base", "exact", "teacher", "export", "qwen"], required=True)
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument("--out", type=Path, default=ROOT / "outputs/training")
    p.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen2.5-1.5B")
    p.add_argument("--teacher-utilities", type=Path)
    args = p.parse_args()
    if args.fold and args.teacher_utilities:
        p.error("five-fold training must use its own fold-trained teacher, not all-data utilities")
    out = args.out.resolve() / (f"fold{args.fold}" if args.fold else "deployment")
    out.mkdir(parents=True, exist_ok=True)
    index = ROOT / f"data/splits/supplement/fold{args.fold}.csv" if args.fold else DATA / "state_index_all.csv"
    common = dict(es_ngs_root=CODE, label_dir=DATA, state_index=index)
    seed = 2026110800 + (args.fold or 6)
    if args.stage == "teacher":
        launch("train_jev_fixed_epoch.py", esngs_root=CODE, label_dir=DATA, state_index=index,
               base_model=args.base_model.resolve(), out_dir=out / "teacher", final_all_data=args.fold is None,
               prior_critic=out / "base.pt", epochs=4, lr=2e-4, head_lr=8e-4,
               grad_accum=8, lora_r=8, max_length=8192, layer_limit=8,
               position_mode="candidate", choice_temp=1, seed=seed)
    elif args.stage == "export":
        launch("export_jev_decision_utilities.py", esngs_root=CODE, label_dir=DATA, state_index=index,
               manifest=out / "teacher/jev_qwen_math.pt", output=out / "teacher_utilities.csv")
    else:
        base = args.stage == "base"
        qwen = args.stage == "qwen"
        launch("train_student_fixed_epoch.py", **common,
               variant="full_v3", soft_csv=DATA / "exact_soft_labels.csv",
               teacher_csv=(args.teacher_utilities or out / "teacher_utilities.csv") if qwen else None,
               teacher_name="jev_direct" if qwen else "", out_ckpt=out / f"{args.stage}.pt",
               init_ckpt=None if base else out / "base.pt",
               epochs=40 if base else 12, batch_size=4, lr=2e-4 if base else 2e-5,
               rank_w=1, soft_w=0.5, kd_w=0.1 if qwen else 0,
               feas_w=0.5, online_feas_weight=1, kd_temp=1, seed=seed,
               final_all_data=args.fold is None)


if __name__ == "__main__":
    main()

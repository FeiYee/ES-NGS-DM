"""Re-score formal_v2b teachers on overlapping 8-candidate groups. No new SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decision_prompt import build_decision_prompt, summarize_action, summarize_global  # noqa: E402
from export_teacher_prefs import infer_scores, ndcg_at_k  # noqa: E402
from overlap_groups import assert_overlap_invariants, overlapping_groups  # noqa: E402

ROOT = Path(os.environ.get("ESNGS_QWEN_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(ROOT / "pydeps"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--state-index", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--teacher-name", type=str, required=True)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--start-stride", type=int, default=3)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(args.adapter), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    idx = pd.read_csv(args.state_index)
    lab = pd.read_csv(args.label_dir / "LABELS.csv")
    groups_by_c: dict[int, list] = {}

    def groups_for(n_cands: int):
        if n_cands not in groups_by_c:
            g = overlapping_groups(n_cands, args.group_size, args.start_stride)
            inv = assert_overlap_invariants(g, n_cands)
            print("OVERLAP", n_cands, inv, flush=True)
            groups_by_c[n_cands] = g
        return groups_by_c[n_cands]

    rows_out = []
    for _, row in idx.iterrows():
        sid = int(row["state_id"])
        d = args.label_dir / "states" / f"s{sid}"
        action = np.load(d / "action.npy")
        glob = np.load(d / "global.npy")
        y = np.load(d / "y.npy").astype(np.float64)
        feas = np.load(d / "feas.npy") > 0.5
        C = min(len(y), action.shape[0], len(feas))
        groups = groups_for(C)
        meta_row = lab[lab["state_id"] == sid].iloc[0]
        meta = {
            "instance_id": str(meta_row["instance_id"]),
            "n": int(meta_row["n"]),
            "family": str(meta_row["family"]),
            "phase": str(meta_row["phase"]),
            "before_T_h": float(meta_row["before_makespan_h"]),
            "global": summarize_global(glob[0] if glob.ndim == 2 else glob),
        }
        for group_id, group in enumerate(groups):
            group_idx = [cid - 1 for cid in group]
            candidate_ids = list(group)
            cands = [{"cid": int(cid), "action": summarize_action(action[cid - 1])} for cid in group]
            prompt = build_decision_prompt(meta, cands)
            pred, prob, audit = infer_scores(model, tok, prompt, candidate_ids, device)
            exact_gain = np.asarray(
                [y[i] if feas[i] and np.isfinite(y[i]) else np.nan for i in group_idx],
                dtype=np.float64,
            )
            valid_exact = np.isfinite(exact_gain)
            if valid_exact.any():
                span = max(float(np.ptp(exact_gain[valid_exact])), 1.0)
                floor = float(np.min(exact_gain[valid_exact]) - span)
                y_group = np.where(valid_exact, exact_gain - floor, 0.0)
            else:
                y_group = np.zeros(len(group_idx), dtype=np.float64)
            target_informative = int(float(np.ptp(y_group)) > 1e-12)
            nd = (
                ndcg_at_k(pred, y_group, np.ones(len(group_idx), dtype=bool), k=8)
                if audit["schema_valid"] and target_informative
                else float("nan")
            )
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for pos, original_i in enumerate(group_idx):
                rows_out.append(
                    {
                        "teacher": args.teacher_name,
                        "state_id": sid,
                        "instance_id": meta["instance_id"],
                        "split": row["split"],
                        "teacher_group": group_id,
                        "candidate_id": int(original_i) + 1,
                        "teacher_score": float(pred[pos]),
                        "teacher_prob": float(prob[pos]),
                        "teacher_valid": audit["schema_valid"],
                        "parse_valid": audit["parse_valid"],
                        "nonconstant": audit["nonconstant"],
                        "score_source": audit["score_source"],
                        "ndcg8_group": nd,
                        "target_informative": target_informative,
                        "prompt_sha256": prompt_sha,
                        "response_sha256": audit["response_sha256"],
                        "grouping": "overlap_circular_v1",
                    }
                )
            print(json.dumps({"state_id": sid, "group": group_id, "ndcg8": nd, **audit}), flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
    print("DONE_OVERLAP", args.teacher_name, "rows", len(rows_out), flush=True)


if __name__ == "__main__":
    main()

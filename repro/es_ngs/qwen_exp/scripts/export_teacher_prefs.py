"""Export teacher preference scores on labeled states for KD + offline NDCG@8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from decision_prompt import build_decision_prompt, summarize_action, summarize_global

ROOT = Path(os.environ.get("ESNGS_QWEN_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(ROOT / "pydeps"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))


def ndcg_at_k(scores, targets, valid, k=8) -> float:
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return float("nan")
    s, t = scores[idx], targets[idx]
    k = min(k, len(idx))
    order = np.argsort(-s)[:k]
    gains = np.maximum(t[order], 0.0)
    dcg = float(np.sum(gains / np.log2(np.arange(2, k + 2))))
    ideal = np.sort(np.maximum(t, 0.0))[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 1e-9 else float("nan")


def parse_json_blob(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@torch.no_grad()
def infer_scores(model, tok, prompt: str, candidate_ids: List[int], device):
    messages = [
        {"role": "system", "content": "You output JSON only for BEVTSPD candidate ranking."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    blob = parse_json_blob(gen)
    n_cands = len(candidate_ids)
    scores = np.full(n_cands, np.nan, dtype=np.float64)
    probs = np.full(n_cands, np.nan, dtype=np.float64)
    audit = {
        "parse_valid": int(isinstance(blob, dict)),
        "schema_valid": 0,
        "choice_valid": 0,
        "nonconstant": 0,
        "score_source": "invalid",
        "response_sha256": hashlib.sha256(gen.encode("utf-8")).hexdigest(),
    }
    if not isinstance(blob, dict):
        return scores, probs, audit
    score_list = blob.get("score") or []
    prob_list = blob.get("probability") or []
    choice = blob.get("choice") or []
    if isinstance(score_list, list) and len(score_list) == n_cands:
        for i, v in enumerate(score_list):
            try:
                scores[i] = float(v)
            except Exception:
                pass
    id_to_pos = {int(cid): i for i, cid in enumerate(candidate_ids)}
    choice_ids = []
    if isinstance(choice, list):
        for cid in choice:
            try:
                parsed = int(cid)
            except Exception:
                continue
            if parsed in id_to_pos and parsed not in choice_ids:
                choice_ids.append(parsed)
    choice_valid = len(choice_ids) == n_cands and set(choice_ids) == set(candidate_ids)
    audit["choice_valid"] = int(choice_valid)
    numeric_complete = bool(np.isfinite(scores).all())
    numeric_nonconstant = numeric_complete and float(np.ptp(scores)) > 1e-12
    if choice_valid and not numeric_nonconstant:
        for rank, cid in enumerate(choice_ids):
            scores[id_to_pos[cid]] = float(n_cands - rank)
        audit["score_source"] = "choice_order"
    elif numeric_complete:
        audit["score_source"] = "score_list"
    if isinstance(prob_list, list) and len(prob_list) == n_cands:
        for i, v in enumerate(prob_list):
            try:
                probs[i] = float(v)
            except Exception:
                pass
    audit["schema_valid"] = int(bool(np.isfinite(scores).all()))
    audit["nonconstant"] = int(
        bool(np.isfinite(scores).all()) and float(np.ptp(scores)) > 1e-12
    )
    return scores, probs, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--state-index", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--teacher-name", type=str, required=True)
    ap.add_argument("--max-cands", type=int, default=8, help="maximum candidates in one teacher prompt")
    ap.add_argument("--coverage", choices=["all", "first"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_cands < 2:
        raise SystemExit("--max-cands must be at least 2")

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
    rows_out = []
    metrics = []
    for _, row in idx.iterrows():
        sid = int(row["state_id"])
        d = args.label_dir / "states" / f"s{sid}"
        action = np.load(d / "action.npy")
        glob = np.load(d / "global.npy")
        y = np.load(d / "y.npy").astype(np.float64)
        feas = np.load(d / "feas.npy") > 0.5
        C = min(len(y), action.shape[0], len(feas))
        meta_row = lab[lab["state_id"] == sid].iloc[0]
        meta = {
            "instance_id": str(meta_row["instance_id"]),
            "n": int(meta_row["n"]),
            "family": str(meta_row["family"]),
            "phase": str(meta_row["phase"]),
            "before_T_h": float(meta_row["before_makespan_h"]),
            "global": summarize_global(glob[0] if glob.ndim == 2 else glob),
        }
        rng = np.random.default_rng(args.seed + sid * 1_000_003)
        candidate_order = rng.permutation(C) if args.coverage == "all" else np.arange(C)
        if args.coverage == "first":
            candidate_order = candidate_order[: args.max_cands]
        for group_id, start in enumerate(range(0, len(candidate_order), args.max_cands)):
            group_idx = candidate_order[start : start + args.max_cands]
            if len(group_idx) < 2:
                continue
            candidate_ids = [int(i) + 1 for i in group_idx]
            cands = [
                {
                    "cid": int(i) + 1,
                    "action": summarize_action(action[int(i)]),
                }
                for i in group_idx
            ]
            prompt = build_decision_prompt(meta, cands)
            pred, prob, audit = infer_scores(model, tok, prompt, candidate_ids, device)
            exact_gain = np.asarray(
                [y[int(i)] if feas[int(i)] and np.isfinite(y[int(i)]) else np.nan for i in group_idx],
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
            position_nd = ndcg_at_k(
                -np.arange(len(group_idx), dtype=np.float64),
                y_group,
                np.ones(len(group_idx), dtype=bool),
                k=8,
            )
            random_scores = np.random.default_rng(
                args.seed + sid * 1_000_003 + group_id * 9_176
            ).random(len(group_idx))
            random_nd = ndcg_at_k(
                random_scores, y_group, np.ones(len(group_idx), dtype=bool), k=8
            )
            group_metric = {
                "state_id": sid,
                "split": row["split"],
                "teacher_group": group_id,
                "ndcg8": nd,
                "position_baseline_ndcg8": position_nd,
                "random_baseline_ndcg8": random_nd,
                "target_informative": target_informative,
                **audit,
            }
            metrics.append(group_metric)
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for pos, original_i in enumerate(group_idx):
                i = int(original_i)
                rows_out.append(
                    {
                        "teacher": args.teacher_name,
                        "state_id": sid,
                        "instance_id": meta["instance_id"],
                        "split": row["split"],
                        "teacher_group": group_id,
                        "candidate_id": i + 1,
                        "position_in_group": pos,
                        "y_pct": float(y[i]) if np.isfinite(y[i]) else float("nan"),
                        "feasible": int(feas[i]),
                        "teacher_score": float(pred[pos]),
                        "teacher_prob": float(prob[pos]),
                        "teacher_valid": audit["schema_valid"],
                        "parse_valid": audit["parse_valid"],
                        "choice_valid": audit["choice_valid"],
                        "nonconstant": audit["nonconstant"],
                        "score_source": audit["score_source"],
                        "ndcg8_group": nd,
                        "position_baseline_ndcg8": position_nd,
                        "random_baseline_ndcg8": random_nd,
                        "target_informative": target_informative,
                        "prompt_sha256": prompt_sha,
                        "response_sha256": audit["response_sha256"],
                    }
                )
            print(json.dumps({"state_id": sid, "group": group_id, "ndcg8": nd, **audit}))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out_csv, index=False)
    mdf = pd.DataFrame(metrics)
    valid_scores = (
        pd.to_numeric(out_df["teacher_score"], errors="coerce")
        if "teacher_score" in out_df
        else pd.Series(dtype=float)
    )
    summary = {
        "teacher": args.teacher_name,
        "n_states": int(idx["state_id"].nunique()),
        "n_groups": int(len(mdf)),
        "coverage_mode": args.coverage,
        "group_size": int(args.max_cands),
        "parse_valid_rate": float(mdf["parse_valid"].mean()) if len(mdf) else 0.0,
        "schema_valid_rate": float(mdf["schema_valid"].mean()) if len(mdf) else 0.0,
        "nonconstant_rate": float(mdf["nonconstant"].mean()) if len(mdf) else 0.0,
        "candidate_score_coverage": float(valid_scores.notna().mean()) if len(valid_scores) else 0.0,
        "n_ndcg_groups": int(mdf["ndcg8"].notna().sum()) if len(mdf) else 0,
        "ndcg8_mean": float(mdf["ndcg8"].mean(skipna=True)) if len(mdf) else float("nan"),
        "ndcg8_val": float(mdf.loc[mdf["split"] == "val", "ndcg8"].mean(skipna=True)) if (mdf["split"] == "val").any() else float("nan"),
        "position_baseline_ndcg8": float(mdf["position_baseline_ndcg8"].mean(skipna=True)) if len(mdf) else float("nan"),
        "random_baseline_ndcg8": float(mdf["random_baseline_ndcg8"].mean(skipna=True)) if len(mdf) else float("nan"),
    }
    (args.out_csv.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", summary)


if __name__ == "__main__":
    main()

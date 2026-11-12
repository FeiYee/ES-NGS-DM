"""Train a Qwen-backed, Jev-style ES-NGS decision model directly on exact outcomes."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


def add_root(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class DecisionStates(Dataset):
    def __init__(self, state_ids, label_dir: Path):
        self.state_ids = list(map(int, state_ids))
        self.label_dir = label_dir

    def __len__(self):
        return len(self.state_ids)

    def __getitem__(self, index):
        sid = self.state_ids[index]
        root = self.label_dir / "states" / f"s{sid}"

        def load(name, dtype=np.float32):
            value = np.load(root / f"{name}.npy").astype(dtype)
            if np.issubdtype(value.dtype, np.floating):
                value[~np.isfinite(value)] = 0.0
            return value

        return {
            "sid": sid,
            "action": load("action"),
            "global": load("global"),
            "y": load("y"),
            "drive": load("y_drive"),
            "swap": load("y_swap"),
            "sync": load("y_sync"),
            "feasible": load("feas") > 0.5,
            "X": load("X"),
            "A": load("A", np.int64),
            "mask": load("mask"),
        }


def ndcg8(scores, gains, feasible):
    relevance = np.where(feasible, np.maximum(gains, 0.0), 0.0)
    ideal = np.sort(relevance)[::-1][:8]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    if idcg <= 1e-12:
        return float("nan")
    order = np.argsort(-scores)[:8]
    dcg = float(np.sum(relevance[order] / np.log2(np.arange(2, len(order) + 2))))
    return dcg / idcg


def evaluate(
    dataset, tokenizer, backbone, heads, build_input, forward_fn, device,
    max_length, position_mode, prior_model=None, prior_feas_weight=0.0,
):
    backbone.eval(); heads.eval()
    ndcgs, recalls, regrets, top1s, latencies = [], [], [], [], []
    with torch.inference_mode():
        for row in dataset:
            ids, qpos, order = build_input(
                tokenizer, row["action"], row["global"], None, max_length, position_mode
            )
            ids = ids.unsqueeze(0).to(device); qpos = qpos.to(device)
            numeric = torch.tensor(
                np.concatenate([row["action"][order], row["global"][order]], axis=1),
                device=device, dtype=torch.float32,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prior_choice = None
            if prior_model is not None:
                prior = prior_model({
                    "X": torch.tensor(row["X"], device=device),
                    "A": torch.tensor(row["A"], device=device),
                    "action": torch.tensor(row["action"], device=device),
                    "global": torch.tensor(row["global"], device=device),
                    "mask": torch.tensor(row["mask"], device=device),
                })
                prior_choice = prior["gain"]
                if prior_feas_weight:
                    prior_choice = prior_choice + prior_feas_weight * torch.log(
                        torch.sigmoid(prior["feas"]).clamp_min(1e-6)
                    )
            out = forward_fn(backbone, heads, ids, qpos, numeric, prior_choice)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            scores = out["choice"].float().cpu().numpy()
            gains = row["y"][order]
            feasible = row["feasible"][order]
            nd = ndcg8(scores, gains, feasible)
            if math.isfinite(nd):
                ndcgs.append(nd)
            valid_gain = np.where(feasible, gains, -np.inf)
            best = int(np.argmax(valid_gain))
            selected = np.argsort(-scores)[: min(8, len(scores))]
            recalls.append(float(best in selected))
            selected_best = float(np.max(valid_gain[selected]))
            regrets.append(max(0.0, float(valid_gain[best] - selected_best)))
            top1s.append(float(int(np.argmax(scores)) == best))
    return {
        "ndcg8": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "top8_recall": float(np.mean(recalls)),
        "regret8": float(np.mean(regrets)),
        "top1": float(np.mean(top1s)),
        "latency_ms_median": 1000.0 * float(np.median(latencies)),
        "states": len(dataset),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--esngs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-all-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--choice-temp", type=float, default=1.0)
    parser.add_argument("--position-mode", choices=["query", "candidate"], default="candidate")
    parser.add_argument("--prior-critic", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-states", type=int, default=0)
    parser.add_argument("--max-val-states", type=int, default=0)
    parser.add_argument(
        "--layer-limit", type=int, default=8,
        help="Use the first N pretrained Qwen layers for a real-time decision backbone; 0 keeps all layers.",
    )
    args = parser.parse_args()

    add_root(args.esngs_root)
    from jev_decision import (  # noqa: WPS433
        JevDecisionHeads,
        build_decision_input,
        forward_decisions,
        write_metadata,
    )
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for Qwen decision-head training")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    index = pd.read_csv(args.state_index)
    train_ids = index.loc[index["split"] == "train", "state_id"].astype(int).tolist()
    val_ids = index.loc[index["split"] == "val", "state_id"].astype(int).tolist()
    if args.final_all_data:
        train_ids = index["state_id"].astype(int).unique().tolist()
        val_ids = list(train_ids)  # training diagnostics only
    if args.max_train_states:
        train_ids = train_ids[: args.max_train_states]
    if args.max_val_states:
        val_ids = val_ids[: args.max_val_states]
    train_ds = DecisionStates(train_ids, args.label_dir)
    val_ds = DecisionStates(val_ids, args.label_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if args.layer_limit:
        if not hasattr(base, "layers"):
            raise ValueError("Qwen backbone does not expose a layers ModuleList")
        base.layers = nn.ModuleList(list(base.layers[: args.layer_limit]))
        base.config.num_hidden_layers = args.layer_limit
    base.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    backbone = get_peft_model(base, lora).to(device)
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()
    heads = JevDecisionHeads(int(base.config.hidden_size), numeric_dim=48).to(device)
    prior_model = None
    prior_feas_weight = 0.0
    if args.prior_critic:
        from model import ESNGSCritic

        prior_blob = torch.load(args.prior_critic, map_location=device, weights_only=False)
        prior_model = ESNGSCritic().to(device)
        prior_model.load_state_dict(prior_blob["model"], strict=True)
        prior_model.eval()
        for parameter in prior_model.parameters():
            parameter.requires_grad_(False)
        prior_feas_weight = float((prior_blob.get("args") or {}).get("online_feas_weight", 1.0))
        # Start exactly from the validated structural critic.  Qwen learns a
        # residual decision correction, so an untrained adapter cannot degrade
        # the baseline merely through random head initialization.
        with torch.no_grad():
            heads.choice.weight.zero_(); heads.choice.bias.zero_()
            heads.numeric[-1].weight[0].zero_(); heads.numeric[-1].bias[0].zero_()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fail early on context overflow and record the actual state/candidate counts.
    token_lengths = []
    candidate_counts = []
    for row in list(train_ds) + list(val_ds):
        ids, _, _ = build_decision_input(
            tokenizer, row["action"], row["global"], None, args.max_length, args.position_mode
        )
        token_lengths.append(len(ids)); candidate_counts.append(len(row["y"]))
    print(json.dumps({
        "device": str(device), "dtype": str(dtype), "train_states": len(train_ds),
        "val_states": len(val_ds), "tokens_min": min(token_lengths),
        "tokens_median": float(np.median(token_lengths)), "tokens_max": max(token_lengths),
        "candidate_counts": sorted(set(candidate_counts)),
    }), flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr},
            {"params": heads.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    trainable_parameters = [p for p in backbone.parameters() if p.requires_grad] + list(heads.parameters())
    best = -float("inf")
    history = []
    adapter_dir = args.out_dir / "adapter"
    manifest_path = args.out_dir / "jev_qwen_math.pt"

    def save_current(metrics):
        backbone.save_pretrained(adapter_dir)
        torch.save(
            {
                "kind": "jev_qwen_decision",
                "version": 1,
                "base_model": str(args.base_model),
                "adapter": str(adapter_dir),
                "heads": {k: v.detach().cpu() for k, v in heads.state_dict().items()},
                "max_length": args.max_length,
                "layer_limit": args.layer_limit,
                "position_mode": args.position_mode,
                "numeric_dim": 48,
                "prior_critic": str(args.prior_critic) if args.prior_critic else "",
                "online_feas_weight": 0.0,
                "metrics": metrics,
                "selected_residual_scale": float(F.softplus(heads.residual_log_scale).detach().cpu()),
            },
            manifest_path,
        )

    if prior_model is not None:
        initial_metrics = evaluate(
            val_ds, tokenizer, backbone, heads, build_decision_input, forward_decisions,
            device, args.max_length, args.position_mode, prior_model, prior_feas_weight,
        )
        initial_metrics.update({"epoch": 0, "train_loss": None})
        history.append(initial_metrics)
        best = initial_metrics["ndcg8"]
        save_current(initial_metrics)
        print(json.dumps(initial_metrics), flush=True)
        print("SAVED_PRIOR_BASELINE", manifest_path, "NDCG8", best, flush=True)

    for epoch in range(1, args.epochs + 1):
        backbone.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        order_states = np.random.permutation(len(train_ds))
        losses = []
        for step, state_index in enumerate(order_states, start=1):
            row = train_ds[int(state_index)]
            permutation = np.random.permutation(len(row["y"]))
            ids, qpos, order = build_decision_input(
                tokenizer, row["action"], row["global"], permutation, args.max_length, args.position_mode
            )
            ids = ids.unsqueeze(0).to(device); qpos = qpos.to(device)
            numeric = torch.tensor(
                np.concatenate([row["action"][order], row["global"][order]], axis=1),
                device=device, dtype=torch.float32,
            )
            exact = torch.tensor(row["y"][order], device=device, dtype=torch.float32)
            drive = torch.tensor(row["drive"][order], device=device, dtype=torch.float32)
            swap = torch.tensor(row["swap"][order], device=device, dtype=torch.float32)
            sync = torch.tensor(row["sync"][order], device=device, dtype=torch.float32)
            feasible = torch.tensor(row["feasible"][order], device=device, dtype=torch.bool)

            with torch.autocast(device_type="cuda", dtype=dtype):
                prior_choice = None
                if prior_model is not None:
                    with torch.no_grad():
                        prior = prior_model({
                            "X": torch.tensor(row["X"], device=device),
                            "A": torch.tensor(row["A"], device=device),
                            "action": torch.tensor(row["action"], device=device),
                            "global": torch.tensor(row["global"], device=device),
                            "mask": torch.tensor(row["mask"], device=device),
                        })
                        prior_choice = prior["gain"]
                        if prior_feas_weight:
                            prior_choice = prior_choice + prior_feas_weight * torch.log(
                                torch.sigmoid(prior["feas"]).clamp_min(1e-6)
                            )
                pred = forward_decisions(backbone, heads, ids, qpos, numeric, prior_choice)
            target_logits = exact / max(args.choice_temp, 1e-6)
            target_logits = target_logits.masked_fill(~feasible, -1e4)
            target_prob = torch.softmax(target_logits, dim=0)
            choice_loss = -(target_prob * torch.log_softmax(pred["choice"], dim=0)).sum()
            valid = feasible.float()
            denom = valid.sum().clamp_min(1.0)

            def huber(name, target):
                value = F.smooth_l1_loss(pred[name], target / 10.0, reduction="none")
                return (value * valid).sum() / denom

            gain_loss = huber("gain", exact)
            component_loss = huber("drive", drive) + huber("swap", swap) + huber("sync", sync)
            consistency = F.smooth_l1_loss(
                pred["gain"], pred["drive"] + pred["swap"] + pred["sync"]
            )
            if feasible.any() and (~feasible).any():
                feas_loss = F.binary_cross_entropy_with_logits(pred["feasible"], feasible.float())
            else:
                feas_loss = pred["feasible"].new_zeros(())
            loss = choice_loss + 0.25 * gain_loss + 0.08 * component_loss + 0.05 * consistency + 0.2 * feas_loss
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(order_states):
                nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step % 40 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(np.mean(losses[-40:]))}), flush=True)

        metrics = evaluate(
            val_ds, tokenizer, backbone, heads, build_decision_input, forward_decisions,
            device, args.max_length, args.position_mode, prior_model, prior_feas_weight,
        )
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if epoch == args.epochs:
            best = metrics["ndcg8"]
            save_current(metrics)
            print("SAVED", manifest_path, "NDCG8", best, flush=True)

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    write_metadata(args.out_dir / "history.json", {"args": serializable_args, "history": history})
    print("TRAINING_COMPLETE", manifest_path, "BEST_NDCG8", best, flush=True)


if __name__ == "__main__":
    main()

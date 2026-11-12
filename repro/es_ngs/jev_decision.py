"""Jev-style, non-generative decision model for ES-NGS candidate ranking.

The Qwen backbone reads one compact state containing every candidate.  A shared
decision head is evaluated at one query position per candidate, so inference
returns bounded scores directly and never generates text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ACTION_INDEX = (4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 24, 25, 26, 27)
ACTION_NAMES = "br,bs,rm,rb,dt,dn,de,soc,sv,dd,ds,dwt,dwd,dso"
GLOBAL_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15)
GLOBAL_NAMES = "n,st,dor,sv,ed,smin,t,tw,dw,sr,sfin,her,bf"


def _number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        value = 0.0
    # Qwen tokenizes compact integers much more efficiently than decimal
    # strings.  Two fixed decimal places are enough for the normalized search
    # features and keep a 64-option state inside its 4k context window.
    return str(int(np.clip(np.rint(value * 100.0), -9999, 9999)))


def build_decision_input(
    tokenizer,
    action: np.ndarray,
    global_features: np.ndarray,
    permutation: Sequence[int] | None = None,
    max_length: int = 4096,
    position_mode: str = "query",
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Serialize a complete candidate set and return ids/query positions/order."""
    action = np.asarray(action, dtype=np.float32)
    global_features = np.asarray(global_features, dtype=np.float32)
    count = int(action.shape[0])
    order = np.asarray(permutation if permutation is not None else np.arange(count), dtype=np.int64)
    if sorted(order.tolist()) != list(range(count)):
        raise ValueError("permutation must cover every candidate exactly once")

    g = global_features[0] if global_features.ndim == 2 else global_features
    state_values = ",".join(_number(g[i]) for i in GLOBAL_INDEX)
    header_lines = [
        "Battery-electric truck-drone search decision.",
        "Choose actions that most reduce total delivery time after exact repair.",
        "Higher decision value is better. Infeasible actions should be ranked last.",
        "All numeric feature values are integers scaled by 100.",
        f"State[{GLOBAL_NAMES}]={state_values}",
        f"Action fields[{ACTION_NAMES}]",
    ]
    ids = tokenizer.encode("\n".join(header_lines), add_special_tokens=True)
    candidate_positions: list[int] = []
    for slot, original in enumerate(order, start=1):
        values = ",".join(_number(action[original, i]) for i in ACTION_INDEX)
        segment = tokenizer.encode(f"\n{slot}:{values};", add_special_tokens=False)
        if not segment:
            raise ValueError("empty candidate tokenization")
        ids.extend(segment)
        candidate_positions.append(len(ids) - 1)
    if position_mode == "query":
        query_positions: list[int] = []
        suffix = tokenizer.encode("\nScore every numbered option.", add_special_tokens=False)
        ids.extend(suffix)
        for slot in range(1, count + 1):
            # Keep the query suffix short: 64-option states must stay within
            # the 4,096-token context of Qwen2.5-1.5B.
            segment = tokenizer.encode(f"\nQ{slot}:", add_special_tokens=False)
            if not segment:
                raise ValueError("empty decision-query tokenization")
            ids.extend(segment)
            query_positions.append(len(ids) - 1)
        decision_positions = query_positions
    elif position_mode == "candidate":
        # Each candidate's terminal marker directly represents that action.
        # The shared head then produces all bounded decisions in one pass.
        decision_positions = candidate_positions
    else:
        raise ValueError(f"unknown position_mode={position_mode}")
    if len(ids) > max_length:
        raise ValueError(f"decision input has {len(ids)} tokens, exceeds {max_length}")
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(decision_positions, dtype=torch.long),
        order,
    )


class JevDecisionHeads(nn.Module):
    """Typed outputs analogous to choice, score, and noul decisions."""

    def __init__(self, hidden_size: int, numeric_dim: int = 0):
        super().__init__()
        self.choice = nn.Linear(hidden_size, 1)
        self.gain = nn.Linear(hidden_size, 1)
        self.drive = nn.Linear(hidden_size, 1)
        self.swap = nn.Linear(hidden_size, 1)
        self.sync = nn.Linear(hidden_size, 1)
        self.feasible = nn.Linear(hidden_size, 1)
        self.residual_log_scale = nn.Parameter(torch.tensor(-2.25))
        self.numeric_dim = int(numeric_dim)
        self.numeric = (
            nn.Sequential(
                nn.LayerNorm(self.numeric_dim),
                nn.Linear(self.numeric_dim, 128),
                nn.GELU(),
                nn.Linear(128, 6),
            )
            if self.numeric_dim
            else None
        )

    def forward(
        self,
        hidden: torch.Tensor,
        numeric_features: torch.Tensor | None = None,
        prior_choice: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = hidden.float()
        result = {
            "choice": self.choice(hidden).squeeze(-1),
            "gain": self.gain(hidden).squeeze(-1),
            "drive": self.drive(hidden).squeeze(-1),
            "swap": self.swap(hidden).squeeze(-1),
            "sync": self.sync(hidden).squeeze(-1),
            "feasible": self.feasible(hidden).squeeze(-1),
        }
        if self.numeric is not None:
            if numeric_features is None:
                raise ValueError("numeric features required by this decision head")
            residual = self.numeric(numeric_features.float())
            for column, name in enumerate(("choice", "gain", "drive", "swap", "sync", "feasible")):
                result[name] = result[name] + residual[:, column]
        if prior_choice is not None:
            residual_scale = F.softplus(self.residual_log_scale)
            result["choice"] = prior_choice.float() + residual_scale * result["choice"]
        return result


def forward_decisions(
    backbone,
    heads,
    input_ids,
    query_positions,
    numeric_features: torch.Tensor | None = None,
    prior_choice: torch.Tensor | None = None,
):
    attention_mask = torch.ones_like(input_ids)
    output = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[0, query_positions]
    return heads(hidden, numeric_features, prior_choice)


def load_jev_decision(manifest_path: str | Path, device: torch.device):
    """Load a trained adapter and decision heads from a small .pt manifest."""
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    manifest_path = Path(manifest_path)
    blob = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if blob.get("kind") != "jev_qwen_decision":
        raise ValueError(f"not a Jev decision manifest: {manifest_path}")
    # Relative artifact paths are resolved against the manifest directory.
    # A local base-model directory may be supplied without editing the manifest.
    import os
    def artifact_path(value):
        path = Path(value)
        return str(path if path.is_absolute() else (manifest_path.parent / path).resolve())
    base_path = os.environ.get("ESNGS_QWEN_BASE", "") or artifact_path(blob["base_model"])
    adapter_path = artifact_path(blob["adapter"])
    if blob.get("prior_critic"):
        blob["prior_critic"] = artifact_path(blob["prior_critic"])
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base = AutoModel.from_pretrained(
        base_path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    layer_limit = int(blob.get("layer_limit", 0) or 0)
    if layer_limit and hasattr(base, "layers"):
        base.layers = nn.ModuleList(list(base.layers[:layer_limit]))
        base.config.num_hidden_layers = layer_limit
    backbone = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to(device).eval()
    heads = JevDecisionHeads(
        int(base.config.hidden_size), int(blob.get("numeric_dim", 0) or 0)
    ).to(device)
    heads.load_state_dict(blob["heads"], strict=False)
    heads.eval()
    return tokenizer, backbone, heads, blob


def write_metadata(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

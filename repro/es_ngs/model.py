"""ES-NGS critic (~0.2M): Transformer over local tokens + action/global heads."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESNGSCritic(nn.Module):
    def __init__(
        self,
        node_dim: int = 24,
        act_dim: int = 32,
        glob_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        nlayers: int = 3,
        n_rel: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.node_proj = nn.Linear(node_dim, d_model)
        self.rel_emb = nn.Embedding(n_rel, d_model)
        self.act_proj = nn.Sequential(nn.Linear(act_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.glob_proj = nn.Sequential(nn.Linear(glob_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        hid = d_model * 2
        self.head_gain = nn.Sequential(nn.Linear(hid, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_drive = nn.Sequential(nn.Linear(hid, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_swap = nn.Sequential(nn.Linear(hid, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_sync = nn.Sequential(nn.Linear(hid, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_feas = nn.Sequential(nn.Linear(hid, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.normal_(self.cls, std=0.02)

    def encode(self, X, A, action, glob, mask) -> torch.Tensor:
        # X [B,M,F], A [B,T,T], action [B,A], glob [B,G], mask [B,T] (1=valid)
        B, M, _ = X.shape
        tok = self.node_proj(X)  # [B,M,D]
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, tok], dim=1)  # [B,T,D]
        # Relation ids → mean-pooled embedding added as structural bias (no custom attn mask).
        rel = self.rel_emb(A.clamp(0, self.rel_emb.num_embeddings - 1).long()).mean(dim=2)  # [B,T,D]
        h = h + rel
        key_pad = mask < 0.5  # True = ignore
        h = self.encoder(h, src_key_padding_mask=key_pad)
        cls_h = h[:, 0, :]
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        ctx = torch.cat([cls_h + pooled, self.act_proj(action) + self.glob_proj(glob)], dim=-1)
        return ctx

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self.encode(batch["X"], batch["A"], batch["action"], batch["global"], batch["mask"])
        return {
            "gain": self.head_gain(ctx).squeeze(-1),
            "drive": self.head_drive(ctx).squeeze(-1),
            "swap": self.head_swap(ctx).squeeze(-1),
            "sync": self.head_sync(ctx).squeeze(-1),
            "feas": self.head_feas(ctx).squeeze(-1),
        }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def listwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    s = scores.clone()
    t = targets.clone()
    s = s.masked_fill(~valid, -1e9)
    t = t.masked_fill(~valid, -1e9)
    if valid.sum() < 2:
        return scores.new_zeros(())
    log_p = F.log_softmax(s, dim=0)
    q = F.softmax(t, dim=0)
    return -(q * log_p).sum()


def listwise_probability_loss(
    scores: torch.Tensor, target_probs: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy against already-normalized soft preference targets."""
    s = scores.masked_fill(~valid, -1e9)
    q = torch.clamp(target_probs, min=0.0).masked_fill(~valid, 0.0)
    mass = q.sum()
    if valid.sum() < 2 or mass <= 1e-12:
        return scores.new_zeros(())
    q = q / mass
    return -(q * F.log_softmax(s, dim=0)).sum()


def three_way_loss(pred, y, yd, ys, ysyn, valid) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = nn.MSELoss(reduction="none")
    v = valid.float()
    denom = v.sum().clamp_min(1.0)
    l_g = (mse(pred["gain"], y) * v).sum() / denom
    l_d = (mse(pred["drive"], yd) * v).sum() / denom
    l_s = (mse(pred["swap"], ys) * v).sum() / denom
    l_y = (mse(pred["sync"], ysyn) * v).sum() / denom
    cons = (mse(pred["gain"], pred["drive"] + pred["swap"] + pred["sync"]) * v).sum() / denom
    total = l_g + 0.5 * (l_d + l_s + l_y) + 0.1 * cons
    return total, {"mse_gain": float(l_g), "mse_drive": float(l_d), "mse_swap": float(l_s), "mse_sync": float(l_y), "cons": float(cons)}


if __name__ == "__main__":
    m = ESNGSCritic()
    print("params", count_params(m))
    B, M, T = 4, 32, 33
    batch = {
        "X": torch.randn(B, M, 24),
        "A": torch.randint(0, 8, (B, T, T)),
        "action": torch.randn(B, 32),
        "global": torch.randn(B, 16),
        "mask": torch.ones(B, T),
    }
    out = m(batch)
    print({k: v.shape for k, v in out.items()})

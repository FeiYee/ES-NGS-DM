"""Leakage-free, interpretable prompts shared by Qwen SFT and teacher export."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


ACTION_FIELDS = (
    ("br", 4), ("bs", 5), ("rm", 6), ("rb", 7),
    ("dt", 8), ("dn", 9), ("de", 16), ("soc", 17),
    ("sv", 18), ("dd", 19), ("ds", 24), ("dwt", 25),
    ("dwd", 26), ("dso", 27),
)

GLOBAL_FIELDS = (
    ("n", 0), ("st", 1), ("dor", 2), ("sv", 3),
    ("ed", 4), ("smin", 5), ("t", 6), ("tw", 7),
    ("dw", 8), ("sr", 9), ("sfin", 10), ("her", 11), ("bf", 15),
)


def _finite(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


def summarize_action(action: np.ndarray) -> Dict[str, float]:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if len(values) <= max(index for _, index in ACTION_FIELDS):
        raise ValueError(f"action vector has only {len(values)} channels")
    return {name: _finite(values[index]) for name, index in ACTION_FIELDS}


def summarize_global(global_features: np.ndarray) -> Dict[str, float]:
    values = np.asarray(global_features, dtype=np.float64).reshape(-1)
    if len(values) <= max(index for _, index in GLOBAL_FIELDS):
        raise ValueError(f"global vector has only {len(values)} channels")
    return {name: _finite(values[index]) for name, index in GLOBAL_FIELDS}


def _format_fields(fields: Dict[str, float]) -> str:
    return " ".join(f"{name}={value:.4f}" for name, value in fields.items())


def build_decision_prompt(meta: Dict[str, Any], candidates: Iterable[Dict[str, Any]]) -> str:
    cands: List[Dict[str, Any]] = list(candidates)
    lines = [
        "You rank pre-repair search actions for battery-electric truck-drone cooperative delivery.",
        "The operational objective is lower total delivery time.",
        "All supplied values are available before exact repair; no exact feasibility or repaired gain is shown.",
        "Positive dt/de/ds/dd/dwt/dwd/dso values mean the raw candidate reduces the corresponding quantity relative to the incumbent.",
        "State fields: n customers; st stations; dor drone-only ratio; sv station visits; ed shared-energy demand; smin/final SOC; t delivery time; tw/dw waits; sr sorties; her high-energy sorties; bf budget fraction.",
        "Candidate fields: br/bs block size/start; rm/rb removed/rebuilt sorties; dt/de/ds time/energy/synchronization reductions; dn drone-node change; soc incumbent minimum SOC; sv station visits; dd drone-energy reduction; dwt/dwd wait reductions; dso sortie reduction.",
        'Return JSON only: {"choice":[candidate ids best-first], "score":[relative decision utilities in input order], "probability":[positive-gain probabilities in input order]}.',
        (
            f"State: instance={meta['instance_id']} n={int(meta['n'])} "
            f"family={meta['family']} phase={meta['phase']} "
            f"before_T_h={float(meta['before_T_h']):.6f}"
        ),
        "State features: " + _format_fields(meta["global"]),
        f"Candidates ({len(cands)}):",
    ]
    for candidate in cands:
        lines.append(f"- id={int(candidate['cid'])} " + _format_fields(candidate["action"]))
    lines.append("Respond with JSON only.")
    prompt = "\n".join(lines)
    banned = ("feas=", "feasible=", "after_T", "y_pct=", "exact_gain=")
    if any(token in prompt for token in banned):
        raise ValueError("decision prompt contains a forbidden exact-evaluation field")
    return prompt

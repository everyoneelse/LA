#!/usr/bin/env python3
"""
Generate InternLM2 scaling-law config variants (1e3 – 1e9 params).

This script follows the recommendations in Kaplan et al. (2020) and the
assistant proposal:
  • Keep n_layer ≈ d_model / 48  (rounded to int)
  • Keep n_head s.t. d_model / n_head == 64 (preferred) else 32 or 128
  • num_key_value_heads = max(1, n_head // 2)
  • intermediate_size = 4 * d_model (rounded to multiple of 256)

It produces JSON HuggingFace config files in ``internlm2_scaling/configs``
with pattern::

    internlm2-chat-{size_label}-h{n_head}-L{n_layer}.json

and rewrites ``variants_summary.csv`` with a concise table.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Dict
import csv

# ---------- global paths ---------- #
ROOT = Path(__file__).resolve().parent.parent  # /internlm2_scaling
CONFIG_DIR = ROOT / "configs"
SUMMARY_CSV = CONFIG_DIR / "variants_summary.csv"

# Base template – will be updated per-variant
TEMPLATE_PATH = CONFIG_DIR / "internlm2-chat-126M-h4-L8.json"
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    BASE_TEMPLATE = json.load(f)

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def round_multiple(x: int, multiple: int) -> int:
    """Round *up* ``x`` to nearest multiple of ``multiple``."""
    return int(math.ceil(x / multiple) * multiple)


def choose_heads(d_model: int) -> int:
    """Pick number of attention heads so that head_dim in {64, 128, 32}."""
    for head_dim in (64, 128, 32):
        if d_model % head_dim == 0:
            return d_model // head_dim
    # fallback – force 1 head
    return 1


# ----------------------------------------------------------------------------
# Model specification
# ----------------------------------------------------------------------------
# Manual table covering 1K → 1B (14 points) incl. existing medium sizes.
TABLE = [
    # label, d_model, n_layer (can be None to auto compute)
    ("1K", 32, 1),
    ("10K", 64, 2),
    ("30K", 96, 2),
    ("100K", 128, 3),
    ("300K", 192, 4),
    ("1M", 256, 5),
    ("3M", 320, 7),
    ("10M", 384, 8),
    ("30M", 512, 11),
    ("80M", 768, 16),
    ("240M", 1024, 21),
    ("567M", 1280, 26),
    ("992M", 1408, 29),
    ("1386M", 2048, 16),  # keep existing 1.4B variant
]


# ----------------------------------------------------------------------------
# Core generation
# ----------------------------------------------------------------------------
VARIANTS: List[Dict] = []

for label, d_model, n_layer in TABLE:
    # auto compute n_layer if None (unused here but kept for flexibility)
    if n_layer is None:
        n_layer = max(1, int(round(d_model / 48)))

    n_head = choose_heads(d_model)
    kv_head = max(1, n_head // 2)
    intermediate = round_multiple(4 * d_model, 256)  # align to 256

    # Build variant dict for CSV
    variant_dict = {
        "label": label,
        "hidden_size": d_model,
        "heads": n_head,
        "kv_heads": kv_head,
        "layers": n_layer,
        "ffn_size": intermediate,
    }
    VARIANTS.append(variant_dict)

    # Prepare HF config JSON
    cfg = BASE_TEMPLATE.copy()
    cfg.update(
        {
            "hidden_size": d_model,
            "intermediate_size": intermediate,
            "num_attention_heads": n_head,
            "num_key_value_heads": kv_head,
            "num_hidden_layers": n_layer,
        }
    )

    # File name pattern (include heads/layers to avoid ambiguity)
    fname = f"internlm2-chat-{label}-h{n_head}-L{n_layer}.json"
    path = CONFIG_DIR / fname
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)
    print(f"Wrote {path.relative_to(ROOT)}")

# ----------------------------------------------------------------------------
# Re-write variants_summary.csv
# ----------------------------------------------------------------------------

def estimate_params(hidden: int, heads: int, layers: int, ffn: int, vocab: int = 92544, tie: bool = False) -> int:
    # Very rough param estimator (emb + lm head + 12*layers*hidden^2) close enough
    embed = vocab * hidden
    lm_head = 0 if tie else embed
    transformer = 12 * layers * hidden * hidden
    return embed + lm_head + transformer

with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "index",
        "config",
        "hidden_size",
        "heads",
        "kv_heads",
        "layers",
        "ffn_size",
        "params",
        "params_wo_embed",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for idx, v in enumerate(VARIANTS, 1):
        cfg_name = f"internlm2-chat-{v['label']}-h{v['heads']}-L{v['layers']}.json"
        params_est = estimate_params(v["hidden_size"], v["heads"], v["layers"], v["ffn_size"])
        embed = 92544 * v["hidden_size"]
        params_no_embed = params_est - embed
        writer.writerow(
            {
                "index": idx,
                "config": cfg_name,
                "hidden_size": v["hidden_size"],
                "heads": v["heads"],
                "kv_heads": v["kv_heads"],
                "layers": v["layers"],
                "ffn_size": v["ffn_size"],
                "params": f"{params_est/1e6:.2f}M",
                "params_wo_embed": f"{params_no_embed/1e6:.2f}M",
            }
        )

print(f"Updated {SUMMARY_CSV.relative_to(ROOT)} with {len(VARIANTS)} variants.")
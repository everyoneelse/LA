#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv

# Base config provided by user (internlm2-chat-1_8b)
BASE_CONFIG = {
    "architectures": ["InternLM2ForCausalLM"],
    "attn_implementation": "eager",
    "auto_map": {
        "AutoConfig": "configuration_internlm2.InternLM2Config",
        "AutoModelForCausalLM": "modeling_internlm2.InternLM2ForCausalLM",
        "AutoModel": "modeling_internlm2.InternLM2ForCausalLM",
    },
    "bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 32768,
    "model_type": "internlm2",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_key_value_heads": 8,
    "pad_token_id": 2,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {"type": "dynamic", "factor": 2.0},
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.0",
    "use_cache": True,
    "vocab_size": 92544,
    "pretraining_tp": 1,
}

# Design choices:
# - Keep head_dim = 128 (as in base: 2048/16)
# - Maintain GQA ratio kv_heads = max(1, heads // 2)
# - Keep MLP ratio ~4x, rounded to multiple of 256 for efficiency
# - Keep max_position_embeddings, rope config, dtype
# - Keep tie_word_embeddings as in base (False). Note this imposes a lower bound
#   on total params due to two large embedding matrices.

HEAD_DIM = 128

# Allowed head counts (even only to keep kv=heads/2 integral)
ALLOWED_NUM_HEADS = [4, 6, 8, 12, 16]
# Allowed layers to search over
ALLOWED_LAYERS = [8, 12, 16, 20, 24]

# Targets (billions of params). Must be >= embedding floor.
TARGETS_B = [0.10, 0.20, 0.35, 0.60, 1.00, 1.50]


def round_up_multiple(x: int, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)


@dataclass
class Variant:
    name: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    intermediate_size: int
    param_count: int  # total params
    tokens_recommended: int  # Chinchilla T ~ 20 * N


def estimate_params(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_layers: int,
    intermediate_size: int,
    vocab_size: int,
    tie_word_embeddings: bool,
) -> int:
    # Embedding and LM head
    embed_params = vocab_size * hidden_size
    if tie_word_embeddings:
        lm_head_params = 0
    else:
        lm_head_params = hidden_size * vocab_size

    # Attention per layer
    kv_hidden = num_kv_heads * HEAD_DIM
    attn_params_per_layer = (
        hidden_size * hidden_size  # Wq
        + hidden_size * kv_hidden  # Wk
        + hidden_size * kv_hidden  # Wv
        + hidden_size * hidden_size  # Wo
    )

    # MLP per layer (SwiGLU-ish: up, gate, down)
    mlp_params_per_layer = 3 * hidden_size * intermediate_size

    # Norms per layer (RMSNorm) - negligible but include
    norms_per_layer = 2 * hidden_size

    layer_params = attn_params_per_layer + mlp_params_per_layer + norms_per_layer

    total = embed_params + lm_head_params + num_layers * layer_params
    return int(total)


def pick_variants(base: Dict) -> List[Variant]:
    vocab_size = int(base["vocab_size"])
    tie_word_embeddings = bool(base.get("tie_word_embeddings", True))

    variants: List[Variant] = []

    # Precompute all feasible shapes within our grid
    candidate_shapes: List[Tuple[int, int, int, int]] = []  # (hidden, heads, kv_heads, layers)
    for heads in ALLOWED_NUM_HEADS:
        hidden = heads * HEAD_DIM
        kv_heads = max(1, heads // 2)
        for layers in ALLOWED_LAYERS:
            candidate_shapes.append((hidden, heads, kv_heads, layers))

    # For each target param count, pick best matching shape by choosing intermediate_size ~ 4x
    for target_b in TARGETS_B:
        target_params = int(target_b * 1e9)
        best: Tuple[float, Variant] = (float("inf"), None)  # (abs diff, variant)
        for hidden, heads, kv_heads, layers in candidate_shapes:
            intermediate = round_up_multiple(4 * hidden, 256)
            params = estimate_params(
                hidden_size=hidden,
                num_heads=heads,
                num_kv_heads=kv_heads,
                num_layers=layers,
                intermediate_size=intermediate,
                vocab_size=vocab_size,
                tie_word_embeddings=tie_word_embeddings,
            )
            diff = abs(params - target_params)
            variant = Variant(
                name=f"hs{hidden}_h{heads}_kv{kv_heads}_L{layers}",
                hidden_size=hidden,
                num_heads=heads,
                num_kv_heads=kv_heads,
                num_layers=layers,
                intermediate_size=intermediate,
                param_count=params,
                tokens_recommended=int(20 * params),
            )
            # Ensure variant is strictly smaller than the 1.8B base (safety)
            if params >= int(1.8e9):
                continue
            # Ensure we don't pick ones below the embedding floor more than 15% off the target
            # (helps avoid selecting trivially too-small configs for large targets)
            rel_err = diff / max(target_params, 1)
            # Keep the best by absolute diff primarily
            if diff < best[0]:
                best = (diff, variant)
        if best[1] is not None:
            variants.append(best[1])

    # Deduplicate by (hidden, layers) to avoid near-duplicates; keep increasing sizes
    unique: Dict[Tuple[int, int], Variant] = {}
    for v in sorted(variants, key=lambda x: x.param_count):
        key = (v.hidden_size, v.num_layers)
        if key not in unique:
            unique[key] = v
    variants = list(unique.values())
    variants.sort(key=lambda x: x.param_count)

    return variants


def make_config(base: Dict, variant: Variant) -> Dict:
    cfg = dict(base)
    cfg["hidden_size"] = variant.hidden_size
    cfg["num_attention_heads"] = variant.num_heads
    cfg["num_key_value_heads"] = variant.num_kv_heads
    cfg["num_hidden_layers"] = variant.num_layers
    cfg["intermediate_size"] = variant.intermediate_size
    # Keep other fields the same
    return cfg


def human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.2f}K"
    return str(n)


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    os.makedirs(out_dir, exist_ok=True)

    variants = pick_variants(BASE_CONFIG)

    rows: List[List[str]] = []
    for i, v in enumerate(variants, 1):
        cfg = make_config(BASE_CONFIG, v)
        # File name: internlm2-chat-{approx_params}params-h{heads}-L{layers}.json
        approx_params_m = int(round(v.param_count / 1e6))
        fname = f"internlm2-chat-{approx_params_m}M-h{v.num_heads}-L{v.num_layers}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        rows.append([
            str(i),
            fname,
            str(v.hidden_size),
            f"{v.num_heads}/{v.num_kv_heads}",
            str(v.num_layers),
            str(v.intermediate_size),
            human_count(v.param_count),
            human_count(v.tokens_recommended),
        ])

    # Write CSV summary
    csv_path = os.path.join(out_dir, "variants_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "index",
            "config",
            "hidden_size",
            "heads",
            "kv_heads",
            "layers",
            "ffn_size",
            "params",
            "params_raw",
            "tokens_recommended",
        ])
        for i, v in enumerate(variants, 1):
            approx_params_m = int(round(v.param_count / 1e6))
            fname = f"internlm2-chat-{approx_params_m}M-h{v.num_heads}-L{v.num_layers}.json"
            writer.writerow([
                i,
                fname,
                v.hidden_size,
                v.num_heads,
                v.num_kv_heads,
                v.num_layers,
                v.intermediate_size,
                human_count(v.param_count),
                v.param_count,
                v.tokens_recommended,
            ])

    # Print summary table
    headers = [
        "#",
        "config",
        "hidden_size",
        "heads(kv)",
        "layers",
        "ffn_size",
        "params",
        "tokens(~20xN)",
    ]

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))))

    print(f"\nSaved {len(rows)} configs to: {out_dir}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
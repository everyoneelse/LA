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
ALLOWED_NUM_HEADS = [2, 4, 6, 8, 12, 16, 20, 24, 32]
# Allowed layers to search over
ALLOWED_LAYERS = [6, 8, 12, 16, 20, 24, 32, 36]

# Targets following OpenAI scaling law (from 10^3 to 10^9 params)
# Based on GPT-3 style configurations and scaling law paper
TARGETS_B = [
    0.001,   # 1M - 10^6
    0.003,   # 3M - 10^6.5  
    0.010,   # 10M - 10^7
    0.030,   # 30M - 10^7.5
    0.100,   # 100M - 10^8
    0.125,   # 125M (GPT-3 small)
    0.350,   # 350M (GPT-3 medium)
    0.760,   # 760M (GPT-3 large)  
    1.300,   # 1.3B (GPT-3 XL)
    2.700,   # 2.7B
    6.700,   # 6.7B
]


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


def get_openai_style_config(target_params: int) -> Tuple[int, int, int]:
    """
    Get OpenAI/GPT-3 style configuration for given parameter count.
    Returns (d_model, n_layers, n_heads) following scaling law principles.
    
    Based on GPT-3 configurations:
    - 125M: d_model=768, n_layers=12
    - 350M: d_model=1024, n_layers=24  
    - 760M: d_model=1536, n_layers=24
    - 1.3B: d_model=2048, n_layers=24
    - 2.7B: d_model=2560, n_layers=32
    - 6.7B: d_model=4096, n_layers=32
    """
    if target_params <= 3e6:  # 3M
        return 256, 6, 2
    elif target_params <= 10e6:  # 10M
        return 384, 8, 4
    elif target_params <= 30e6:  # 30M
        return 512, 12, 4
    elif target_params <= 125e6:  # 125M
        return 768, 12, 6
    elif target_params <= 350e6:  # 350M
        return 1024, 24, 8
    elif target_params <= 760e6:  # 760M
        return 1536, 24, 12
    elif target_params <= 1.3e9:  # 1.3B
        return 2048, 24, 16
    elif target_params <= 2.7e9:  # 2.7B
        return 2560, 32, 20
    else:  # 6.7B+
        return 4096, 32, 32


def pick_variants(base: Dict) -> List[Variant]:
    vocab_size = int(base["vocab_size"])
    tie_word_embeddings = bool(base.get("tie_word_embeddings", True))

    variants: List[Variant] = []

    # For each target param count, use OpenAI-style configuration as starting point
    for target_b in TARGETS_B:
        target_params = int(target_b * 1e9)
        
        # Get OpenAI-style base configuration
        base_d_model, base_n_layers, base_n_heads = get_openai_style_config(target_params)
        
        # Ensure heads is compatible with HEAD_DIM=128 and kv_heads
        if base_d_model % HEAD_DIM != 0:
            base_n_heads = base_d_model // HEAD_DIM
        else:
            base_n_heads = base_d_model // HEAD_DIM
            
        # Ensure n_heads is in allowed list
        if base_n_heads not in ALLOWED_NUM_HEADS:
            # Find closest allowed heads
            base_n_heads = min(ALLOWED_NUM_HEADS, key=lambda x: abs(x - base_n_heads))
            base_d_model = base_n_heads * HEAD_DIM
            
        # Ensure n_layers is in allowed list
        if base_n_layers not in ALLOWED_LAYERS:
            base_n_layers = min(ALLOWED_LAYERS, key=lambda x: abs(x - base_n_layers))
        
        kv_heads = max(1, base_n_heads // 2)
        intermediate = round_up_multiple(4 * base_d_model, 256)
        
        # Calculate actual params with this configuration
        params = estimate_params(
            hidden_size=base_d_model,
            num_heads=base_n_heads,
            num_kv_heads=kv_heads,
            num_layers=base_n_layers,
            intermediate_size=intermediate,
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
        )
        
        # If too far from target, try to adjust
        if abs(params - target_params) / target_params > 0.3:
            # Try different layer counts to get closer
            best_diff = abs(params - target_params)
            best_config = (base_d_model, base_n_heads, kv_heads, base_n_layers, intermediate)
            
            for adj_layers in ALLOWED_LAYERS:
                adj_params = estimate_params(
                    hidden_size=base_d_model,
                    num_heads=base_n_heads,
                    num_kv_heads=kv_heads,
                    num_layers=adj_layers,
                    intermediate_size=intermediate,
                    vocab_size=vocab_size,
                    tie_word_embeddings=tie_word_embeddings,
                )
                diff = abs(adj_params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (base_d_model, base_n_heads, kv_heads, adj_layers, intermediate)
                    params = adj_params
            
            base_d_model, base_n_heads, kv_heads, base_n_layers, intermediate = best_config

        variant = Variant(
            name=f"hs{base_d_model}_h{base_n_heads}_kv{kv_heads}_L{base_n_layers}",
            hidden_size=base_d_model,
            num_heads=base_n_heads,
            num_kv_heads=kv_heads,
            num_layers=base_n_layers,
            intermediate_size=intermediate,
            param_count=params,
            tokens_recommended=int(20 * params),
        )
        
        # Skip if too large
        if params >= int(7e9):
            continue
            
        variants.append(variant)

    # Remove duplicates and sort
    unique: Dict[Tuple[int, int], Variant] = {}
    for v in sorted(variants, key=lambda x: x.param_count):
        key = (v.hidden_size, v.num_layers)
        if key not in unique or unique[key].param_count > v.param_count:
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
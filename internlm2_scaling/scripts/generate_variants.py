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
ALLOWED_NUM_HEADS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32]
# Allowed layers to search over  
ALLOWED_LAYERS = [2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 36]

# Targets following OpenAI scaling law (from 10^3 to 10^9 params)
# Extended to include more micro models for comprehensive scaling law study
TARGETS_B = [
    # Micro models (10^3 to 10^6)
    0.0001,   # 100K - 10^5
    0.0003,   # 300K - 10^5.5
    0.0005,   # 500K - 10^5.7
    0.001,    # 1M - 10^6
    0.002,    # 2M - 10^6.3
    0.003,    # 3M - 10^6.5
    0.005,    # 5M - 10^6.7
    0.007,    # 7M - 10^6.85
    # Small models (10^6 to 10^8)
    0.010,    # 10M - 10^7
    0.020,    # 20M - 10^7.3
    0.030,    # 30M - 10^7.5
    0.050,    # 50M - 10^7.7
    0.070,    # 70M - 10^7.85
    0.100,    # 100M - 10^8
    0.125,    # 125M (GPT-3 small)
    # Medium models (10^8 to 10^9)
    0.200,    # 200M
    0.350,    # 350M (GPT-3 medium)
    0.500,    # 500M
    0.760,    # 760M (GPT-3 large)  
    1.000,    # 1B
    1.300,    # 1.3B (GPT-3 XL)
    # Large models (10^9+)
    2.000,    # 2B
    2.700,    # 2.7B
    4.000,    # 4B
    6.700,    # 6.7B
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
    non_embed_params: int  # non-embedding params (for scaling law)
    tokens_recommended: int  # Chinchilla T ~ 20 * N


def estimate_params(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_layers: int,
    intermediate_size: int,
    vocab_size: int,
    tie_word_embeddings: bool,
    head_dim: int = HEAD_DIM,
    exclude_embeddings: bool = False,
) -> int:
    # Embedding and LM head
    embed_params = vocab_size * hidden_size
    if tie_word_embeddings:
        lm_head_params = 0
    else:
        lm_head_params = hidden_size * vocab_size

    # Attention per layer
    kv_hidden = num_kv_heads * head_dim
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
    
    # Non-embedding parameters (for scaling law)
    non_embed_params = num_layers * layer_params
    
    if exclude_embeddings:
        return int(non_embed_params)
    else:
        total = embed_params + lm_head_params + non_embed_params
        return int(total)


def estimate_non_embedding_params(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_layers: int,
    intermediate_size: int,
    head_dim: int = HEAD_DIM,
) -> int:
    """Calculate only non-embedding parameters for scaling law analysis."""
    return estimate_params(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        vocab_size=0,  # Not used when exclude_embeddings=True
        tie_word_embeddings=True,  # Not used when exclude_embeddings=True
        head_dim=head_dim,
        exclude_embeddings=True,
    )


def get_openai_style_config(target_params: int) -> Tuple[int, int, int]:
    """
    Get OpenAI/GPT-3 style configuration for given parameter count.
    Returns (d_model, n_layers, n_heads) following scaling law principles.
    
    Extended to cover micro models (10^3 to 10^6) and larger models.
    Based on GPT-3 configurations and scaling law principles.
    """
    if target_params <= 100e3:  # 100K
        return 128, 2, 1
    elif target_params <= 300e3:  # 300K
        return 128, 3, 1
    elif target_params <= 500e3:  # 500K
        return 128, 4, 1
    elif target_params <= 1e6:  # 1M
        return 192, 3, 1  # Slightly larger d_model
    elif target_params <= 2e6:  # 2M
        return 192, 4, 1
    elif target_params <= 3e6:  # 3M
        return 256, 3, 2
    elif target_params <= 5e6:  # 5M
        return 256, 4, 2
    elif target_params <= 7e6:  # 7M
        return 256, 6, 2
    elif target_params <= 10e6:  # 10M
        return 384, 4, 3
    elif target_params <= 20e6:  # 20M
        return 384, 6, 3
    elif target_params <= 30e6:  # 30M
        return 512, 6, 4
    elif target_params <= 50e6:  # 50M
        return 512, 8, 4
    elif target_params <= 70e6:  # 70M
        return 640, 8, 5  # Non-standard but works
    elif target_params <= 100e6:  # 100M
        return 768, 8, 6
    elif target_params <= 125e6:  # 125M (GPT-3 small)
        return 768, 12, 6
    elif target_params <= 200e6:  # 200M
        return 896, 12, 7
    elif target_params <= 350e6:  # 350M (GPT-3 medium)
        return 1024, 16, 8
    elif target_params <= 500e6:  # 500M
        return 1280, 16, 10
    elif target_params <= 760e6:  # 760M (GPT-3 large)
        return 1536, 20, 12
    elif target_params <= 1e9:  # 1B
        return 1664, 20, 13
    elif target_params <= 1.3e9:  # 1.3B (GPT-3 XL)
        return 2048, 20, 16
    elif target_params <= 2e9:  # 2B
        return 2304, 24, 18
    elif target_params <= 2.7e9:  # 2.7B
        return 2560, 28, 20
    elif target_params <= 4e9:  # 4B
        return 3200, 28, 25
    else:  # 6.7B+
        return 4096, 32, 32


def pick_variants(base: Dict) -> List[Variant]:
    vocab_size = int(base["vocab_size"])
    tie_word_embeddings = bool(base.get("tie_word_embeddings", True))

    variants: List[Variant] = []

    # For each target param count, use OpenAI-style configuration as starting point
    # NOTE: Now targeting NON-EMBEDDING parameters for scaling law analysis
    for target_b in TARGETS_B:
        target_non_embed_params = int(target_b * 1e9)  # Target non-embedding params
        
        # Get OpenAI-style base configuration based on non-embedding param target
        base_d_model, base_n_layers, base_n_heads = get_openai_style_config(target_non_embed_params)
        
        # For vocab_size, use reasonable defaults that don't dominate parameter count
        # but are still practical for training
        if target_non_embed_params < 1e6:  # < 1M non-embed params
            micro_vocab_size = 8000
            micro_tie_embeddings = True
        elif target_non_embed_params < 10e6:  # < 10M non-embed params
            micro_vocab_size = 16000
            micro_tie_embeddings = True
        elif target_non_embed_params < 50e6:  # < 50M non-embed params
            micro_vocab_size = 32000
            micro_tie_embeddings = False
        else:
            micro_vocab_size = vocab_size
            micro_tie_embeddings = tie_word_embeddings
        
        # For very small models, we may need smaller head_dim
        if base_d_model < HEAD_DIM:
            # For micro models, use head_dim = d_model for single head
            actual_head_dim = base_d_model
            base_n_heads = 1
        else:
            # Ensure heads is compatible with HEAD_DIM=128 and kv_heads
            actual_head_dim = HEAD_DIM
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
        
        # For micro models, use smaller MLP ratios to avoid parameter explosion
        if target_non_embed_params < 1e6:  # < 1M non-embed params
            mlp_ratio = 2  # 2x instead of 4x
            intermediate = round_up_multiple(mlp_ratio * base_d_model, 64)
        elif target_non_embed_params < 10e6:  # < 10M non-embed params
            mlp_ratio = 3  # 3x instead of 4x
            intermediate = round_up_multiple(mlp_ratio * base_d_model, 128)
        else:
            intermediate = round_up_multiple(4 * base_d_model, 256)
        
        # Calculate actual params with this configuration
        actual_head_dim = base_d_model if base_d_model < HEAD_DIM else HEAD_DIM
        
        # Calculate both total and non-embedding params
        total_params = estimate_params(
            hidden_size=base_d_model,
            num_heads=base_n_heads,
            num_kv_heads=kv_heads,
            num_layers=base_n_layers,
            intermediate_size=intermediate,
            vocab_size=micro_vocab_size,
            tie_word_embeddings=micro_tie_embeddings,
            head_dim=actual_head_dim,
        )
        
        non_embed_params = estimate_non_embedding_params(
            hidden_size=base_d_model,
            num_heads=base_n_heads,
            num_kv_heads=kv_heads,
            num_layers=base_n_layers,
            intermediate_size=intermediate,
            head_dim=actual_head_dim,
        )
        
        # If too far from target NON-EMBEDDING params, try to adjust
        if abs(non_embed_params - target_non_embed_params) / target_non_embed_params > 0.3:
            # Try different layer counts to get closer
            best_diff = abs(non_embed_params - target_non_embed_params)
            best_config = (base_d_model, base_n_heads, kv_heads, base_n_layers, intermediate)
            best_total_params = total_params
            best_non_embed_params = non_embed_params
            
            for adj_layers in ALLOWED_LAYERS:
                adj_total_params = estimate_params(
                    hidden_size=base_d_model,
                    num_heads=base_n_heads,
                    num_kv_heads=kv_heads,
                    num_layers=adj_layers,
                    intermediate_size=intermediate,
                    vocab_size=micro_vocab_size,
                    tie_word_embeddings=micro_tie_embeddings,
                    head_dim=actual_head_dim,
                )
                adj_non_embed_params = estimate_non_embedding_params(
                    hidden_size=base_d_model,
                    num_heads=base_n_heads,
                    num_kv_heads=kv_heads,
                    num_layers=adj_layers,
                    intermediate_size=intermediate,
                    head_dim=actual_head_dim,
                )
                diff = abs(adj_non_embed_params - target_non_embed_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (base_d_model, base_n_heads, kv_heads, adj_layers, intermediate)
                    best_total_params = adj_total_params
                    best_non_embed_params = adj_non_embed_params
            
            base_d_model, base_n_heads, kv_heads, base_n_layers, intermediate = best_config
            total_params = best_total_params
            non_embed_params = best_non_embed_params

        # Store the vocab info for config generation
        variant = Variant(
            name=f"hs{base_d_model}_h{base_n_heads}_kv{kv_heads}_L{base_n_layers}",
            hidden_size=base_d_model,
            num_heads=base_n_heads,
            num_kv_heads=kv_heads,
            num_layers=base_n_layers,
            intermediate_size=intermediate,
            param_count=total_params,
            non_embed_params=non_embed_params,
            tokens_recommended=int(20 * non_embed_params),  # Base on non-embed params
        )
        # Add custom attributes for micro models
        variant.vocab_size = micro_vocab_size
        variant.tie_word_embeddings = micro_tie_embeddings
        
        # Skip if too large (based on non-embedding params)
        if non_embed_params >= int(7e9):
            continue
            
        variants.append(variant)

    # Remove duplicates and sort by non-embedding params
    unique: Dict[Tuple[int, int], Variant] = {}
    for v in sorted(variants, key=lambda x: x.non_embed_params):
        key = (v.hidden_size, v.num_layers)
        if key not in unique or unique[key].non_embed_params > v.non_embed_params:
            unique[key] = v
    
    variants = list(unique.values())
    variants.sort(key=lambda x: x.non_embed_params)

    return variants


def make_config(base: Dict, variant: Variant) -> Dict:
    cfg = dict(base)
    cfg["hidden_size"] = variant.hidden_size
    cfg["num_attention_heads"] = variant.num_heads
    cfg["num_key_value_heads"] = variant.num_kv_heads
    cfg["num_hidden_layers"] = variant.num_layers
    cfg["intermediate_size"] = variant.intermediate_size
    
    # Use micro model settings if available
    if hasattr(variant, 'vocab_size'):
        cfg["vocab_size"] = variant.vocab_size
    if hasattr(variant, 'tie_word_embeddings'):
        cfg["tie_word_embeddings"] = variant.tie_word_embeddings
    
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
        # File name based on non-embedding params for scaling law clarity
        approx_non_embed_params = v.non_embed_params
        if approx_non_embed_params >= 1e9:
            approx_str = f"{approx_non_embed_params/1e9:.1f}B"
            fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
        elif approx_non_embed_params >= 1e6:
            approx_str = f"{int(round(approx_non_embed_params / 1e6))}M"
            fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
        else:
            approx_str = f"{int(round(approx_non_embed_params / 1e3))}K"
            fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
            
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
            human_count(v.non_embed_params),  # Show non-embedding params
            human_count(v.param_count),       # Show total params
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
            "non_embed_params",
            "non_embed_params_raw",
            "total_params",
            "total_params_raw",
            "vocab_size",
            "tied_embeddings",
            "tokens_recommended",
        ])
        for i, v in enumerate(variants, 1):
            # Generate filename based on non-embedding params
            approx_non_embed_params = v.non_embed_params
            if approx_non_embed_params >= 1e9:
                approx_str = f"{approx_non_embed_params/1e9:.1f}B"
                fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
            elif approx_non_embed_params >= 1e6:
                approx_str = f"{int(round(approx_non_embed_params / 1e6))}M"
                fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
            else:
                approx_str = f"{int(round(approx_non_embed_params / 1e3))}K"
                fname = f"internlm2-chat-{approx_str}-h{v.num_heads}-L{v.num_layers}.json"
                
            writer.writerow([
                i,
                fname,
                v.hidden_size,
                v.num_heads,
                v.num_kv_heads,
                v.num_layers,
                v.intermediate_size,
                human_count(v.non_embed_params),
                v.non_embed_params,
                human_count(v.param_count),
                v.param_count,
                v.vocab_size,
                v.tie_word_embeddings,
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
        "non_embed_params",
        "total_params",
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
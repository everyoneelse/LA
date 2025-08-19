#!/usr/bin/env python3

import argparse
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


ITER_LINE_REGEX = re.compile(
    r"\[Iter\s+(?P<step>\d+)\].*?Tokens:\s+"
    r"(?P<batch_tokens>[\d.]+)(?P<batch_suffix>[KMG])?\s+tokens\s+\(batch\),\s+"
    r"(?P<total_tokens>[\d.]+)(?P<total_suffix>[KMG])?\s+tokens\s+\(total\)",
)

EPOCH_LINE_REGEX = re.compile(
    r"Epoch:\s*\[\d+\]\s*\[(?P<step>\d+)/[^\]]*\]\s*lr:\s*[\d.eE+-]+\s*"
    r"closs:\s*(?P<closs>[\d.eE+-]+)\s*\((?P<closs_avg>[\d.eE+-]+)\)"
)

# Validation lines do not contain 'lr: ...'
VAL_EPOCH_LINE_REGEX = re.compile(
    r"Epoch:\s*\[\d+\]\s*\[(?P<step>\d+)/[^\]]*\]\s*"
    r"closs:\s*(?P<closs>[\d.eE+-]+)\s*\((?P<closs_avg>[\d.eE+-]+)\)"
)

SUFFIX_MULTIPLIERS = {
    None: 1.0,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def _to_number(value_str: str, suffix: str | None) -> float:
    try:
        base = float(value_str)
    except ValueError as exc:
        raise ValueError(f"Unable to parse numeric value: '{value_str}'") from exc
    mult = SUFFIX_MULTIPLIERS.get(suffix, 1.0)
    return base * mult


def parse_log(filepath: str) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Parse the log file.

    Returns:
        step_to_total_tokens, step_to_closs_current, step_to_closs_avg
    """
    step_to_total_tokens: Dict[int, float] = {}
    step_to_closs_current: Dict[int, float] = {}
    step_to_closs_avg: Dict[int, float] = {}

    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            # Parse iteration line for tokens
            iter_match = ITER_LINE_REGEX.search(line)
            if iter_match:
                step = int(iter_match.group("step"))
                total_tokens = _to_number(
                    iter_match.group("total_tokens"), iter_match.group("total_suffix")
                )
                step_to_total_tokens[step] = total_tokens
                continue

            # Parse epoch line for closs values
            epoch_match = EPOCH_LINE_REGEX.search(line)
            if epoch_match:
                step = int(epoch_match.group("step"))
                closs_current = float(epoch_match.group("closs"))
                closs_avg = float(epoch_match.group("closs_avg"))
                step_to_closs_current[step] = closs_current
                step_to_closs_avg[step] = closs_avg
                continue

    return step_to_total_tokens, step_to_closs_current, step_to_closs_avg


def parse_validation_losses(filepath: str) -> List[float]:
    """Parse validation closs values after the '!!!start validation!!!' marker.

    Returns a list of current closs values in the order they appear.
    """
    in_validation: bool = False
    val_closs_values: List[float] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "!!!start validation!!!" in line:
                in_validation = True
                continue

            if not in_validation:
                continue

            # Heuristic stop: after validation finishes, many logs print 'Averaged stats:' or new sections
            if line.strip().startswith("Averaged stats:"):
                # keep going in case there are multiple validation sections; do not turn off yet
                # We'll rely on absence of VAL_EPOCH_LINE_REGEX to naturally skip other lines
                pass

            val_match = VAL_EPOCH_LINE_REGEX.search(line)
            if val_match:
                closs_current = float(val_match.group("closs"))
                val_closs_values.append(closs_current)

    return val_closs_values


def build_series(
    step_to_total_tokens: Dict[int, float],
    step_to_closs_current: Dict[int, float],
    step_to_closs_avg: Dict[int, float],
    metric: str,
) -> Tuple[List[float], List[float]]:
    common_steps = sorted(
        set(step_to_total_tokens) & set(step_to_closs_current) & set(step_to_closs_avg)
    )
    if not common_steps:
        raise RuntimeError(
            "No overlapping steps found between tokens and closs entries. "
            "Ensure your log contains both '[Iter <n>]' and 'Epoch: [... <n>/...] closs: ...' lines for the same steps."
        )

    # Build token-loss pairs to ensure proper sorting
    token_loss_pairs = []
    for step in common_steps:
        tokens = step_to_total_tokens[step]
        if metric == "avg":
            loss = step_to_closs_avg[step]
        else:
            loss = step_to_closs_current[step]
        token_loss_pairs.append((tokens, loss))

    # Sort by tokens to ensure proper ordering
    token_loss_pairs.sort(key=lambda x: x[0])
    
    xs_tokens = [pair[0] for pair in token_loss_pairs]
    ys_closs = [pair[1] for pair in token_loss_pairs]

    return xs_tokens, ys_closs


def plot_log_log(
    tokens: List[float],
    closs: List[float],
    metric_label: str,
    title: str | None = None,
    out_path: str | None = None,
    show: bool = False,
) -> None:
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(tokens, closs, marker="o", markersize=3, linewidth=1.25, label=metric_label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total tokens processed")
    plt.ylabel("closs")
    if title:
        plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved figure to: {out_path}")

    if show or not out_path:
        plt.show()


def plot_validation(
    val_losses: List[float],
    title: str | None = None,
    out_path: str | None = None,
    show: bool = False,
) -> None:
    if not val_losses:
        print("No validation closs entries found to plot.")
        return

    indices = list(range(1, len(val_losses) + 1))
    avg_loss = sum(val_losses) / float(len(val_losses))

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(indices, val_losses, marker="o", markersize=2.5, linewidth=1.0, label="val closs")
    plt.axhline(avg_loss, color="red", linestyle="--", linewidth=1.25, label=f"mean={avg_loss:.4f}")
    plt.xlabel("validation step (batch index)")
    plt.ylabel("closs")
    if title:
        plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved validation figure to: {out_path}")

    if show or not out_path:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse training logs to draw a log-log plot of total tokens vs closs, "
            "aligned by iteration step."
        )
    )
    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to the training log file",
    )
    parser.add_argument(
        "--metric",
        choices=["current", "avg"],
        default="avg",
        help=(
            "Which closs to plot: 'current' (value before parentheses) or 'avg' "
            "(value inside parentheses). Default: avg"
        ),
    )
    parser.add_argument(
        "--plot-val",
        action="store_true",
        help=(
            "If set, also parse validation closs (after '!!!start validation!!!') and plot "
            "validation loss vs batch index with a horizontal mean line."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output image file path (e.g., /workspace/tokens_vs_closs.png)",
    )
    parser.add_argument(
        "--val-out",
        default=None,
        help="Optional output image path for validation plot (e.g., /workspace/val_closs.png)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (in addition to saving if --out is provided)",
    )

    args = parser.parse_args()

    try:
        step_to_total_tokens, step_to_closs_current, step_to_closs_avg = parse_log(args.log_path)
        xs_tokens, ys_closs = build_series(
            step_to_total_tokens,
            step_to_closs_current,
            step_to_closs_avg,
            metric=args.metric,
        )
    except Exception as exc:
        print(f"Error building training series: {exc}", file=sys.stderr)
        xs_tokens, ys_closs = [], []

    if xs_tokens and ys_closs:
        plot_title = args.title or f"Tokens vs closs ({args.metric})"
        metric_label = f"closs ({args.metric})"
        plot_log_log(xs_tokens, ys_closs, metric_label, title=plot_title, out_path=args.out, show=args.show)

    if args.plot_val:
        val_losses = parse_validation_losses(args.log_path)
        if val_losses:
            avg_loss = sum(val_losses) / float(len(val_losses))
            print(f"Validation closs count: {len(val_losses)} | mean: {avg_loss:.6f}")
            val_title = (args.title + " - validation") if args.title else "Validation closs"
            plot_validation(val_losses, title=val_title, out_path=args.val_out, show=args.show)
        else:
            print("No validation losses found in the log.")


if __name__ == "__main__":
    main()
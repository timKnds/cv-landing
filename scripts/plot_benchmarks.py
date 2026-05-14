"""
Plots performance benchmark results (latency vs resolution).

Usage:
    uv run scripts/plot_benchmarks.py
"""

import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


FPS_BUDGETS = {
    "10 FPS": 1000 / 10,
    "20 FPS": 1000 / 20,
    "30 FPS": 1000 / 30,
}


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot(csv_path: str, out_path: str):
    rows = load_csv(csv_path)

    labels = [f"{r['width']}×{r['height']}" for r in rows]
    mins   = [float(r["min_ms"])  for r in rows]
    maxs   = [float(r["max_ms"])  for r in rows]
    means  = [float(r["mean_ms"]) for r in rows]
    stds   = [float(r["std_ms"])  for r in rows]
    xs     = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))

    candle_w = 0.3
    body_color  = "steelblue"
    wick_color  = "#1a3a5c"

    for i, (mn, mx, mean, std) in enumerate(zip(mins, maxs, means, stds)):
        # wick: min → max
        ax.plot([i, i], [mn, mx], color=wick_color, linewidth=1.5, zorder=2)
        # body: mean-std → mean+std
        body_lo = max(mn, mean - std)
        body_hi = min(mx, mean + std)
        ax.add_patch(plt.Rectangle(
            (i - candle_w / 2, body_lo), candle_w, body_hi - body_lo,
            color=body_color, alpha=0.85, zorder=3
        ))
        # mean tick
        ax.plot([i - candle_w / 2, i + candle_w / 2], [mean, mean],
                color="#f0a500", linewidth=1, zorder=4)
        # labels
        ax.text(i, mx * 1.05, f"{mx:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
        ax.text(i, mn * 0.95, f"{mn:.2f}", ha="center", va="top",    fontsize=7, color="#333")
        ax.text(i + candle_w / 2 + 0.03, mean, f"μ {mean:.2f}",
                ha="left", va="center", fontsize=7, color="#333", zorder=5)

    colors = ["#e74c3c", "#e67e22", "#2ecc71"]
    for (fps_label, budget_ms), color in zip(FPS_BUDGETS.items(), colors):
        ax.axhline(budget_ms, linestyle="--", linewidth=1.5, color=color,
                   label=f"{fps_label} budget ({budget_ms:.0f} ms)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Patch(facecolor=body_color, alpha=0.85, label="mean ± 1σ (body)"),
        Line2D([0], [0], color=wick_color, linewidth=1.5, label="min/max (wick)"),
        Line2D([0], [0], color="#f0a500", linewidth=1, label="mean (center line)"),
    ] + [
        Line2D([0], [0], linestyle="--", color=c, linewidth=1.5, label=f"{lbl} budget ({ms:.0f} ms)")
        for (lbl, ms), c in zip(FPS_BUDGETS.items(), colors)
    ]
    ax.legend(handles=legend_els, fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("ArUco Detection — Latency vs Resolution")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/perf_resolution.csv")
    parser.add_argument("--out", default="results/plot_perf_resolution.png")
    args = parser.parse_args()
    plot(args.csv, args.out)

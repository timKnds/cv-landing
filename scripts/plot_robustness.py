"""
Plots robustness benchmark results (detection pass/fail per scenario).

Usage:
    uv run scripts/plot_robustness.py
    uv run scripts/plot_robustness.py --scenario yaw
"""

import csv
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


SCENARIOS = ["yaw", "pitch", "brightness", "noise", "blur", "scale"]

X_LABELS = {
    "yaw":        "Marker Yaw [°]",
    "pitch":      "Marker Pitch [°]",
    "brightness": "Brightness offset [%]",
    "noise":      "Gaussian noise σ",
    "blur":       "Blur kernel size",
    "scale":      "Distance [m]",
}


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_scenario(rows: list[dict], scenario: str, out_path: str):
    params   = [r["param"] for r in rows]
    detected = [int(r["detected"]) for r in rows]

    x = range(len(params))
    colors = ["steelblue" if d else "#e74c3c" for d in detected]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, [1] * len(params), color=colors, alpha=0.85)
    ax.set_xlabel(X_LABELS.get(scenario, scenario))
    ax.set_yticks([])
    ax.set_ylim(0, 1.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_title(f"Robustness — {scenario}")
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.85, label="Detected"),
        Patch(facecolor="#e74c3c",   alpha=0.85, label="Not detected"),
    ], loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_all(results_dir: str):
    for scenario in SCENARIOS:
        csv_path = f"{results_dir}/robustness_{scenario}.csv"
        out_path = f"{results_dir}/plot_robustness_{scenario}.png"
        if not Path(csv_path).exists():
            print(f"Skip (not found): {csv_path}")
            continue
        rows = load_csv(csv_path)
        plot_scenario(rows, scenario, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="results")
    parser.add_argument("--scenario", default="all",
                        choices=SCENARIOS + ["all"])
    args = parser.parse_args()

    if args.scenario == "all":
        plot_all(args.results)
    else:
        csv_path = f"{args.results}/robustness_{args.scenario}.csv"
        out_path = f"{args.results}/plot_robustness_{args.scenario}.png"
        rows = load_csv(csv_path)
        plot_scenario(rows, args.scenario, out_path)

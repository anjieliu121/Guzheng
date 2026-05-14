#!/usr/bin/env python3
"""
Generate Figure 4: melodic / rhythmic / structural metric comparison
across training data, baseline (from-scratch transformer), and fine-tuned NotaGen.

Legend is placed below the figure so it cannot overlap any bar or error bar.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Order: (metric_label, key_in_data) per dimension
DIMENSIONS = {
    "Melodic quality": [
        ("Pitch\nautocorrelation\n(higher = coherent)", "autocorr_1"),
        ("Inverse interval\npenalty\n(higher = smoother)", "smoothness"),
    ],
    "Rhythmic quality": [
        ("CV(IOI)\n(closer to training\n= more stable)", "ioi_cv"),
    ],
    "Structural quality": [
        ("5-gram interval\nrecurrence\n(higher = more motifs)", "motif_rep_5"),
        ("LZ compression\nratio\n(lower = structured)", "compression_ratio"),
    ],
}

GROUP_ORDER = ["Training data", "Baseline", "Fine-tuned NotaGen"]
GROUP_COLORS = {
    "Training data": "#4C72B0",
    "Baseline": "#C44E52",
    "Fine-tuned NotaGen": "#55A868",
}


# Hardcoded values from Table 3 in the thesis. Override with --json to load
# from a JSON file produced by compare_musicality.py.
DEFAULT_VALUES = {
    "Training data": {
        "autocorr_1":        (0.314, 0.241),
        "smoothness":        (0.628, 0.176),
        "ioi_cv":            (0.590, 0.192),
        "motif_rep_5":       (0.369, 0.262),
        "compression_ratio": (0.607, 0.179),
    },
    "Baseline": {
        "autocorr_1":        (-0.001, 0.083),
        "smoothness":        (0.316, 0.048),
        "ioi_cv":            (1.648, 1.614),
        "motif_rep_5":       (0.001, 0.005),
        "compression_ratio": (0.853, 0.025),
    },
    "Fine-tuned NotaGen": {
        "autocorr_1":        (0.188, 0.409),
        "smoothness":        (0.608, 0.281),
        "ioi_cv":            (0.494, 0.274),
        "motif_rep_5":       (0.418, 0.276),
        "compression_ratio": (0.566, 0.173),
    },
}


def plot(values, out_path):
    """Render the three-panel bar chart."""
    n_panels = len(DIMENSIONS)
    width_ratios = [len(metrics) for metrics in DIMENSIONS.values()]

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(11, 4.5),
        gridspec_kw={"width_ratios": width_ratios},
    )

    bar_width = 0.25

    for ax, (dim_name, metrics) in zip(axes, DIMENSIONS.items()):
        n_metrics = len(metrics)
        x = np.arange(n_metrics)

        for offset_idx, group in enumerate(GROUP_ORDER):
            means = [values[group][key][0] for _, key in metrics]
            stds  = [values[group][key][1] for _, key in metrics]
            ax.bar(
                x + (offset_idx - 1) * bar_width,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=GROUP_COLORS[group],
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.8, "ecolor": "#444"},
                label=group if ax is axes[0] else None,  # only add to legend once
            )

        ax.set_title(dim_name, fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in metrics], fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Metric value", fontsize=10)

    # Reserve space at the bottom for the legend, then place a figure-level
    # legend there. This guarantees the legend never overlaps any bar or
    # error bar regardless of data values.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(GROUP_ORDER),
        frameon=False,
        fontsize=10,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def load_json(path):
    """Load metric values from a JSON file shaped as
    {group: {metric_key: [mean, std]}}.
    """
    with open(path) as f:
        raw = json.load(f)
    return {g: {k: tuple(v) for k, v in metrics.items()}
            for g, metrics in raw.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional JSON file with measured (mean, std) per group/metric. "
             "Falls back to thesis Table 3 values if omitted.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
                / "thesis" / "images" / "metric_comparison.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    values = load_json(args.json) if args.json else DEFAULT_VALUES
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(values, args.out)


if __name__ == "__main__":
    main()

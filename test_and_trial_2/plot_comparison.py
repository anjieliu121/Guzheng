#!/usr/bin/env python3
"""Create comparison bar chart of OA metrics across variants."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"

def main():
    with open(f"{ROOT}/outputs/evaluation/full_metrics.json") as f:
        data = json.load(f)

    variants = data.get("variants", {})
    if not variants:
        print("No variants found in full_metrics.json")
        return

    names = []
    oa_pc = []
    oa_dur = []
    penta = []

    for name, v in sorted(variants.items(), key=lambda x: x[1]["oa"]["OA_pitch_class"], reverse=True):
        names.append(name.replace("_", "\n"))
        oa_pc.append(v["oa"]["OA_pitch_class"])
        oa_dur.append(v["oa"]["OA_duration"])
        penta.append(v["aggregate"]["mean_penta_purity"])

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, oa_pc, width, label='OA Pitch Class', color='steelblue')
    bars2 = ax.bar(x, oa_dur, width, label='OA Duration', color='coral')
    bars3 = ax.bar(x + width, penta, width, label='Pentatonic Purity', color='seagreen', alpha=0.7)

    ax.set_ylabel('Score (0-1)')
    ax.set_title('Guzheng Generation: Model Comparison (OA Metrics)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out_path = f"{ROOT}/outputs/evaluation/model_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Comparison chart saved to {out_path}")

if __name__ == "__main__":
    main()

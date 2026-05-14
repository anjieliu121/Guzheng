"""Rank NotaGen and PatchGPT samples by musicality proximity to training.

Score = mean |z-score| across 22 musicality metrics (lower = closer to training).
Each metric is z-scored against the training distribution's (mean, std).
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_musicality import (
    analyze_file, TRAINING_DIR, TRAINING_SUFFIX,
    NOTAGEN_DIRS, PATCHGPT_DIRS, METRICS,
)

MUSICALITY_KEYS = [m[0] for m in METRICS if m[4] in ('A', 'B', 'C')]


def collect(dirs, suffix=None):
    files = []
    for d in dirs:
        pat = f"*{suffix}" if suffix else "*.mid"
        files.extend(sorted(glob.glob(os.path.join(d, pat))))
    out = []
    for f in files:
        r = analyze_file(f)
        if r is not None:
            out.append((f, r))
    return out


def main():
    print("Analyzing training data …")
    train = collect([TRAINING_DIR], suffix=TRAINING_SUFFIX)
    print(f"  training: {len(train)} files")

    # training mean/std for z-scoring
    ref = {}
    for k in MUSICALITY_KEYS:
        vals = np.array([r[k] for _, r in train], dtype=float)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd < 1e-9:
            sd = 1.0
        ref[k] = (mu, sd)

    print("Analyzing NotaGen …")
    nota = collect(NOTAGEN_DIRS)
    print(f"  notagen: {len(nota)} files")
    print("Analyzing PatchGPT …")
    patch = collect(PATCHGPT_DIRS)
    print(f"  patchgpt: {len(patch)} files")

    def score_set(samples):
        rows = []
        for path, r in samples:
            zs = []
            for k in MUSICALITY_KEYS:
                mu, sd = ref[k]
                zs.append(abs((r[k] - mu) / sd))
            rows.append((path, float(np.mean(zs)), r))
        rows.sort(key=lambda x: x[1])
        return rows

    nota_ranked = score_set(nota)
    patch_ranked = score_set(patch)

    def print_top(label, ranked, n=5):
        print("\n" + "=" * 90)
        print(f"TOP {n} — {label}  (lower mean |z| = closer to training)")
        print("=" * 90)
        print(f"{'rank':>4s} {'mean|z|':>8s} {'dur':>6s} {'N':>5s} {'dens':>5s}  file")
        print("-" * 90)
        for i, (path, s, r) in enumerate(ranked[:n], 1):
            print(f"{i:>4d} {s:>8.3f} {r['duration']:>6.1f} {r['n_notes']:>5d} "
                  f"{r['density']:>5.2f}  {os.path.basename(path)}")
        # show key metrics for the top samples
        key_show = ['autocorr_1', 'conjunct_ratio', 'leap_ratio',
                    'motif_rep_3', 'motif_rep_5', 'compression_ratio',
                    'long_range_recurrence']
        print()
        hdr = f"{'rank':>4s} " + " ".join(f"{k[:11]:>11s}" for k in key_show)
        print(hdr)
        print("-" * len(hdr))
        for i, (path, s, r) in enumerate(ranked[:n], 1):
            row = f"{i:>4d} " + " ".join(f"{r[k]:>11.3f}" for k in key_show)
            print(row)

    print_top("NotaGen fine-tuned (medium, 244M)", nota_ranked)
    print_top("PatchGPT from scratch (242M)", patch_ranked)

    print("\n" + "=" * 90)
    print("DISTRIBUTION OF SCORES")
    print("=" * 90)
    for label, ranked in [("NotaGen", nota_ranked), ("PatchGPT", patch_ranked)]:
        scores = np.array([s for _, s, _ in ranked])
        print(f"  {label:<10s} min={scores.min():.3f}  median={np.median(scores):.3f}  "
              f"mean={scores.mean():.3f}  max={scores.max():.3f}")


if __name__ == "__main__":
    main()

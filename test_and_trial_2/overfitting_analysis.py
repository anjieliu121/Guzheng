#!/usr/bin/env python3
"""
Overfitting analysis: detect how much generated MIDI content is copied from training data.

Approach:
1. Extract pitch sequences from all training and generated MIDI files
2. Find longest common substrings (exact melodic matches)
3. Compute n-gram coverage: what % of generated n-grams exist in training data
4. Report per-file and aggregate statistics
"""

import os, sys, json, collections
import numpy as np
import mido

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"
TRAIN_DIRS = [
    os.path.join(ROOT, "MIDI"),
    os.path.join(ROOT, "MIDI_transposed"),
]


def midi_to_pitch_sequence(midi_path):
    """Extract time-ordered pitch sequence from MIDI file."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((abs_tick, msg.note))
    notes.sort(key=lambda x: (x[0], x[1]))
    return [n[1] for n in notes]


def midi_to_interval_sequence(midi_path):
    """Extract interval sequence (differences between consecutive pitches)."""
    pitches = midi_to_pitch_sequence(midi_path)
    if len(pitches) < 2:
        return []
    return [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]


def extract_ngrams(seq, n):
    """Extract all n-grams from a sequence."""
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]


def longest_common_substring(seq1, seq2):
    """Find length of longest common substring between two sequences."""
    if not seq1 or not seq2:
        return 0, []
    m, n = len(seq1), len(seq2)
    # Use rolling row approach for memory efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    max_len = 0
    end_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
                    end_pos = i
            else:
                curr[j] = 0
        prev, curr = curr, [0] * (n + 1)
    match_seq = seq1[end_pos - max_len:end_pos] if max_len > 0 else []
    return max_len, match_seq


def ngram_coverage(gen_seq, train_ngram_set, n):
    """What fraction of generated n-grams appear in training data."""
    gen_ngrams = extract_ngrams(gen_seq, n)
    if not gen_ngrams:
        return 0.0, 0, 0
    hits = sum(1 for ng in gen_ngrams if ng in train_ngram_set)
    return hits / len(gen_ngrams), hits, len(gen_ngrams)


def find_all_common_runs(gen_seq, train_ngram_set, n):
    """Find all runs of consecutive matching n-grams (i.e. longer copied passages)."""
    gen_ngrams = extract_ngrams(gen_seq, n)
    runs = []
    current_run_start = None
    current_run_len = 0
    for i, ng in enumerate(gen_ngrams):
        if ng in train_ngram_set:
            if current_run_start is None:
                current_run_start = i
                current_run_len = 1
            else:
                current_run_len += 1
        else:
            if current_run_start is not None and current_run_len > 0:
                # A run of current_run_len consecutive matching n-grams
                # means a copied passage of length current_run_len + n - 1
                runs.append((current_run_start, current_run_len + n - 1))
            current_run_start = None
            current_run_len = 0
    if current_run_start is not None and current_run_len > 0:
        runs.append((current_run_start, current_run_len + n - 1))
    return runs


def analyze_generated_dir(gen_dir, train_pitch_seqs, train_interval_seqs,
                          train_pitch_ngrams, train_interval_ngrams):
    """Analyze a single generated output directory."""
    gen_files = sorted([
        os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(".mid")
    ])
    if not gen_files:
        return None

    results = []
    for gf in gen_files:
        gen_pitches = midi_to_pitch_sequence(gf)
        gen_intervals = midi_to_interval_sequence(gf)
        fname = os.path.basename(gf)

        if len(gen_pitches) < 5:
            results.append({"file": fname, "n_notes": len(gen_pitches), "skip": True})
            continue

        # Longest common substring against each training file
        best_lcs_pitch = 0
        best_lcs_train_file = ""
        best_lcs_interval = 0
        best_lcs_interval_file = ""

        for tf_name, tseq in train_pitch_seqs.items():
            lcs_len, _ = longest_common_substring(gen_pitches, tseq)
            if lcs_len > best_lcs_pitch:
                best_lcs_pitch = lcs_len
                best_lcs_train_file = tf_name

        for tf_name, tseq in train_interval_seqs.items():
            lcs_len, _ = longest_common_substring(gen_intervals, tseq)
            if lcs_len > best_lcs_interval:
                best_lcs_interval = lcs_len
                best_lcs_interval_file = tf_name

        # N-gram coverage at various n values
        coverages = {}
        for n in [4, 8, 12, 16, 20]:
            if n in train_pitch_ngrams:
                cov, hits, total = ngram_coverage(gen_pitches, train_pitch_ngrams[n], n)
                coverages[f"pitch_{n}gram"] = {"coverage": round(cov, 4), "hits": hits, "total": total}
            if n in train_interval_ngrams:
                cov, hits, total = ngram_coverage(gen_intervals, train_interval_ngrams[n], n)
                coverages[f"interval_{n}gram"] = {"coverage": round(cov, 4), "hits": hits, "total": total}

        # Find long copied runs (using 8-gram matches)
        if 8 in train_pitch_ngrams:
            runs = find_all_common_runs(gen_pitches, train_pitch_ngrams[8], 8)
            long_runs = [r for r in runs if r[1] >= 16]  # passages of 16+ notes
        else:
            long_runs = []

        results.append({
            "file": fname,
            "n_notes": len(gen_pitches),
            "best_lcs_pitch": best_lcs_pitch,
            "best_lcs_pitch_ratio": round(best_lcs_pitch / len(gen_pitches), 4),
            "best_lcs_train_file": best_lcs_train_file,
            "best_lcs_interval": best_lcs_interval,
            "best_lcs_interval_ratio": round(best_lcs_interval / max(len(gen_intervals), 1), 4),
            "best_lcs_interval_file": best_lcs_interval_file,
            "coverages": coverages,
            "long_copied_runs": len(long_runs),
            "longest_copied_run": max((r[1] for r in long_runs), default=0),
        })

    return results


def main():
    print("=" * 70)
    print("OVERFITTING ANALYSIS")
    print("=" * 70)

    # Load training data
    print("\nLoading training data...")
    train_pitch_seqs = {}
    train_interval_seqs = {}
    for td in TRAIN_DIRS:
        if not os.path.isdir(td):
            continue
        for f in sorted(os.listdir(td)):
            if not f.endswith(".mid"):
                continue
            fp = os.path.join(td, f)
            key = f"{os.path.basename(td)}/{f}"
            pitches = midi_to_pitch_sequence(fp)
            intervals = midi_to_interval_sequence(fp)
            if pitches:
                train_pitch_seqs[key] = pitches
                train_interval_seqs[key] = intervals
    print(f"  Loaded {len(train_pitch_seqs)} training files")
    total_train_notes = sum(len(s) for s in train_pitch_seqs.values())
    print(f"  Total training notes: {total_train_notes}")

    # Build training n-gram sets
    print("\nBuilding training n-gram index...")
    train_pitch_ngrams = {}
    train_interval_ngrams = {}
    for n in [4, 8, 12, 16, 20]:
        pitch_set = set()
        interval_set = set()
        for seq in train_pitch_seqs.values():
            pitch_set.update(extract_ngrams(seq, n))
        for seq in train_interval_seqs.values():
            interval_set.update(extract_ngrams(seq, n))
        train_pitch_ngrams[n] = pitch_set
        train_interval_ngrams[n] = interval_set
        print(f"  {n}-gram: {len(pitch_set)} unique pitch, {len(interval_set)} unique interval")

    # Analyze each generated output directory
    output_dirs = {}
    outputs_root = os.path.join(ROOT, "outputs")
    for d in sorted(os.listdir(outputs_root)):
        dp = os.path.join(outputs_root, d)
        if os.path.isdir(dp) and any(f.endswith(".mid") for f in os.listdir(dp)):
            output_dirs[d] = dp

    # Also check final/best_samples
    best_samples = os.path.join(outputs_root, "final/best_samples")
    if os.path.isdir(best_samples):
        output_dirs["final_best_samples"] = best_samples

    print(f"\nAnalyzing {len(output_dirs)} output directories...\n")

    all_results = {}
    for name, path in sorted(output_dirs.items()):
        print(f"--- {name} ---")
        results = analyze_generated_dir(path, train_pitch_seqs, train_interval_seqs,
                                        train_pitch_ngrams, train_interval_ngrams)
        if not results:
            print("  (no files)")
            continue

        valid = [r for r in results if not r.get("skip")]
        if not valid:
            print("  (all files too short)")
            continue

        # Summary stats
        avg_lcs_pitch = np.mean([r["best_lcs_pitch"] for r in valid])
        avg_lcs_ratio = np.mean([r["best_lcs_pitch_ratio"] for r in valid])
        avg_lcs_interval = np.mean([r["best_lcs_interval"] for r in valid])
        avg_lcs_int_ratio = np.mean([r["best_lcs_interval_ratio"] for r in valid])

        print(f"  Files: {len(valid)}")
        print(f"  Avg notes: {np.mean([r['n_notes'] for r in valid]):.0f}")
        print(f"  Best LCS (pitch):    avg={avg_lcs_pitch:.1f} notes, avg_ratio={avg_lcs_ratio:.3f}")
        print(f"  Best LCS (interval): avg={avg_lcs_interval:.1f}, avg_ratio={avg_lcs_int_ratio:.3f}")

        # N-gram coverage summary
        for n in [4, 8, 16]:
            key_p = f"pitch_{n}gram"
            key_i = f"interval_{n}gram"
            p_covs = [r["coverages"].get(key_p, {}).get("coverage", 0) for r in valid]
            i_covs = [r["coverages"].get(key_i, {}).get("coverage", 0) for r in valid]
            if p_covs:
                print(f"  {n}-gram coverage: pitch={np.mean(p_covs):.3f}, interval={np.mean(i_covs):.3f}")

        # Long copied passages
        total_long_runs = sum(r.get("long_copied_runs", 0) for r in valid)
        max_run = max((r.get("longest_copied_run", 0) for r in valid), default=0)
        if total_long_runs > 0:
            print(f"  Copied passages (16+ notes): {total_long_runs} total, longest={max_run} notes")

        # Per-file details for files with high overlap
        for r in valid:
            if r["best_lcs_pitch_ratio"] > 0.3 or r.get("long_copied_runs", 0) > 0:
                print(f"    ** {r['file']}: LCS={r['best_lcs_pitch']} ({r['best_lcs_pitch_ratio']:.1%}) "
                      f"from {r['best_lcs_train_file']}, "
                      f"long_runs={r.get('long_copied_runs', 0)}")

        all_results[name] = {
            "summary": {
                "n_files": len(valid),
                "avg_notes": round(float(np.mean([r['n_notes'] for r in valid])), 1),
                "avg_lcs_pitch": round(float(avg_lcs_pitch), 1),
                "avg_lcs_pitch_ratio": round(float(avg_lcs_ratio), 4),
                "avg_lcs_interval": round(float(avg_lcs_interval), 1),
                "avg_lcs_interval_ratio": round(float(avg_lcs_int_ratio), 4),
            },
            "per_file": valid,
        }
        print()

    # Overall summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Variant':<45} {'Files':<6} {'Notes':<7} {'LCS':<6} {'LCS%':<7} "
          f"{'8g_cov':<8} {'16g_cov':<8} {'Runs'}")
    print("-" * 90)
    for name, res in sorted(all_results.items()):
        s = res["summary"]
        valid = [r for r in res["per_file"] if not r.get("skip")]
        cov_8 = np.mean([r["coverages"].get("pitch_8gram", {}).get("coverage", 0) for r in valid])
        cov_16 = np.mean([r["coverages"].get("pitch_16gram", {}).get("coverage", 0) for r in valid])
        total_runs = sum(r.get("long_copied_runs", 0) for r in valid)
        print(f"{name:<45} {s['n_files']:<6} {s['avg_notes']:<7} "
              f"{s['avg_lcs_pitch']:<6.0f} {s['avg_lcs_pitch_ratio']:<7.3f} "
              f"{cov_8:<8.3f} {cov_16:<8.3f} {total_runs}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
LCS (Longest Common Substring):
  - LCS% > 0.5  → More than half the piece is a direct copy. SEVERE overfitting.
  - LCS% 0.3-0.5 → Large chunks copied. Significant overfitting.
  - LCS% 0.1-0.3 → Some phrases copied. Moderate memorization.
  - LCS% < 0.1  → Minimal direct copying.

N-gram coverage (what fraction of generated n-grams appear in training):
  - 4-gram coverage ~1.0 is normal (short motifs will naturally overlap)
  - 8-gram coverage > 0.5 → Substantial phrase copying
  - 16-gram coverage > 0.2 → Long passages directly from training
  - 16-gram coverage > 0.5 → SEVERE overfitting

Copied passages (runs of 16+ consecutive notes matching training):
  - 0 runs → Good, no long verbatim copies
  - >0 runs → Direct memorization detected
""")

    # Save to JSON
    out_path = os.path.join(ROOT, "outputs/evaluation/overfitting_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()

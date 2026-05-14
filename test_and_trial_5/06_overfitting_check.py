#!/usr/bin/env python3
"""
Step 6: Overfitting / plagiarism / repetition detection (Trial 4).

Stricter than Trial 3:
- LCS ratio threshold: 0.20 (was 30 absolute notes)
- 5-gram coverage threshold: 0.50
- Per-prompt-category analysis (val, test, synthetic)
- Compare to test set baseline

Evaluates post-processed files in generated/<checkpoint>/<category>_postprocessed/.
"""

import os
import json
import mido
import numpy as np
from collections import Counter

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_pitch_sequence(midi_path):
    """Extract ordered pitch sequence from MIDI."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []

    notes = []
    for track in mid.tracks:
        abs_time = 0
        pending = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.note, msg.channel)] = abs_time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in pending:
                    onset = pending.pop(key)
                    notes.append((onset, msg.note))
    notes.sort()
    return [n[1] for n in notes]


def extract_interval_sequence(pitches):
    return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]


def ngram_coverage(gen_seq, train_seqs, n=5):
    """What fraction of n-grams in gen_seq appear in any training sequence."""
    if len(gen_seq) < n:
        return 0.0

    train_ngrams = set()
    for seq in train_seqs:
        for i in range(len(seq) - n + 1):
            train_ngrams.add(tuple(seq[i:i+n]))

    gen_ngrams = [tuple(gen_seq[i:i+n]) for i in range(len(gen_seq) - n + 1)]
    if not gen_ngrams:
        return 0.0

    matches = sum(1 for ng in gen_ngrams if ng in train_ngrams)
    return matches / len(gen_ngrams)


def self_repetition_score(pitches, n=4):
    """What fraction of n-grams appear more than once."""
    if len(pitches) < n + 1:
        return 0.0
    ngrams = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for ng in ngrams if counts[ng] > 1)
    return repeated / len(ngrams)


def longest_repeated_substring_len(pitches):
    """Find the longest substring that appears at least twice."""
    if len(pitches) < 4:
        return 0
    n = len(pitches)
    if n > 3000:
        pitches = pitches[:3000]
        n = 3000

    max_len = 0
    for length in range(min(n // 2, 50), 1, -1):
        seen = set()
        found = False
        for i in range(n - length + 1):
            sub = tuple(pitches[i:i+length])
            if sub in seen:
                max_len = max(max_len, length)
                found = True
                break
            seen.add(sub)
        if found:
            break
    return max_len


def longest_common_substring_len(seq1, seq2):
    """Find length of longest common contiguous substring."""
    if not seq1 or not seq2:
        return 0
    len1, len2 = len(seq1), len(seq2)
    if len1 > 5000:
        seq1 = seq1[:5000]
        len1 = 5000
    if len2 > 5000:
        seq2 = seq2[:5000]
        len2 = 5000

    max_len = 0
    for i in range(len1):
        for j in range(len2):
            k = 0
            while (i + k < len1 and j + k < len2 and seq1[i + k] == seq2[j + k]):
                k += 1
            max_len = max(max_len, k)
            if max_len > 50:
                return max_len
    return max_len


def load_training_sequences(train_dir):
    """Load pitch and interval sequences from training data."""
    pitch_seqs = []
    interval_seqs = []
    filenames = []
    if os.path.isdir(train_dir):
        for f in sorted(os.listdir(train_dir)):
            if f.endswith(".mid"):
                pitches = extract_pitch_sequence(os.path.join(train_dir, f))
                if pitches:
                    pitch_seqs.append(pitches)
                    interval_seqs.append(extract_interval_sequence(pitches))
                    filenames.append(f)
    return pitch_seqs, interval_seqs, filenames


def analyze_generated_dir(var_dir, train_pitch_seqs, train_interval_seqs, train_filenames):
    """Analyze a directory of generated MIDI files with stricter thresholds."""
    file_results = []

    for f in sorted(os.listdir(var_dir)):
        if not f.endswith(".mid"):
            continue

        pitches = extract_pitch_sequence(os.path.join(var_dir, f))
        if not pitches or len(pitches) < 5:
            continue

        intervals = extract_interval_sequence(pitches)

        # N-gram coverage
        coverage = {}
        for n in [3, 5, 8, 12]:
            pitch_cov = ngram_coverage(pitches, train_pitch_seqs, n)
            interval_cov = ngram_coverage(intervals, train_interval_seqs, n)
            coverage[f"pitch_{n}gram"] = round(pitch_cov, 4)
            coverage[f"interval_{n}gram"] = round(interval_cov, 4)

        # Self-repetition
        self_rep = {}
        for n in [4, 8, 12]:
            self_rep[f"self_rep_{n}gram"] = round(self_repetition_score(pitches, n), 4)
        longest_self_rep = longest_repeated_substring_len(pitches)
        self_rep["longest_self_repeat"] = longest_self_rep

        # Longest common substring with any training piece
        max_lcs = 0
        closest_train = ""
        for i, tseq in enumerate(train_pitch_seqs):
            lcs = longest_common_substring_len(pitches, tseq)
            if lcs > max_lcs:
                max_lcs = lcs
                closest_train = train_filenames[i] if i < len(train_filenames) else "?"

        lcs_ratio = max_lcs / max(len(pitches), 1)

        entry = {
            "file": f,
            "n_notes": len(pitches),
            "ngram_coverage": coverage,
            "self_repetition": self_rep,
            "max_lcs_length": max_lcs,
            "lcs_ratio": round(lcs_ratio, 4),
            "closest_train_file": closest_train,
        }
        file_results.append(entry)

        # Stricter flags (Trial 4 thresholds)
        flags = []
        if lcs_ratio > 0.20:
            flags.append("MEMORIZATION(lcs>0.20)")
        if coverage.get("pitch_5gram", 0) > 0.50:
            flags.append("MEMORIZATION(5gram>0.50)")
        if coverage.get("pitch_8gram", 0) > 0.50:
            flags.append("MEMORIZATION(8gram>0.50)")
        if longest_self_rep > 20 or self_rep.get("self_rep_4gram", 0) > 0.6:
            flags.append("REPETITIVE")

        flag_str = " ".join(flags) if flags else ""
        print(f"  {f}: n={len(pitches)}, lcs={max_lcs}({lcs_ratio:.2f}), "
              f"5gram={coverage.get('pitch_5gram', 0):.2f}, "
              f"self4={self_rep.get('self_rep_4gram', 0):.2f}, "
              f"selfmax={longest_self_rep} {flag_str}")

    return file_results


def summarize_results(file_results):
    if not file_results:
        return {}
    return {
        "n_files": len(file_results),
        "mean_n_notes": round(np.mean([r["n_notes"] for r in file_results]), 1),
        "mean_pitch_5gram_coverage": round(np.mean([r["ngram_coverage"]["pitch_5gram"] for r in file_results]), 4),
        "mean_pitch_8gram_coverage": round(np.mean([r["ngram_coverage"]["pitch_8gram"] for r in file_results]), 4),
        "mean_pitch_12gram_coverage": round(np.mean([r["ngram_coverage"]["pitch_12gram"] for r in file_results]), 4),
        "mean_max_lcs": round(np.mean([r["max_lcs_length"] for r in file_results]), 1),
        "mean_lcs_ratio": round(np.mean([r["lcs_ratio"] for r in file_results]), 4),
        "mean_self_rep_4gram": round(np.mean([r["self_repetition"]["self_rep_4gram"] for r in file_results]), 4),
        "mean_self_rep_8gram": round(np.mean([r["self_repetition"]["self_rep_8gram"] for r in file_results]), 4),
        "mean_longest_self_repeat": round(np.mean([r["self_repetition"]["longest_self_repeat"] for r in file_results]), 1),
        "n_memorization_lcs": sum(1 for r in file_results if r["lcs_ratio"] > 0.20),
        "n_memorization_5gram": sum(1 for r in file_results if r["ngram_coverage"]["pitch_5gram"] > 0.50),
        "n_excessive_repetition": sum(
            1 for r in file_results
            if r["self_repetition"]["longest_self_repeat"] > 20
            or r["self_repetition"].get("self_rep_4gram", 0) > 0.6
        ),
    }


def main():
    eval_dir = os.path.join(TRIAL_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Load training data (original only, not augmented, since augmented have same pitches)
    train_dir = os.path.join(TRIAL_ROOT, "data", "train")
    print("Loading training data pitch sequences...")
    all_pitch_seqs, all_interval_seqs, all_filenames = load_training_sequences(train_dir)

    # Filter to original files only for LCS check (augmented have same pitch sequences)
    orig_pitch_seqs = []
    orig_interval_seqs = []
    orig_filenames = []
    for ps, ints, fn in zip(all_pitch_seqs, all_interval_seqs, all_filenames):
        if not fn.startswith("aug_"):
            orig_pitch_seqs.append(ps)
            orig_interval_seqs.append(ints)
            orig_filenames.append(fn)
    print(f"  {len(orig_pitch_seqs)} original training sequences loaded (excluding augmented)")

    # Reference: test set coverage
    test_dir = os.path.join(TRIAL_ROOT, "data", "test")
    test_pitch_seqs, _, _ = load_training_sequences(test_dir)

    results = {}

    # Analyze all generated checkpoint variants (post-processed)
    gen_dir = os.path.join(TRIAL_ROOT, "generated")
    if os.path.isdir(gen_dir):
        for ckpt_name in sorted(os.listdir(gen_dir)):
            ckpt_gen_dir = os.path.join(gen_dir, ckpt_name)
            if not os.path.isdir(ckpt_gen_dir):
                continue

            for category in ["val_postprocessed", "test_postprocessed", "synthetic_postprocessed"]:
                var_dir = os.path.join(ckpt_gen_dir, category)
                if not os.path.isdir(var_dir):
                    continue

                variant_name = f"{ckpt_name}/{category}"
                print(f"\nAnalyzing {variant_name}...")
                file_results = analyze_generated_dir(
                    var_dir, orig_pitch_seqs, orig_interval_seqs, orig_filenames
                )
                if file_results:
                    results[variant_name] = {
                        "files": file_results,
                        "summary": summarize_results(file_results),
                    }

            # Combined analysis
            combined_results = []
            for category in ["val_postprocessed", "test_postprocessed", "synthetic_postprocessed"]:
                var_dir = os.path.join(ckpt_gen_dir, category)
                if os.path.isdir(var_dir):
                    combined_results.extend(
                        analyze_generated_dir(var_dir, orig_pitch_seqs, orig_interval_seqs, orig_filenames)
                    )
            if combined_results:
                results[f"{ckpt_name}/all_postprocessed"] = {
                    "files": combined_results,
                    "summary": summarize_results(combined_results),
                }

    # Test reference
    if test_pitch_seqs and orig_pitch_seqs:
        print("\nReference: test set n-gram coverage against training...")
        test_covs_5 = []
        test_covs_8 = []
        test_lcs_ratios = []
        for tseq in test_pitch_seqs:
            test_covs_5.append(ngram_coverage(tseq, orig_pitch_seqs, 5))
            test_covs_8.append(ngram_coverage(tseq, orig_pitch_seqs, 8))
            max_lcs = 0
            for train_seq in orig_pitch_seqs:
                lcs = longest_common_substring_len(tseq, train_seq)
                max_lcs = max(max_lcs, lcs)
            test_lcs_ratios.append(max_lcs / max(len(tseq), 1))
        print(f"  Mean test 5-gram coverage: {np.mean(test_covs_5):.4f}")
        print(f"  Mean test 8-gram coverage: {np.mean(test_covs_8):.4f}")
        print(f"  Mean test LCS ratio: {np.mean(test_lcs_ratios):.4f}")
        results["test_reference"] = {
            "mean_pitch_5gram_coverage": round(np.mean(test_covs_5), 4),
            "mean_pitch_8gram_coverage": round(np.mean(test_covs_8), 4),
            "mean_lcs_ratio": round(np.mean(test_lcs_ratios), 4),
        }

    # Save results
    json_path = os.path.join(eval_dir, "overfitting_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("OVERFITTING & MEMORIZATION ANALYSIS SUMMARY (Trial 4)")
    print(f"{'='*80}")

    # Thresholds
    print("\nThresholds: LCS ratio > 0.20 = memorization, 5-gram > 0.50 = memorization")
    print("")

    for variant, data in results.items():
        if variant == "test_reference":
            print(f"Test set baseline:")
            print(f"  Mean 5-gram coverage: {data['mean_pitch_5gram_coverage']}")
            print(f"  Mean 8-gram coverage: {data['mean_pitch_8gram_coverage']}")
            print(f"  Mean LCS ratio:       {data['mean_lcs_ratio']}")
        else:
            s = data["summary"]
            print(f"\n{variant}: ({s['n_files']} files, mean {s['mean_n_notes']:.0f} notes)")
            print(f"  Mean pitch 5-gram coverage: {s['mean_pitch_5gram_coverage']}")
            print(f"  Mean pitch 8-gram coverage: {s['mean_pitch_8gram_coverage']}")
            print(f"  Mean LCS ratio:             {s['mean_lcs_ratio']}")
            print(f"  Mean max LCS length:        {s['mean_max_lcs']}")
            print(f"  Mean self-rep (4-gram):      {s['mean_self_rep_4gram']}")
            print(f"  Mean longest self-repeat:    {s['mean_longest_self_repeat']}")
            print(f"  Memorization (LCS>0.20):     {s['n_memorization_lcs']}/{s['n_files']} files")
            print(f"  Memorization (5gram>0.50):   {s['n_memorization_5gram']}/{s['n_files']} files")
            print(f"  Excessive repetition:        {s['n_excessive_repetition']}/{s['n_files']} files")

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()

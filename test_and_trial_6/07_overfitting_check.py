#!/usr/bin/env python3
"""
Step 7: Overfitting / plagiarism detection.

Checks:
1. N-gram coverage: what % of generated n-grams appear in training data
2. Longest common subsequence (pitch sequence) between generated and training
3. Per-file similarity scores
"""

import os
import json
import mido
import numpy as np
from collections import Counter

from config import trial_root


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
    """Convert pitch sequence to interval sequence (pitch-agnostic)."""
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


def longest_common_substring_len(seq1, seq2):
    """Find length of longest common contiguous substring."""
    if not seq1 or not seq2:
        return 0

    # Use rolling hash for efficiency with long sequences
    max_len = 0
    len1, len2 = len(seq1), len(seq2)

    # Limit to reasonable comparison length
    if len1 > 5000:
        seq1 = seq1[:5000]
        len1 = 5000
    if len2 > 5000:
        seq2 = seq2[:5000]
        len2 = 5000

    for i in range(len1):
        for j in range(len2):
            k = 0
            while (i + k < len1 and j + k < len2 and seq1[i + k] == seq2[j + k]):
                k += 1
            max_len = max(max_len, k)
            if max_len > 50:  # Early exit if clearly memorized
                return max_len
    return max_len


def main():
    root = trial_root()
    eval_dir = os.path.join(root, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Load training pitch sequences
    train_dir = os.path.join(root, "data", "train")
    print("Loading training data pitch sequences...")
    train_pitch_seqs = []
    train_interval_seqs = []
    if os.path.isdir(train_dir):
        for f in sorted(os.listdir(train_dir)):
            if f.endswith(".mid") and not f.startswith("aug_"):
                pitches = extract_pitch_sequence(os.path.join(train_dir, f))
                if pitches:
                    train_pitch_seqs.append(pitches)
                    train_interval_seqs.append(extract_interval_sequence(pitches))
    print(f"  {len(train_pitch_seqs)} training sequences loaded")

    # Load test set for reference
    test_dir = os.path.join(root, "data", "test")
    test_pitch_seqs = []
    if os.path.isdir(test_dir):
        for f in sorted(os.listdir(test_dir)):
            if f.endswith(".mid"):
                pitches = extract_pitch_sequence(os.path.join(test_dir, f))
                if pitches:
                    test_pitch_seqs.append(pitches)

    # Analyze generated files
    gen_dir = os.path.join(root, "generated")
    results = {}

    for variant in ["constrained", "unconstrained"]:
        var_dir = os.path.join(gen_dir, variant)
        if not os.path.isdir(var_dir):
            continue

        print(f"\nAnalyzing {variant} generation...")
        file_results = []

        for f in sorted(os.listdir(var_dir)):
            if not f.endswith(".mid"):
                continue

            pitches = extract_pitch_sequence(os.path.join(var_dir, f))
            if not pitches or len(pitches) < 5:
                continue

            intervals = extract_interval_sequence(pitches)

            # N-gram coverage at different n values
            coverage = {}
            for n in [3, 5, 8, 12]:
                pitch_cov = ngram_coverage(pitches, train_pitch_seqs, n)
                interval_cov = ngram_coverage(intervals, train_interval_seqs, n)
                coverage[f"pitch_{n}gram"] = round(pitch_cov, 4)
                coverage[f"interval_{n}gram"] = round(interval_cov, 4)

            # Longest common substring with any training piece
            max_lcs = 0
            closest_train = ""
            for i, tseq in enumerate(train_pitch_seqs):
                lcs = longest_common_substring_len(pitches, tseq)
                if lcs > max_lcs:
                    max_lcs = lcs
                    train_files = sorted(
                        tf for tf in os.listdir(train_dir)
                        if tf.endswith(".mid") and not tf.startswith("aug_")
                    )
                    closest_train = train_files[i] if i < len(train_files) else "?"

            entry = {
                "file": f,
                "n_notes": len(pitches),
                "ngram_coverage": coverage,
                "max_lcs_length": max_lcs,
                "lcs_ratio": round(max_lcs / max(len(pitches), 1), 4),
                "closest_train_file": closest_train,
            }
            file_results.append(entry)

            flag = " POSSIBLE MEMORIZATION" if max_lcs > 30 or coverage.get("pitch_8gram", 0) > 0.5 else ""
            print(f"  {f}: lcs={max_lcs}, 5gram_cov={coverage.get('pitch_5gram', 0):.2f}{flag}")

        if file_results:
            avg_5gram = np.mean([r["ngram_coverage"]["pitch_5gram"] for r in file_results])
            avg_lcs = np.mean([r["max_lcs_length"] for r in file_results])
            results[variant] = {
                "files": file_results,
                "summary": {
                    "mean_pitch_5gram_coverage": round(avg_5gram, 4),
                    "mean_max_lcs": round(avg_lcs, 1),
                    "n_possible_memorization": sum(
                        1 for r in file_results
                        if r["max_lcs_length"] > 30 or r["ngram_coverage"].get("pitch_8gram", 0) > 0.5
                    ),
                }
            }

    # Reference: test set n-gram coverage (expected to be low)
    if test_pitch_seqs and train_pitch_seqs:
        print("\nReference: test set n-gram coverage against training...")
        test_covs = []
        for tseq in test_pitch_seqs:
            cov = ngram_coverage(tseq, train_pitch_seqs, 5)
            test_covs.append(cov)
        print(f"  Mean test 5-gram coverage: {np.mean(test_covs):.4f}")
        results["test_reference"] = {
            "mean_pitch_5gram_coverage": round(np.mean(test_covs), 4),
        }

    # Save results
    json_path = os.path.join(eval_dir, "overfitting_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("OVERFITTING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    for variant, data in results.items():
        if variant == "test_reference":
            print(f"\nTest set baseline: mean 5-gram coverage = {data['mean_pitch_5gram_coverage']}")
        else:
            s = data["summary"]
            print(f"\n{variant}:")
            print(f"  Mean pitch 5-gram coverage: {s['mean_pitch_5gram_coverage']}")
            print(f"  Mean max LCS length: {s['mean_max_lcs']}")
            print(f"  Possible memorization: {s['n_possible_memorization']} files")

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()

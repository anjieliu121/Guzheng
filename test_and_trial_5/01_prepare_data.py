#!/usr/bin/env python3
"""
Step 1: Prepare training data for Trial 5.

Splits MIDI_transposed/ (590 files, 125 pieces) into train/val/test by piece name.
All transpositions of a piece go to the same split to prevent leakage.

For training: selects 2 transpositions per piece (ensuring all 5 scales are
well-represented) to keep training fast (~200 files, ~18h total pipeline).
Val and test use all available transpositions for thorough evaluation.

Split: 80% train / 10% val / 10% test (by unique piece, not file).
"""

import os
import shutil
import random
from collections import defaultdict, Counter

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRIAL_ROOT)

SOURCE_DIR = os.path.join(REPO_ROOT, "MIDI_transposed")
DATA_DIR = os.path.join(TRIAL_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SEED = 42
SCALES = {"A", "C", "D", "F", "G"}
TRAIN_TRANSPOSITIONS_PER_PIECE = 2


def extract_piece_name(filename):
    """Strip scale suffix to get piece name: 'bu_bu_gao_A.mid' -> 'bu_bu_gao'."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in SCALES:
        return parts[0]
    return base


def extract_scale(filename):
    """Get scale from filename: 'bu_bu_gao_A.mid' -> 'A'."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in SCALES:
        return parts[1]
    return None


def select_train_transpositions(pieces, train_pieces, n_per_piece, rng):
    """Select n transpositions per piece, balancing scale representation.

    Uses a greedy approach: for each piece, prefer scales that are
    underrepresented so far.
    """
    scale_counts = Counter()
    selected = {}

    # Shuffle piece order for fairness
    ordered = list(train_pieces)
    rng.shuffle(ordered)

    for piece in ordered:
        files = pieces[piece]
        if len(files) <= n_per_piece:
            # Piece has fewer transpositions than requested — use all
            selected[piece] = files
            for f in files:
                s = extract_scale(f)
                if s:
                    scale_counts[s] += 1
            continue

        # Score each file by how underrepresented its scale is
        file_scales = [(f, extract_scale(f)) for f in files]
        # Sort by scale count (ascending) so underrepresented scales are picked first
        file_scales.sort(key=lambda x: (scale_counts.get(x[1], 0), x[1]))

        chosen = []
        for f, s in file_scales:
            if len(chosen) >= n_per_piece:
                break
            chosen.append(f)
            if s:
                scale_counts[s] += 1

        selected[piece] = chosen

    return selected, scale_counts


def main():
    print("=" * 60)
    print("STEP 1: PREPARE DATA (Trial 5)")
    print("=" * 60)

    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return

    # Collect files grouped by piece name
    all_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".mid"))
    print(f"Total MIDI files in MIDI_transposed/: {len(all_files)}")

    pieces = defaultdict(list)
    for f in all_files:
        piece = extract_piece_name(f)
        pieces[piece].append(f)

    piece_names = sorted(pieces.keys())
    print(f"Unique pieces: {len(piece_names)}")

    # Shuffle and split by piece
    rng = random.Random(SEED)
    rng.shuffle(piece_names)

    n = len(piece_names)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train_pieces = piece_names[:n_train]
    val_pieces = piece_names[n_train:n_train + n_val]
    test_pieces = piece_names[n_train + n_val:]

    print(f"\nSplit (by piece): {len(train_pieces)} train / {len(val_pieces)} val / {len(test_pieces)} test")

    # Select subset of transpositions for training
    train_selected, scale_dist = select_train_transpositions(
        pieces, train_pieces, TRAIN_TRANSPOSITIONS_PER_PIECE, rng
    )
    train_file_count = sum(len(v) for v in train_selected.values())
    print(f"\nTraining: {TRAIN_TRANSPOSITIONS_PER_PIECE} transpositions/piece "
          f"-> {train_file_count} files (from {len(train_pieces)} pieces)")
    print(f"  Scale distribution: {dict(sorted(scale_dist.items()))}")

    # Create directories and copy files
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".mid"):
                os.remove(os.path.join(d, f))

    # Copy training files (subsampled)
    train_count = 0
    for piece in train_pieces:
        for fname in train_selected[piece]:
            src = os.path.join(SOURCE_DIR, fname)
            dst = os.path.join(TRAIN_DIR, fname)
            shutil.copy2(src, dst)
            train_count += 1
    print(f"  train: {train_count} files copied")

    # Copy val and test files (all transpositions)
    for split_name, split_pieces, split_dir in [("val", val_pieces, VAL_DIR), ("test", test_pieces, TEST_DIR)]:
        count = 0
        for piece in split_pieces:
            for fname in pieces[piece]:
                src = os.path.join(SOURCE_DIR, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copy2(src, dst)
                count += 1
        print(f"  {split_name}: {count} files ({len(split_pieces)} pieces, all transpositions)")

    # Print piece lists for reference
    print(f"\nVal pieces ({len(val_pieces)}):")
    for p in sorted(val_pieces):
        print(f"  {p} ({len(pieces[p])} transpositions)")

    print(f"\nTest pieces ({len(test_pieces)}):")
    for p in sorted(test_pieces):
        print(f"  {p} ({len(pieces[p])} transpositions)")

    total = sum(
        len([f for f in os.listdir(d) if f.endswith(".mid")])
        for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]
    )
    print(f"\nTotal files across splits: {total}")
    print(f"Data prepared in: {DATA_DIR}")


if __name__ == "__main__":
    main()

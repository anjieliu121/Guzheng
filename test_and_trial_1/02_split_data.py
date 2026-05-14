#!/usr/bin/env python3
"""
Step 2: Create train/val/test split at piece level.

- Groups curated files by base piece name (all transpositions together)
- Holds out 2 pieces for val, 2 for test (with all their transpositions)
- Scraped data: uses existing train/val/test directory structure
- Copies files to data/train/, data/val/, data/test/
- Saves split manifest to data/splits/split.json
"""

import os
import json
import random
import shutil

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TRIAL_ROOT, "data")

CURATED_DIR = os.path.join(DATA_DIR, "curated")
SCRAPED_DIR = os.path.join(DATA_DIR, "scraped")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SEED = 42
N_VAL_PIECES = 2
N_TEST_PIECES = 2


def piece_base_name(filename):
    """Extract base piece name, removing scale suffix."""
    name = os.path.splitext(filename)[0]
    for suffix in ("_A", "_C", "_D", "_F", "_G"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def split_curated():
    """Split curated files by piece (all transpositions of a piece go together)."""
    if not os.path.isdir(CURATED_DIR):
        print("No curated directory found")
        return {}, {}, {}

    files = sorted(f for f in os.listdir(CURATED_DIR) if f.endswith(".mid"))

    # Group by piece
    pieces = {}
    for f in files:
        base = piece_base_name(f)
        pieces.setdefault(base, []).append(f)

    piece_names = sorted(pieces.keys())
    print(f"Curated: {len(files)} files from {len(piece_names)} unique pieces")

    # Shuffle deterministically and split
    random.seed(SEED)
    random.shuffle(piece_names)

    test_pieces = set(piece_names[:N_TEST_PIECES])
    val_pieces = set(piece_names[N_TEST_PIECES:N_TEST_PIECES + N_VAL_PIECES])
    train_pieces = set(piece_names[N_TEST_PIECES + N_VAL_PIECES:])

    train_files = {f: "curated" for p in train_pieces for f in pieces[p]}
    val_files = {f: "curated" for p in val_pieces for f in pieces[p]}
    test_files = {f: "curated" for p in test_pieces for f in pieces[p]}

    print(f"  Train pieces: {sorted(train_pieces)}")
    print(f"  Val pieces:   {sorted(val_pieces)}")
    print(f"  Test pieces:  {sorted(test_pieces)}")
    print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    return train_files, val_files, test_files


def split_scraped():
    """Use existing train/val/test naming from tech99 data."""
    if not os.path.isdir(SCRAPED_DIR):
        print("No scraped directory found")
        return {}, {}, {}

    files = sorted(f for f in os.listdir(SCRAPED_DIR) if f.endswith(".mid"))

    train_files = {}
    val_files = {}
    test_files = {}

    for f in files:
        if "train" in f:
            train_files[f] = "scraped"
        elif "validation" in f:
            val_files[f] = "scraped"
        elif "test" in f:
            test_files[f] = "scraped"
        else:
            # Unknown split, put in training
            train_files[f] = "scraped"

    print(f"Scraped: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    return train_files, val_files, test_files


def copy_files(file_dict, src_dirs, dst_dir):
    """Copy files from source directories to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname, source in file_dict.items():
        if source == "curated":
            src = os.path.join(CURATED_DIR, fname)
        else:
            src = os.path.join(SCRAPED_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


def main():
    print("=" * 60)
    print("STEP 2: TRAIN/VAL/TEST SPLIT")
    print("=" * 60)

    # Split each source
    c_train, c_val, c_test = split_curated()
    s_train, s_val, s_test = split_scraped()

    # Merge
    train = {**c_train, **s_train}
    val = {**c_val, **s_val}
    test = {**c_test, **s_test}

    print(f"\nCombined: train={len(train)}, val={len(val)}, test={len(test)}")

    # Copy files
    for split_name, files, dst in [("train", train, TRAIN_DIR),
                                     ("val", val, VAL_DIR),
                                     ("test", test, TEST_DIR)]:
        copy_files(files, [CURATED_DIR, SCRAPED_DIR], dst)
        actual = len([f for f in os.listdir(dst) if f.endswith(".mid")]) if os.path.isdir(dst) else 0
        print(f"  {split_name}: {actual} files copied to {dst}")

    # Save split manifest
    split_info = {
        "seed": SEED,
        "n_val_pieces": N_VAL_PIECES,
        "n_test_pieces": N_TEST_PIECES,
        "train": {f: s for f, s in sorted(train.items())},
        "val": {f: s for f, s in sorted(val.items())},
        "test": {f: s for f, s in sorted(test.items())},
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }

    os.makedirs(os.path.join(DATA_DIR, "splits"), exist_ok=True)
    split_path = os.path.join(DATA_DIR, "splits", "split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit manifest saved to: {split_path}")


if __name__ == "__main__":
    main()

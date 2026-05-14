#!/usr/bin/env python3
"""
Step 2: Create train/val/test split at piece level.

- Groups curated files by base piece name (all transpositions together)
- Splits pittstate files by piece name (no transpositions)
- Scraped data: uses existing train/val/test naming from tech99
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
PITTSTATE_DIR = os.path.join(DATA_DIR, "pittstate")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SEED = 42
N_VAL_PIECES = 2
N_TEST_PIECES = 2

# Pittstate: hold out 2 for val, 2 for test
PITTSTATE_N_VAL = 2
PITTSTATE_N_TEST = 2


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


def split_pittstate():
    """Split pittstate files by piece name (each file is one piece)."""
    if not os.path.isdir(PITTSTATE_DIR):
        print("No pittstate directory found")
        return {}, {}, {}

    files = sorted(f for f in os.listdir(PITTSTATE_DIR) if f.endswith(".mid"))
    print(f"Pittstate: {len(files)} files")

    if len(files) == 0:
        return {}, {}, {}

    # Shuffle deterministically and split
    random.seed(SEED + 1)  # Different seed to avoid correlation with curated split
    shuffled = list(files)
    random.shuffle(shuffled)

    n_test = min(PITTSTATE_N_TEST, len(shuffled))
    n_val = min(PITTSTATE_N_VAL, len(shuffled) - n_test)

    test_files = {f: "pittstate" for f in shuffled[:n_test]}
    val_files = {f: "pittstate" for f in shuffled[n_test:n_test + n_val]}
    train_files = {f: "pittstate" for f in shuffled[n_test + n_val:]}

    print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"  Val pieces:  {sorted(val_files.keys())}")
    print(f"  Test pieces: {sorted(test_files.keys())}")

    return train_files, val_files, test_files


def copy_files(file_dict, dst_dir):
    """Copy files from source directories to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    source_dirs = {
        "curated": CURATED_DIR,
        "scraped": SCRAPED_DIR,
        "pittstate": PITTSTATE_DIR,
    }
    for fname, source in file_dict.items():
        src = os.path.join(source_dirs[source], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


def main():
    print("=" * 60)
    print("STEP 2: TRAIN/VAL/TEST SPLIT")
    print("=" * 60)

    # Split each source
    c_train, c_val, c_test = split_curated()
    s_train, s_val, s_test = split_scraped()
    p_train, p_val, p_test = split_pittstate()

    # Merge
    train = {**c_train, **s_train, **p_train}
    val = {**c_val, **s_val, **p_val}
    test = {**c_test, **s_test, **p_test}

    print(f"\nCombined: train={len(train)}, val={len(val)}, test={len(test)}")

    # Copy files
    for split_name, files, dst in [("train", train, TRAIN_DIR),
                                     ("val", val, VAL_DIR),
                                     ("test", test, TEST_DIR)]:
        copy_files(files, dst)
        actual = len([f for f in os.listdir(dst) if f.endswith(".mid")]) if os.path.isdir(dst) else 0
        print(f"  {split_name}: {actual} files copied to {dst}")

    # Save split manifest
    split_info = {
        "seed": SEED,
        "n_val_pieces": N_VAL_PIECES,
        "n_test_pieces": N_TEST_PIECES,
        "pittstate_n_val": PITTSTATE_N_VAL,
        "pittstate_n_test": PITTSTATE_N_TEST,
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

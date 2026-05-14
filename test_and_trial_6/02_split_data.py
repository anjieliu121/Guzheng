#!/usr/bin/env python3
"""
Step 2: Piece-level train/val/test split for Trial 6.

Critical: all transpositions of the same piece (e.g. shang_lou_A, shang_lou_C,
shang_lou_D, ...) must go into the SAME split. Otherwise we leak the melody.

We hold out 10 pieces for val and 10 for test (~16% of pieces total). With
~125 unique pieces this leaves ~105 pieces (~496 files) for training.

Outputs data/{train,val,test}/ and data/splits/split.json.
"""

import os
import json
import random
import shutil

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TRIAL_ROOT, "data")
CURATED_DIR = os.path.join(DATA_DIR, "curated")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SEED = 42
N_VAL_PIECES = 10
N_TEST_PIECES = 10
SCALE_SUFFIXES = ("_A", "_C", "_D", "_F", "_G")


def piece_base_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    for suf in SCALE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def main():
    print("=" * 60)
    print("STEP 2: PIECE-LEVEL TRAIN/VAL/TEST SPLIT (Trial 6)")
    print("=" * 60)

    if not os.path.isdir(CURATED_DIR):
        print(f"ERROR: run 01_prepare_data.py first ({CURATED_DIR} missing)")
        return

    files = sorted(f for f in os.listdir(CURATED_DIR) if f.endswith(".mid"))
    pieces = {}
    for f in files:
        pieces.setdefault(piece_base_name(f), []).append(f)

    piece_names = sorted(pieces.keys())
    print(f"Curated: {len(files)} files from {len(piece_names)} unique pieces")

    random.seed(SEED)
    shuffled = list(piece_names)
    random.shuffle(shuffled)

    test_pieces = sorted(shuffled[:N_TEST_PIECES])
    val_pieces = sorted(shuffled[N_TEST_PIECES:N_TEST_PIECES + N_VAL_PIECES])
    train_pieces = sorted(shuffled[N_TEST_PIECES + N_VAL_PIECES:])

    print(f"\nVal pieces  ({len(val_pieces)}):  {val_pieces}")
    print(f"Test pieces ({len(test_pieces)}): {test_pieces}")
    print(f"Train pieces: {len(train_pieces)} (not listed)")

    train_files = sorted(f for p in train_pieces for f in pieces[p])
    val_files = sorted(f for p in val_pieces for f in pieces[p])
    test_files = sorted(f for p in test_pieces for f in pieces[p])

    print(f"\nFile counts: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    for split, dst in [(train_files, TRAIN_DIR), (val_files, VAL_DIR), (test_files, TEST_DIR)]:
        os.makedirs(dst, exist_ok=True)
        # Clean any prior contents (no augmentation files yet)
        for old in os.listdir(dst):
            if old.endswith(".mid"):
                os.remove(os.path.join(dst, old))
        for f in split:
            shutil.copy2(os.path.join(CURATED_DIR, f), os.path.join(dst, f))

    os.makedirs(os.path.join(DATA_DIR, "splits"), exist_ok=True)
    with open(os.path.join(DATA_DIR, "splits", "split.json"), "w") as f:
        json.dump({
            "seed": SEED,
            "n_val_pieces": N_VAL_PIECES,
            "n_test_pieces": N_TEST_PIECES,
            "train_pieces": train_pieces,
            "val_pieces": val_pieces,
            "test_pieces": test_pieces,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "counts": {
                "train_files": len(train_files),
                "val_files": len(val_files),
                "test_files": len(test_files),
                "train_pieces": len(train_pieces),
                "val_pieces": len(val_pieces),
                "test_pieces": len(test_pieces),
            },
        }, f, indent=2)

    print(f"\nSplit manifest: data/splits/split.json")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1: Prepare training data for MIDI-RWKV state-tuning.

Collects and filters MIDI files from three sources:
- MIDI_transposed/ (curated, hand-validated, 100% pentatonic)
- raw_data_web_scraped/guzheng_tech99/ (scraped guzheng)
- raw_data_web_scraped/pittstate_chinese/ (classical Chinese repertoire)

Filters by pentatonic purity, note count, and guzheng range.
Copies valid files to data/train/ and data/val/ for MIDI-RWKV training.

The MIDI-RWKV train.py reads raw .mid files from a data directory and
tokenizes on-the-fly using the MMM tokenizer.
"""

import os
import json
import random
import shutil
import mido
import numpy as np

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRIAL_ROOT)

# Source directories
CURATED_SRC = os.path.join(REPO_ROOT, "MIDI_transposed")
SCRAPED_SRC = os.path.join(REPO_ROOT, "raw_data_web_scraped", "guzheng_tech99")
PITTSTATE_SRC = os.path.join(REPO_ROOT, "raw_data_web_scraped", "pittstate_chinese")

# Output directories
DATA_DIR = os.path.join(TRIAL_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},
    "G": {7, 9, 11, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "A": {9, 11, 1, 4, 6},
    "F": {5, 7, 9, 0, 2},
}
PRESSED_PCS = {
    "D": {7, 1}, "G": {0, 6}, "C": {5, 11}, "A": {2, 8}, "F": {10, 4},
}
GUZHENG_RANGE = (37, 86)

MIN_NOTES = 20
MIN_PURITY = 0.80

SEED = 42
# Hold out 2 curated pieces and 2 pittstate pieces for validation
N_VAL_CURATED_PIECES = 2
N_VAL_PITTSTATE = 2
# Hold out 2 curated pieces and 2 pittstate pieces for test
N_TEST_CURATED_PIECES = 2
N_TEST_PITTSTATE = 2


def extract_notes(midi_path):
    """Extract (pitch, onset_tick, duration_tick, velocity) from MIDI."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  ERROR reading {midi_path}: {e}")
        return []

    notes = []
    for track in mid.tracks:
        abs_time = 0
        pending = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.note, msg.channel)] = (abs_time, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in pending:
                    onset, vel = pending.pop(key)
                    dur = abs_time - onset
                    if dur > 0:
                        notes.append((msg.note, onset, dur, vel))
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes


def detect_scale_and_purity(notes):
    """Detect best-matching pentatonic scale and compute purity."""
    if not notes:
        return "?", 0.0

    pc_counts = np.zeros(12)
    for pitch, _, _, _ in notes:
        pc_counts[pitch % 12] += 1

    best_scale = "?"
    best_purity = 0.0
    for sname, pcs in PENTATONIC_SCALES.items():
        ext = pcs | PRESSED_PCS.get(sname, set())
        match = sum(pc_counts[pc] for pc in ext)
        ratio = match / max(sum(pc_counts), 1)
        if ratio > best_purity:
            best_purity = ratio
            best_scale = sname
    return best_scale, best_purity


def check_range(notes):
    """Check what fraction of notes are in guzheng range."""
    if not notes:
        return 0.0
    in_range = sum(1 for p, _, _, _ in notes if GUZHENG_RANGE[0] <= p <= GUZHENG_RANGE[1])
    return in_range / len(notes)


def piece_base_name(filename):
    """Extract base piece name, removing scale suffix (e.g. _D, _A)."""
    name = os.path.splitext(filename)[0]
    for suffix in ("_A", "_C", "_D", "_F", "_G"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def filter_files(src_dir, label, min_purity=MIN_PURITY, min_notes=MIN_NOTES):
    """Filter MIDI files by quality criteria. Returns list of (filename, metadata)."""
    if not os.path.isdir(src_dir):
        print(f"  Source directory not found: {src_dir}")
        return []

    midi_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".mid"))
    print(f"\n{'='*60}")
    print(f"Filtering {label}: {len(midi_files)} files from {src_dir}")
    print(f"{'='*60}")

    accepted = []
    rejected = 0

    for fname in midi_files:
        src_path = os.path.join(src_dir, fname)
        notes = extract_notes(src_path)
        n_notes = len(notes)

        if n_notes < min_notes:
            print(f"  REJECT {fname}: only {n_notes} notes (min {min_notes})")
            rejected += 1
            continue

        scale, purity = detect_scale_and_purity(notes)
        range_ratio = check_range(notes)

        if purity < min_purity:
            print(f"  REJECT {fname}: purity={purity:.2f} < {min_purity} (scale={scale})")
            rejected += 1
            continue

        if range_ratio < 0.5:
            print(f"  REJECT {fname}: only {range_ratio:.0%} notes in guzheng range")
            rejected += 1
            continue

        pitches = [n[0] for n in notes]
        entry = {
            "file": fname,
            "source": label,
            "n_notes": n_notes,
            "scale": scale,
            "purity": round(purity, 4),
            "range_ratio": round(range_ratio, 4),
            "pitch_range": [min(pitches), max(pitches)],
        }
        accepted.append((fname, entry))
        print(f"  OK     {fname}: {n_notes} notes, scale={scale}, purity={purity:.2f}")

    print(f"\n{label} summary: {len(accepted)} accepted, {rejected} rejected")
    return accepted


def split_and_copy(curated_files, scraped_files, pittstate_files):
    """Split into train/val/test and copy to output directories."""
    random.seed(SEED)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    train_manifest = []
    val_manifest = []
    test_manifest = []

    # --- Curated: split by piece (all transpositions of a piece go together) ---
    pieces = {}
    for fname, meta in curated_files:
        base = piece_base_name(fname)
        pieces.setdefault(base, []).append((fname, meta))

    piece_names = sorted(pieces.keys())
    random.shuffle(piece_names)

    test_pieces = set(piece_names[:N_TEST_CURATED_PIECES])
    val_pieces = set(piece_names[N_TEST_CURATED_PIECES:N_TEST_CURATED_PIECES + N_VAL_CURATED_PIECES])
    train_pieces = set(piece_names[N_TEST_CURATED_PIECES + N_VAL_CURATED_PIECES:])

    for p in train_pieces:
        for fname, meta in pieces[p]:
            shutil.copy2(os.path.join(CURATED_SRC, fname), os.path.join(TRAIN_DIR, fname))
            train_manifest.append(meta)
    for p in val_pieces:
        for fname, meta in pieces[p]:
            shutil.copy2(os.path.join(CURATED_SRC, fname), os.path.join(VAL_DIR, fname))
            val_manifest.append(meta)
    for p in test_pieces:
        for fname, meta in pieces[p]:
            shutil.copy2(os.path.join(CURATED_SRC, fname), os.path.join(TEST_DIR, fname))
            test_manifest.append(meta)

    print(f"\nCurated split: train={len(train_pieces)} pieces, "
          f"val={sorted(val_pieces)}, test={sorted(test_pieces)}")

    # --- Scraped: use existing train/val/test naming from tech99 ---
    for fname, meta in scraped_files:
        if "train" in fname:
            shutil.copy2(os.path.join(SCRAPED_SRC, fname), os.path.join(TRAIN_DIR, fname))
            train_manifest.append(meta)
        elif "validation" in fname:
            shutil.copy2(os.path.join(SCRAPED_SRC, fname), os.path.join(VAL_DIR, fname))
            val_manifest.append(meta)
        elif "test" in fname:
            shutil.copy2(os.path.join(SCRAPED_SRC, fname), os.path.join(TEST_DIR, fname))
            test_manifest.append(meta)
        else:
            shutil.copy2(os.path.join(SCRAPED_SRC, fname), os.path.join(TRAIN_DIR, fname))
            train_manifest.append(meta)

    # --- Pittstate: hold out some for val/test ---
    random.seed(SEED + 1)
    pitt_shuffled = list(pittstate_files)
    random.shuffle(pitt_shuffled)

    n_test = min(N_TEST_PITTSTATE, len(pitt_shuffled))
    n_val = min(N_VAL_PITTSTATE, len(pitt_shuffled) - n_test)

    for fname, meta in pitt_shuffled[:n_test]:
        shutil.copy2(os.path.join(PITTSTATE_SRC, fname), os.path.join(TEST_DIR, fname))
        test_manifest.append(meta)
    for fname, meta in pitt_shuffled[n_test:n_test + n_val]:
        shutil.copy2(os.path.join(PITTSTATE_SRC, fname), os.path.join(VAL_DIR, fname))
        val_manifest.append(meta)
    for fname, meta in pitt_shuffled[n_test + n_val:]:
        shutil.copy2(os.path.join(PITTSTATE_SRC, fname), os.path.join(TRAIN_DIR, fname))
        train_manifest.append(meta)

    return train_manifest, val_manifest, test_manifest


def main():
    print("=" * 60)
    print("STEP 1: DATA PREPARATION FOR MIDI-RWKV STATE-TUNING")
    print("=" * 60)

    # Filter each source
    curated_files = filter_files(CURATED_SRC, "curated", min_purity=0.95, min_notes=10)
    scraped_files = filter_files(SCRAPED_SRC, "scraped", min_purity=0.80, min_notes=20)
    pittstate_files = filter_files(PITTSTATE_SRC, "pittstate", min_purity=0.80, min_notes=20)

    # Split and copy
    train_m, val_m, test_m = split_and_copy(curated_files, scraped_files, pittstate_files)

    # Save manifest
    manifest = {
        "total_files": len(train_m) + len(val_m) + len(test_m),
        "train_files": len(train_m),
        "val_files": len(val_m),
        "test_files": len(test_m),
        "sources": {
            "curated": len(curated_files),
            "scraped": len(scraped_files),
            "pittstate": len(pittstate_files),
        },
        "train": train_m,
        "val": val_m,
        "test": test_m,
    }
    manifest_path = os.path.join(DATA_DIR, "data_manifest.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Curated accepted: {len(curated_files)}")
    print(f"Scraped accepted: {len(scraped_files)}")
    print(f"Pittstate accepted: {len(pittstate_files)}")
    print(f"Total accepted: {len(curated_files) + len(scraped_files) + len(pittstate_files)}")
    print(f"  Train: {len(train_m)} files")
    print(f"  Val:   {len(val_m)} files")
    print(f"  Test:  {len(test_m)} files")
    print(f"Manifest saved to: {manifest_path}")

    # Verify output
    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if os.path.isdir(split_dir):
            n = len([f for f in os.listdir(split_dir) if f.endswith(".mid")])
            print(f"  {split_name}/: {n} .mid files")


if __name__ == "__main__":
    main()

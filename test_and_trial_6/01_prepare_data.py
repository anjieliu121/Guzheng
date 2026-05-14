#!/usr/bin/env python3
"""
Step 1: Prepare and clean MIDI data for Trial 6.

Source: MIDI_transposed/ — 590 files, 125 unique pieces, each transposed to up
to 5 pentatonic scales. Curated and hand-validated.

Filters:
- Min 20 notes per file
- Pentatonic purity >= 95% (curated, so most pass trivially)
- >= 50% of notes in guzheng range (MIDI 37-86)

Output:
- data/curated/ : copies of all accepted files
- data/data_manifest.json : statistics for every accepted file
"""

import os
import json
import shutil

import mido
import numpy as np

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRIAL_ROOT)

CURATED_SRC = os.path.join(REPO_ROOT, "MIDI_transposed")
CURATED_DST = os.path.join(TRIAL_ROOT, "data", "curated")

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
MIN_PURITY = 0.95
MIN_RANGE = 0.5


def extract_notes(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  ERROR reading {midi_path}: {e}")
        return []
    notes = []
    for track in mid.tracks:
        abs_t = 0
        pending = {}
        for msg in track:
            abs_t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.note, msg.channel)] = (abs_t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in pending:
                    onset, vel = pending.pop(key)
                    if abs_t - onset > 0:
                        notes.append((msg.note, onset, abs_t - onset, vel))
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes


def detect_scale_and_purity(notes):
    if not notes:
        return "?", 0.0
    pc_counts = np.zeros(12)
    for p, _, _, _ in notes:
        pc_counts[p % 12] += 1
    best_scale, best_purity = "?", 0.0
    for sname, pcs in PENTATONIC_SCALES.items():
        ext = pcs | PRESSED_PCS.get(sname, set())
        ratio = sum(pc_counts[pc] for pc in ext) / max(pc_counts.sum(), 1)
        if ratio > best_purity:
            best_purity, best_scale = ratio, sname
    return best_scale, best_purity


def check_range(notes):
    if not notes:
        return 0.0
    return sum(1 for p, _, _, _ in notes if GUZHENG_RANGE[0] <= p <= GUZHENG_RANGE[1]) / len(notes)


def main():
    print("=" * 60)
    print("STEP 1: DATA PREPARATION (Trial 6)")
    print("=" * 60)

    if not os.path.isdir(CURATED_SRC):
        print(f"ERROR: source dir not found: {CURATED_SRC}")
        return

    os.makedirs(CURATED_DST, exist_ok=True)

    midi_files = sorted(f for f in os.listdir(CURATED_SRC) if f.endswith(".mid"))
    print(f"Found {len(midi_files)} files in {CURATED_SRC}")

    manifest = []
    accepted = rejected = 0

    for fname in midi_files:
        src = os.path.join(CURATED_SRC, fname)
        notes = extract_notes(src)
        n_notes = len(notes)

        if n_notes < MIN_NOTES:
            print(f"  REJECT {fname}: only {n_notes} notes")
            rejected += 1
            continue

        scale, purity = detect_scale_and_purity(notes)
        rng_ratio = check_range(notes)

        if purity < MIN_PURITY:
            print(f"  REJECT {fname}: purity={purity:.2f} < {MIN_PURITY}")
            rejected += 1
            continue
        if rng_ratio < MIN_RANGE:
            print(f"  REJECT {fname}: only {rng_ratio:.0%} in guzheng range")
            rejected += 1
            continue

        pitches = [n[0] for n in notes]
        shutil.copy2(src, os.path.join(CURATED_DST, fname))
        accepted += 1
        manifest.append({
            "file": fname,
            "n_notes": n_notes,
            "scale": scale,
            "purity": round(purity, 4),
            "range_ratio": round(rng_ratio, 4),
            "pitch_range": [min(pitches), max(pitches)],
        })

    manifest_path = os.path.join(TRIAL_ROOT, "data", "data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"total_files": accepted, "files": manifest}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DATA PREPARATION COMPLETE: {accepted} accepted, {rejected} rejected")
    print("=" * 60)
    if manifest:
        purities = [e["purity"] for e in manifest]
        n_notes_list = [e["n_notes"] for e in manifest]
        print(f"Purity: mean={np.mean(purities):.3f} min={min(purities):.3f}")
        print(f"Notes:  mean={np.mean(n_notes_list):.0f} range=[{min(n_notes_list)}, {max(n_notes_list)}]")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

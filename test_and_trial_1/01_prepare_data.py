#!/usr/bin/env python3
"""
Step 1: Prepare and clean all MIDI data.

- Reads curated MIDI_transposed/ files (already validated, 100% pentatonic)
- Reads scraped guzheng_tech99/ files, filters by pentatonic purity and note count
- Copies valid files to data/curated/ and data/scraped/
- Generates a data manifest with statistics
"""

import os
import json
import shutil
import mido
import numpy as np

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRIAL_ROOT)

CURATED_SRC = os.path.join(REPO_ROOT, "MIDI_transposed")
SCRAPED_SRC = os.path.join(REPO_ROOT, "raw_data_web_scraped", "guzheng_tech99")

CURATED_DST = os.path.join(TRIAL_ROOT, "data", "curated")
SCRAPED_DST = os.path.join(TRIAL_ROOT, "data", "scraped")

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


def process_directory(src_dir, dst_dir, label, min_purity=MIN_PURITY, min_notes=MIN_NOTES):
    """Process a directory of MIDI files."""
    os.makedirs(dst_dir, exist_ok=True)
    manifest = []

    if not os.path.isdir(src_dir):
        print(f"  Source directory not found: {src_dir}")
        return manifest

    midi_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".mid"))
    print(f"\n{'='*60}")
    print(f"Processing {label}: {len(midi_files)} files from {src_dir}")
    print(f"{'='*60}")

    accepted = 0
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

        # Accept
        pitches = [n[0] for n in notes]
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(src_path, dst_path)
        accepted += 1

        entry = {
            "file": fname,
            "source": label,
            "n_notes": n_notes,
            "scale": scale,
            "purity": round(purity, 4),
            "range_ratio": round(range_ratio, 4),
            "pitch_range": [min(pitches), max(pitches)],
        }
        manifest.append(entry)
        print(f"  OK     {fname}: {n_notes} notes, scale={scale}, purity={purity:.2f}")

    print(f"\n{label} summary: {accepted} accepted, {rejected} rejected")
    return manifest


def main():
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    all_manifest = []

    # Process curated data (already validated, use lower threshold)
    curated = process_directory(CURATED_SRC, CURATED_DST, "curated", min_purity=0.95, min_notes=10)
    all_manifest.extend(curated)

    # Process scraped data (need stricter filtering)
    scraped = process_directory(SCRAPED_SRC, SCRAPED_DST, "scraped", min_purity=0.80, min_notes=20)
    all_manifest.extend(scraped)

    # Save manifest
    manifest_path = os.path.join(TRIAL_ROOT, "data", "data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "total_files": len(all_manifest),
            "curated_files": len(curated),
            "scraped_files": len(scraped),
            "files": all_manifest,
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Curated files: {len(curated)}")
    print(f"Scraped files: {len(scraped)}")
    print(f"Total files: {len(all_manifest)}")
    print(f"Manifest saved to: {manifest_path}")

    if all_manifest:
        purities = [e["purity"] for e in all_manifest]
        note_counts = [e["n_notes"] for e in all_manifest]
        print(f"\nPurity: mean={np.mean(purities):.3f}, min={min(purities):.3f}")
        print(f"Notes: mean={np.mean(note_counts):.0f}, range=[{min(note_counts)}, {max(note_counts)}]")


if __name__ == "__main__":
    main()

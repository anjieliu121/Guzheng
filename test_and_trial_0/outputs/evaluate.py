#!/usr/bin/env python3
"""
Evaluate generated MIDI files using objective metrics.
Metrics: pitch class distribution, large leap rate (>12 semitones),
         note density (notes/second), mean note duration, pitch class entropy.

Run with any Python that has mido + numpy:
  python3 outputs/evaluate.py
"""

import os, json, math
import numpy as np
import mido

ROOT         = "/Users/anjie/Documents/MyGuzheng/Guzheng"
EVAL_DIRS    = {
    "moonbeam_pretrained"    : os.path.join(ROOT, "outputs/moonbeam_pretrained"),
    "moonbeam_finetuned"     : os.path.join(ROOT, "outputs/moonbeam_finetuned"),
    "moonbeam_finetuned_t95" : os.path.join(ROOT, "outputs/moonbeam_finetuned_t95"),
    "midirwkv_pretrained"    : os.path.join(ROOT, "outputs/midirwkv_pretrained"),
    "midirwkv_finetuned"     : os.path.join(ROOT, "outputs/midirwkv_finetuned"),
}
TRAIN_DIR   = os.path.join(ROOT, "MIDI_transposed")
OUT_JSON    = os.path.join(ROOT, "outputs/evaluation/metrics.json")
OUT_TXT     = os.path.join(ROOT, "outputs/evaluation/summary.txt")

# Guzheng pentatonic scales (MIDI note classes 0-11)
PENTATONIC_SETS = {
    "A": {9, 11, 0, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "D": {2, 4, 6, 9, 11},
    "F": {5, 7, 9, 0, 2},
    "G": {7, 9, 11, 2, 4},
}
ALL_PENTATONIC = set().union(*PENTATONIC_SETS.values())


def midi_to_notes(midi_path):
    """Return list of (pitch, onset_sec, duration_sec)."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  WARN: could not load {midi_path}: {e}")
        return []

    tempo = 500000  # default 120 BPM
    tpb   = mid.ticks_per_beat
    notes = []

    for track in mid.tracks:
        abs_tick = 0
        active   = {}   # note -> onset_tick
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type in ("note_on", "note_off"):
                vel = getattr(msg, "velocity", 0)
                if msg.type == "note_on" and vel > 0:
                    # If same note is already active, close it first (handles repeated note_on)
                    if msg.note in active:
                        onset_s = mido.tick2second(active[msg.note], tpb, tempo)
                        off_s   = mido.tick2second(abs_tick, tpb, tempo)
                        if off_s > onset_s:
                            notes.append((msg.note, onset_s, off_s - onset_s))
                    active[msg.note] = abs_tick
                else:
                    if msg.note in active:
                        onset_s = mido.tick2second(active.pop(msg.note), tpb, tempo)
                        off_s   = mido.tick2second(abs_tick, tpb, tempo)
                        if off_s > onset_s:
                            notes.append((msg.note, onset_s, off_s - onset_s))

    return notes


def compute_metrics(notes):
    """Compute all objective metrics from a note list."""
    if len(notes) < 2:
        return None

    pitches   = [n[0] for n in notes]
    onsets    = [n[1] for n in notes]
    durations = [n[2] for n in notes]

    # Pitch class distribution (12 bins, normalised)
    pc_counts = np.zeros(12)
    for p in pitches:
        pc_counts[p % 12] += 1
    pc_dist = pc_counts / pc_counts.sum()

    # Large leap rate (consecutive pitch intervals > 12 semitones)
    intervals  = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    large_leaps = sum(1 for iv in intervals if iv > 12)
    leap_rate   = large_leaps / max(len(intervals), 1)

    # Note density (notes per second)
    total_duration = max(onsets) - min(onsets) + durations[onsets.index(max(onsets))]
    density = len(notes) / max(total_duration, 0.001)

    # Mean duration
    mean_dur = float(np.mean(durations))

    # Pitch class entropy
    eps = 1e-9
    entropy = float(-np.sum(pc_dist * np.log2(pc_dist + eps)))

    # Pentatonic purity (fraction of notes in any pentatonic scale)
    penta_count = sum(1 for p in pitches if p % 12 in ALL_PENTATONIC)
    penta_purity = penta_count / len(pitches)

    # Mean pitch (rough tessitura check for guzheng range D4-A5 = 62-81)
    mean_pitch = float(np.mean(pitches))

    return {
        "n_notes"       : len(notes),
        "pc_distribution": pc_dist.tolist(),
        "large_leap_rate": round(leap_rate, 4),
        "note_density"  : round(density, 3),
        "mean_duration" : round(mean_dur, 4),
        "pc_entropy"    : round(entropy, 4),
        "penta_purity"  : round(penta_purity, 4),
        "mean_pitch"    : round(mean_pitch, 2),
    }


def evaluate_dir(label, midi_dir):
    files = sorted([
        os.path.join(midi_dir, f)
        for f in os.listdir(midi_dir)
        if f.endswith(".mid")
    ])
    if not files:
        print(f"  No MIDI files in {midi_dir}")
        return {}

    all_metrics = []
    for fp in files:
        notes = midi_to_notes(fp)
        m = compute_metrics(notes)
        if m is not None:
            m["file"] = os.path.basename(fp)
            all_metrics.append(m)

    if not all_metrics:
        return {}

    # Aggregate
    agg = {
        "n_files"        : len(all_metrics),
        "mean_n_notes"   : round(np.mean([m["n_notes"] for m in all_metrics]), 1),
        "mean_leap_rate" : round(np.mean([m["large_leap_rate"] for m in all_metrics]), 4),
        "mean_density"   : round(np.mean([m["note_density"] for m in all_metrics]), 3),
        "mean_duration"  : round(np.mean([m["mean_duration"] for m in all_metrics]), 4),
        "mean_pc_entropy": round(np.mean([m["pc_entropy"] for m in all_metrics]), 4),
        "mean_penta_purity": round(np.mean([m["penta_purity"] for m in all_metrics]), 4),
        "mean_pitch"     : round(np.mean([m["mean_pitch"] for m in all_metrics]), 2),
        "per_file"       : all_metrics,
    }
    return agg


def evaluate_training_data():
    files = [os.path.join(TRAIN_DIR, f)
             for f in os.listdir(TRAIN_DIR) if f.endswith(".mid")]
    all_m = [compute_metrics(midi_to_notes(f)) for f in files]
    all_m = [m for m in all_m if m is not None]
    if not all_m:
        return {}
    return {
        "n_files"        : len(all_m),
        "mean_n_notes"   : round(np.mean([m["n_notes"] for m in all_m]), 1),
        "mean_leap_rate" : round(np.mean([m["large_leap_rate"] for m in all_m]), 4),
        "mean_density"   : round(np.mean([m["note_density"] for m in all_m]), 3),
        "mean_duration"  : round(np.mean([m["mean_duration"] for m in all_m]), 4),
        "mean_pc_entropy": round(np.mean([m["pc_entropy"] for m in all_m]), 4),
        "mean_penta_purity": round(np.mean([m["penta_purity"] for m in all_m]), 4),
        "mean_pitch"     : round(np.mean([m["mean_pitch"] for m in all_m]), 2),
    }


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    results = {}

    print("Evaluating training data ...")
    results["training_data"] = evaluate_training_data()

    for label, midi_dir in EVAL_DIRS.items():
        if not os.path.isdir(midi_dir):
            print(f"Skipping {label} (directory not found)")
            continue
        print(f"Evaluating {label} ...")
        results[label] = evaluate_dir(label, midi_dir)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {OUT_JSON}")

    # Print summary table
    keys = ["mean_n_notes", "mean_leap_rate", "mean_density",
            "mean_duration", "mean_pc_entropy", "mean_penta_purity", "mean_pitch"]
    lines = []
    header = f"{'Variant':<25} " + " ".join(f"{k:<18}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))

    for variant, m in results.items():
        if not m:
            continue
        row = f"{variant:<25} " + " ".join(f"{m.get(k, 'N/A'):<18}" for k in keys)
        lines.append(row)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_TXT, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {OUT_TXT}")

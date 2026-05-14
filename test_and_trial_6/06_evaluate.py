#!/usr/bin/env python3
"""
Step 6: Evaluation for Trial 6.

Computes literature-standard metrics:

DISTRIBUTIONAL (Yang & Lerch 2018):
  - OA pitch class, OA duration, OA interval, OA IOI

STRUCTURAL (Pearce & Wiggins 2012, Wu & Yang 2020):
  - Compression ratio (gzip on token sequence)
  - Structureness Indicator (max self-similarity over off-diagonals)
  - 2nd-order pitch transition entropy
  - Groove consistency

Plus the basic per-piece statistics from Trial 1 (note density, pentatonic
purity, pc entropy, etc.).

Reference baselines: training set (= 1.0 for OA) and held-out test set
(natural ceiling for unseen real guzheng).
"""

import os
import sys
import json
import gzip
import math
from collections import Counter

import numpy as np
import mido

from config import trial_root
from scales import PENTATONIC_SCALES, PRESSED_PCS

# Reuse the structural metric implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musicality_metrics import (
    compression_ratio,
    structureness_indicator,
    pitch_transition_entropy,
    groove_consistency,
)


# --------------------------- MIDI loading -------------------------------

def midi_to_notes(midi_path):
    """Return list of (pitch, onset_sec, duration_sec, velocity)."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []
    tempo = 500000
    tpb = mid.ticks_per_beat
    # Pre-pass for tempo
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        else:
            continue
        break

    def t2s(t):
        return mido.tick2second(t, tpb, tempo)

    notes = []
    for tr in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in active:
                    on_tick, on_vel = active[msg.note]
                    if abs_tick > on_tick:
                        notes.append((msg.note, t2s(on_tick), t2s(abs_tick) - t2s(on_tick), on_vel))
                active[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    on_tick, on_vel = active.pop(msg.note)
                    if abs_tick > on_tick:
                        notes.append((msg.note, t2s(on_tick), t2s(abs_tick) - t2s(on_tick), on_vel))
    return notes


def notes_to_struct(notes):
    """Convert (pitch, onset, dur, vel) tuples to (onset, dur, pitch) tuples
    used by musicality_metrics functions."""
    return [(o, d, p) for (p, o, d, _v) in sorted(notes, key=lambda n: (n[1], n[0]))]


# ----------------------------- Per-file metrics --------------------------

def compute_distributions(notes):
    if len(notes) < 2:
        return None
    pitches = [n[0] for n in notes]
    durations = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    onsets = sorted([n[1] for n in notes])
    sorted_notes = sorted(notes, key=lambda x: x[1])
    intervals = [sorted_notes[i + 1][0] - sorted_notes[i][0]
                 for i in range(len(sorted_notes) - 1)]
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)
            if onsets[i + 1] > onsets[i]]

    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[p % 12] += 1
    pc_hist_norm = pc_hist / max(pc_hist.sum(), 1)

    dur_bins = np.arange(0, 5.05, 0.1)
    dur_hist, _ = np.histogram(durations, bins=dur_bins, density=True)
    int_bins = np.arange(-24.5, 25.5, 1)
    int_hist, _ = np.histogram(intervals, bins=int_bins, density=True)
    ioi_bins = np.arange(0, 2.05, 0.05)
    ioi_hist, _ = np.histogram(iois, bins=ioi_bins, density=True)

    total_dur = max(onsets[-1] + durations[-1], 0.001) - onsets[0]

    best_purity, best_scale = 0.0, "?"
    for sname, pcs in PENTATONIC_SCALES.items():
        ext = pcs | PRESSED_PCS.get(sname, set())
        ratio = sum(pc_hist[pc] for pc in ext) / max(pc_hist.sum(), 1)
        if ratio > best_purity:
            best_purity, best_scale = ratio, sname

    pc_entropy = float(-np.sum(pc_hist_norm * np.log2(pc_hist_norm + 1e-9)))

    # Structural metrics
    struct = notes_to_struct(notes)
    cr = compression_ratio(struct)
    si = structureness_indicator(struct)
    h2 = pitch_transition_entropy(struct)
    gc = groove_consistency(struct)

    return {
        "n_notes": len(notes),
        "total_duration": round(total_dur, 1),
        "pitch_range": (min(pitches), max(pitches)),
        "mean_pitch": round(np.mean(pitches), 1),
        "pc_hist": pc_hist_norm.tolist(),
        "dur_hist": dur_hist.tolist(),
        "int_hist": int_hist.tolist(),
        "ioi_hist": ioi_hist.tolist(),
        "mean_duration": round(float(np.mean(durations)), 4),
        "mean_velocity": round(float(np.mean(velocities)), 1),
        "note_density": round(len(notes) / max(total_dur, 0.001), 2),
        "pc_entropy": round(pc_entropy, 3),
        "penta_purity": round(best_purity, 4),
        "best_scale": best_scale,
        "mean_interval": round(float(np.mean([abs(i) for i in intervals])), 2) if intervals else 0,
        # structural
        "compression_ratio": round(cr, 3),
        "structureness": round(si, 3),
        "trans_entropy_2nd": round(h2, 3),
        "groove_consistency": round(gc, 3),
    }


# ---------------------------- Aggregation --------------------------------

def overlapping_area(h1, h2):
    a, b = np.array(h1), np.array(h2)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return float(np.sum(np.minimum(a, b)))


def evaluate_directory(midi_dir, exclude_aug=False):
    if not os.path.isdir(midi_dir):
        return []
    files = sorted(os.path.join(midi_dir, f) for f in os.listdir(midi_dir)
                   if f.endswith(".mid")
                   and (not exclude_aug or not f.startswith("aug_")))
    out = []
    for fp in files:
        notes = midi_to_notes(fp)
        d = compute_distributions(notes)
        if d:
            d["file"] = os.path.basename(fp)
            out.append(d)
    return out


def aggregate_metrics(dists):
    if not dists:
        return {}
    keys_avg = ["n_notes", "mean_duration", "note_density", "mean_velocity",
                "pc_entropy", "penta_purity", "mean_interval",
                "compression_ratio", "structureness", "trans_entropy_2nd",
                "groove_consistency", "mean_pitch"]
    out = {"n_files": len(dists)}
    for k in keys_avg:
        out["mean_" + k] = round(float(np.mean([d[k] for d in dists])), 4)
    out["agg_pc_hist"] = np.mean([d["pc_hist"] for d in dists], axis=0).tolist()
    out["agg_dur_hist"] = np.mean([d["dur_hist"] for d in dists], axis=0).tolist()
    out["agg_int_hist"] = np.mean([d["int_hist"] for d in dists], axis=0).tolist()
    out["agg_ioi_hist"] = np.mean([d["ioi_hist"] for d in dists], axis=0).tolist()
    return out


def oa_against(ref, agg):
    if not ref or not agg:
        return {}
    return {
        "OA_pitch_class": round(overlapping_area(ref["agg_pc_hist"], agg["agg_pc_hist"]), 4),
        "OA_duration": round(overlapping_area(ref["agg_dur_hist"], agg["agg_dur_hist"]), 4),
        "OA_interval": round(overlapping_area(ref["agg_int_hist"], agg["agg_int_hist"]), 4),
        "OA_ioi": round(overlapping_area(ref["agg_ioi_hist"], agg["agg_ioi_hist"]), 4),
    }


# ---------------------------- Main driver -------------------------------

def main():
    root = trial_root()
    eval_dir = os.path.join(root, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # ---- Reference: training (originals only, exclude augmented) ----
    train_dir = os.path.join(root, "data", "train")
    print("Evaluating training data baseline (originals only)...")
    train_dists = evaluate_directory(train_dir, exclude_aug=True)
    train_agg = aggregate_metrics(train_dists)
    print(f"  {train_agg.get('n_files', 0)} files")

    # ---- Held-out test set ----
    test_dir = os.path.join(root, "data", "test")
    print("Evaluating held-out test data...")
    test_dists = evaluate_directory(test_dir)
    test_agg = aggregate_metrics(test_dists)
    test_oa = oa_against(train_agg, test_agg) if test_agg else {}
    print(f"  {test_agg.get('n_files', 0)} files, OA_pc={test_oa.get('OA_pitch_class', 'N/A')}")

    # ---- Generated variants ----
    gen_dir = os.path.join(root, "generated")
    variants = {}
    for variant in ("constrained", "unconstrained"):
        var_dir = os.path.join(gen_dir, variant)
        if not os.path.isdir(var_dir):
            continue
        print(f"Evaluating {variant} generation...")
        dists = evaluate_directory(var_dir)
        agg = aggregate_metrics(dists)
        oa = oa_against(train_agg, agg)
        variants[variant] = {"aggregate": agg, "oa": oa, "per_file": dists}
        print(f"  {agg['n_files']} files, OA_pc={oa.get('OA_pitch_class','N/A')}, "
              f"CR={agg.get('mean_compression_ratio')}, SI={agg.get('mean_structureness')}, "
              f"H2={agg.get('mean_trans_entropy_2nd')}")

    # ---- Save JSON ----
    results = {
        "training": train_agg,
        "test": test_agg,
        "test_oa_vs_train": test_oa,
        "variants": {n: {"aggregate": v["aggregate"], "oa": v["oa"]} for n, v in variants.items()},
    }
    json_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        # strip the per-file histograms to keep file size sane
        clean = json.loads(json.dumps(results))
        for d in [clean.get("training", {}), clean.get("test", {})]:
            for k in list(d.keys()):
                if k.startswith("agg_"):
                    d.pop(k)
        for v in clean.get("variants", {}).values():
            for k in list(v.get("aggregate", {}).keys()):
                if k.startswith("agg_"):
                    v["aggregate"].pop(k)
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 110)
    print(f"{'Variant':<30} {'Files':>5} {'Notes':>6} {'Penta':>6} {'Dens':>5} "
          f"{'OA_PC':>7} {'OA_Dur':>7} {'CR':>7} {'SI':>6} {'H2':>6} {'GC':>6}")
    print("-" * 110)

    def row(name, agg, oa):
        cr = agg.get("mean_compression_ratio", 0)
        si = agg.get("mean_structureness", 0)
        h2 = agg.get("mean_trans_entropy_2nd", 0)
        gc = agg.get("mean_groove_consistency", 0)
        return (f"{name:<30} {agg.get('n_files', 0):>5} "
                f"{agg.get('mean_n_notes', 0):>6.0f} "
                f"{agg.get('mean_penta_purity', 0):>6.3f} "
                f"{agg.get('mean_note_density', 0):>5.2f} "
                f"{oa.get('OA_pitch_class', 1.0):>7.3f} "
                f"{oa.get('OA_duration', 1.0):>7.3f} "
                f"{cr:>7.3f} {si:>6.3f} {h2:>6.3f} {gc:>6.3f}")

    print(row("training (reference)", train_agg, {"OA_pitch_class": 1.0, "OA_duration": 1.0}))
    if test_agg:
        print(row("test (held-out real)", test_agg, test_oa))
    for name, v in variants.items():
        print(row(name, v["aggregate"], v["oa"]))
    print("=" * 110)
    print("CR = compression ratio (higher = more structural repetition)")
    print("SI = structureness indicator [0..1]")
    print("H2 = 2nd-order pitch transition entropy (closer to training = better)")
    print("GC = groove consistency [0..1]")


if __name__ == "__main__":
    main()

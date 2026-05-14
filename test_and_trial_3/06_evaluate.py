#!/usr/bin/env python3
"""
Step 6: Evaluate generated MIDI files.

Metrics:
- Pitch class distribution overlap (OA) vs training data
- Duration distribution overlap
- Interval distribution overlap
- IOI distribution overlap
- Pentatonic purity
- Note density, pitch range, polyphony
- Self-repetition: measures how repetitive each generated piece is internally
- Comparison: constrained vs unconstrained
"""

import os
import json
import math
import numpy as np
import mido
from collections import Counter

from config import trial_root
from scales import PENTATONIC_SCALES, PRESSED_PCS


def midi_to_notes(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []
    tempo = 500000
    tpb = mid.ticks_per_beat
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type in ("note_on", "note_off"):
                vel = getattr(msg, "velocity", 0)
                if msg.type == "note_on" and vel > 0:
                    if msg.note in active:
                        onset_s = mido.tick2second(active[msg.note][0], tpb, tempo)
                        off_s = mido.tick2second(abs_tick, tpb, tempo)
                        if off_s > onset_s:
                            notes.append((msg.note, onset_s, off_s - onset_s, active[msg.note][1]))
                    active[msg.note] = (abs_tick, vel)
                else:
                    if msg.note in active:
                        onset_tick, on_vel = active.pop(msg.note)
                        onset_s = mido.tick2second(onset_tick, tpb, tempo)
                        off_s = mido.tick2second(abs_tick, tpb, tempo)
                        if off_s > onset_s:
                            notes.append((msg.note, onset_s, off_s - onset_s, on_vel))
    return notes


def compute_self_repetition(pitches, window_sizes=(4, 8, 12)):
    """Measure how repetitive a pitch sequence is internally.

    For each window size, counts what fraction of n-grams appear more than once.
    High values indicate excessive repetition within the piece.
    """
    results = {}
    for n in window_sizes:
        if len(pitches) < n + 1:
            results[f"self_rep_{n}gram"] = 0.0
            continue
        ngrams = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for ng in ngrams if counts[ng] > 1)
        results[f"self_rep_{n}gram"] = round(repeated / len(ngrams), 4)
    return results


def compute_distributions(notes):
    if len(notes) < 2:
        return None
    pitches = [n[0] for n in notes]
    durations = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    onsets = sorted([n[1] for n in notes])
    sorted_notes = sorted(notes, key=lambda x: x[1])
    intervals = [sorted_notes[i+1][0] - sorted_notes[i][0] for i in range(len(sorted_notes)-1)]
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1) if onsets[i+1] > onsets[i]]

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

    best_purity = 0
    best_scale = "?"
    for sname, pcs in PENTATONIC_SCALES.items():
        ext = pcs | PRESSED_PCS.get(sname, set())
        match = sum(pc_hist[pc] for pc in ext)
        ratio = match / max(sum(pc_hist), 1)
        if ratio > best_purity:
            best_purity = ratio
            best_scale = sname

    eps = 1e-9
    pc_entropy = float(-np.sum(pc_hist_norm * np.log2(pc_hist_norm + eps)))

    events = []
    for p, on, dur, v in notes:
        events.append((on, 1))
        events.append((on + dur, -1))
    events.sort()
    max_simul = cur = 0
    for _, delta in events:
        cur += delta
        max_simul = max(max_simul, cur)

    # Self-repetition metrics
    self_rep = compute_self_repetition(pitches)

    result = {
        "n_notes": len(notes),
        "total_duration": round(total_dur, 1),
        "pitch_range": (min(pitches), max(pitches)),
        "mean_pitch": round(np.mean(pitches), 1),
        "pc_hist": pc_hist_norm.tolist(),
        "dur_hist": dur_hist.tolist(),
        "int_hist": int_hist.tolist(),
        "ioi_hist": ioi_hist.tolist(),
        "mean_duration": round(np.mean(durations), 4),
        "mean_velocity": round(np.mean(velocities), 1),
        "note_density": round(len(notes) / max(total_dur, 0.001), 2),
        "pc_entropy": round(pc_entropy, 3),
        "penta_purity": round(best_purity, 4),
        "best_scale": best_scale,
        "large_leap_rate": round(sum(1 for i in intervals if abs(i) > 12) / max(len(intervals), 1), 4),
        "mean_interval": round(np.mean([abs(i) for i in intervals]), 2) if intervals else 0,
        "max_simultaneous": max_simul,
    }
    result.update(self_rep)
    return result


def overlapping_area(hist1, hist2):
    h1 = np.array(hist1)
    h2 = np.array(hist2)
    min_len = min(len(h1), len(h2))
    h1, h2 = h1[:min_len], h2[:min_len]
    s1, s2 = h1.sum(), h2.sum()
    if s1 > 0:
        h1 = h1 / s1
    if s2 > 0:
        h2 = h2 / s2
    return float(np.sum(np.minimum(h1, h2)))


def evaluate_directory(midi_dir):
    if not os.path.isdir(midi_dir):
        return []
    files = sorted(os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(".mid"))
    dists = []
    for fp in files:
        notes = midi_to_notes(fp)
        d = compute_distributions(notes)
        if d:
            d["file"] = os.path.basename(fp)
            dists.append(d)
    return dists


def aggregate_metrics(dists):
    if not dists:
        return {}
    result = {
        "n_files": len(dists),
        "mean_n_notes": round(np.mean([d["n_notes"] for d in dists]), 1),
        "mean_duration": round(np.mean([d["mean_duration"] for d in dists]), 4),
        "mean_density": round(np.mean([d["note_density"] for d in dists]), 2),
        "mean_velocity": round(np.mean([d["mean_velocity"] for d in dists]), 1),
        "mean_pc_entropy": round(np.mean([d["pc_entropy"] for d in dists]), 3),
        "mean_penta_purity": round(np.mean([d["penta_purity"] for d in dists]), 4),
        "mean_large_leap_rate": round(np.mean([d["large_leap_rate"] for d in dists]), 4),
        "mean_interval": round(np.mean([d["mean_interval"] for d in dists]), 2),
        "mean_max_simul": round(np.mean([d["max_simultaneous"] for d in dists]), 1),
        "mean_pitch": round(np.mean([d["mean_pitch"] for d in dists]), 1),
        "agg_pc_hist": np.mean([d["pc_hist"] for d in dists], axis=0).tolist(),
        "agg_dur_hist": np.mean([d["dur_hist"] for d in dists], axis=0).tolist(),
        "agg_int_hist": np.mean([d["int_hist"] for d in dists], axis=0).tolist(),
        "agg_ioi_hist": np.mean([d["ioi_hist"] for d in dists], axis=0).tolist(),
    }
    # Self-repetition aggregates
    for key in ("self_rep_4gram", "self_rep_8gram", "self_rep_12gram"):
        vals = [d[key] for d in dists if key in d]
        if vals:
            result[f"mean_{key}"] = round(np.mean(vals), 4)
    return result


def main():
    root = trial_root()
    eval_dir = os.path.join(root, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate training data baseline
    train_dir = os.path.join(root, "data", "train")
    print("Evaluating training data baseline...")
    # Only use original files (not augmented) for baseline
    train_dists = []
    if os.path.isdir(train_dir):
        for f in sorted(os.listdir(train_dir)):
            if f.endswith(".mid") and not f.startswith("aug_"):
                notes = midi_to_notes(os.path.join(train_dir, f))
                d = compute_distributions(notes)
                if d:
                    d["file"] = f
                    train_dists.append(d)
    train_agg = aggregate_metrics(train_dists)
    print(f"  {train_agg.get('n_files', 0)} files")

    # Evaluate test data
    test_dir = os.path.join(root, "data", "test")
    print("Evaluating test data...")
    test_dists = evaluate_directory(test_dir)
    test_agg = aggregate_metrics(test_dists)
    print(f"  {test_agg.get('n_files', 0)} files")

    # Evaluate generated variants
    gen_dir = os.path.join(root, "generated")
    variants = {}

    constrained_dir = os.path.join(gen_dir, "constrained")
    if os.path.isdir(constrained_dir):
        print("Evaluating constrained generation...")
        dists = evaluate_directory(constrained_dir)
        agg = aggregate_metrics(dists)
        oa = {
            "OA_pitch_class": round(overlapping_area(train_agg["agg_pc_hist"], agg["agg_pc_hist"]), 4),
            "OA_duration": round(overlapping_area(train_agg["agg_dur_hist"], agg["agg_dur_hist"]), 4),
            "OA_interval": round(overlapping_area(train_agg["agg_int_hist"], agg["agg_int_hist"]), 4),
            "OA_ioi": round(overlapping_area(train_agg["agg_ioi_hist"], agg["agg_ioi_hist"]), 4),
        } if train_agg else {}
        variants["constrained"] = {"aggregate": agg, "oa": oa, "per_file": dists}
        print(f"  {agg['n_files']} files, purity={agg['mean_penta_purity']}, OA_pc={oa.get('OA_pitch_class', 'N/A')}")

    unconstrained_dir = os.path.join(gen_dir, "unconstrained")
    if os.path.isdir(unconstrained_dir):
        print("Evaluating unconstrained generation...")
        dists = evaluate_directory(unconstrained_dir)
        agg = aggregate_metrics(dists)
        oa = {
            "OA_pitch_class": round(overlapping_area(train_agg["agg_pc_hist"], agg["agg_pc_hist"]), 4),
            "OA_duration": round(overlapping_area(train_agg["agg_dur_hist"], agg["agg_dur_hist"]), 4),
            "OA_interval": round(overlapping_area(train_agg["agg_int_hist"], agg["agg_int_hist"]), 4),
            "OA_ioi": round(overlapping_area(train_agg["agg_ioi_hist"], agg["agg_ioi_hist"]), 4),
        } if train_agg else {}
        variants["unconstrained"] = {"aggregate": agg, "oa": oa, "per_file": dists}
        print(f"  {agg['n_files']} files, purity={agg['mean_penta_purity']}, OA_pc={oa.get('OA_pitch_class', 'N/A')}")

    # Save results
    results = {
        "training_data": train_agg,
        "test_data": test_agg,
        "variants": {n: {"aggregate": v["aggregate"], "oa": v["oa"]} for n, v in variants.items()},
    }
    json_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    lines = ["# Evaluation Report", ""]
    lines.append("## Training Data Baseline")
    for k, v in train_agg.items():
        if not k.startswith("agg_"):
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    if test_agg:
        lines.append("## Test Data")
        for k, v in test_agg.items():
            if not k.startswith("agg_"):
                lines.append(f"- **{k}:** {v}")
        lines.append("")

    if variants:
        lines.append("## Generated Variants")
        lines.append("")
        header = "| Metric | Training |"
        sep = "|--------|----------|"
        for name in variants:
            header += f" {name} |"
            sep += "----------|"
        lines.append(header)
        lines.append(sep)

        metrics_keys = ["mean_n_notes", "mean_density", "mean_duration", "mean_velocity",
                        "mean_pc_entropy", "mean_penta_purity", "mean_large_leap_rate",
                        "mean_interval", "mean_max_simul", "mean_pitch",
                        "mean_self_rep_4gram", "mean_self_rep_8gram", "mean_self_rep_12gram"]
        for m in metrics_keys:
            row = f"| {m} | {train_agg.get(m, 'N/A')} |"
            for name, v in variants.items():
                row += f" {v['aggregate'].get(m, 'N/A')} |"
            lines.append(row)
        lines.append("")

        lines.append("## OA Metrics (1.0 = identical to training)")
        lines.append("")
        oa_header = "| Distribution |"
        oa_sep = "|-------------|"
        for name in variants:
            oa_header += f" {name} |"
            oa_sep += "----------|"
        lines.append(oa_header)
        lines.append(oa_sep)
        for oa_key in ["OA_pitch_class", "OA_duration", "OA_interval", "OA_ioi"]:
            row = f"| {oa_key} |"
            for name, v in variants.items():
                row += f" {v['oa'].get(oa_key, 'N/A')} |"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(eval_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {json_path}")
    print(f"Report saved to: {report_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Variant':<20} {'Files':<6} {'Penta%':<8} {'Density':<8} {'OA_PC':<8} {'OA_Dur':<8} {'Rep4g':<8}")
    print("-" * 66)
    if train_agg:
        rep4 = train_agg.get('mean_self_rep_4gram', 'N/A')
        print(f"{'training':<20} {train_agg['n_files']:<6} {train_agg['mean_penta_purity']:<8} {train_agg['mean_density']:<8} {'1.0000':<8} {'1.0000':<8} {rep4:<8}")
    for name, v in variants.items():
        a, o = v["aggregate"], v["oa"]
        rep4 = a.get('mean_self_rep_4gram', 'N/A')
        print(f"{name:<20} {a['n_files']:<6} {a['mean_penta_purity']:<8} {a['mean_density']:<8} {o.get('OA_pitch_class','N/A'):<8} {o.get('OA_duration','N/A'):<8} {rep4:<8}")


if __name__ == "__main__":
    main()

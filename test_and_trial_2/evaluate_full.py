#!/usr/bin/env python3
"""
Full evaluation pipeline for generated MIDI files.
Metrics: pitch class distribution, pentatonic adherence, note density,
duration distribution, interval distribution, OA metrics, self-similarity.

Run: python3 scripts/evaluate_full.py
"""

import os, json, math, collections, argparse
import numpy as np
import mido

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"
TRAIN_DIR = os.path.join(ROOT, "MIDI_transposed")

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
ALL_VALID_PCS = set()
for s in PENTATONIC_SCALES:
    ALL_VALID_PCS |= PENTATONIC_SCALES[s] | PRESSED_PCS[s]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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

    # Pitch class histogram
    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[p % 12] += 1
    pc_hist_norm = pc_hist / max(pc_hist.sum(), 1)

    # Duration histogram (bins: 0-5s in 0.1s steps)
    dur_bins = np.arange(0, 5.05, 0.1)
    dur_hist, _ = np.histogram(durations, bins=dur_bins, density=True)

    # Interval histogram (bins: -24 to +24 semitones)
    int_bins = np.arange(-24.5, 25.5, 1)
    int_hist, _ = np.histogram(intervals, bins=int_bins, density=True)

    # IOI histogram
    ioi_bins = np.arange(0, 2.05, 0.05)
    ioi_hist, _ = np.histogram(iois, bins=ioi_bins, density=True)

    total_dur = max(onsets[-1] + durations[-1], 0.001) - onsets[0]

    # Pentatonic purity (best-matching scale)
    best_purity = 0
    best_scale = "?"
    for sname, pcs in PENTATONIC_SCALES.items():
        ext = pcs | PRESSED_PCS.get(sname, set())
        match = sum(pc_hist[pc] for pc in ext)
        ratio = match / max(sum(pc_hist), 1)
        if ratio > best_purity:
            best_purity = ratio
            best_scale = sname

    # Pitch entropy
    eps = 1e-9
    pc_entropy = float(-np.sum(pc_hist_norm * np.log2(pc_hist_norm + eps)))

    # Track count / polyphony
    events = []
    for p, on, dur, v in notes:
        events.append((on, 1))
        events.append((on + dur, -1))
    events.sort()
    max_simul = 0
    cur = 0
    for _, delta in events:
        cur += delta
        max_simul = max(max_simul, cur)

    return {
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


def overlapping_area(hist1, hist2):
    """Compute overlapping area between two normalized histograms."""
    h1 = np.array(hist1)
    h2 = np.array(hist2)
    if len(h1) != len(h2):
        min_len = min(len(h1), len(h2))
        h1 = h1[:min_len]
        h2 = h2[:min_len]
    # Normalize
    s1 = h1.sum()
    s2 = h2.sum()
    if s1 > 0:
        h1 = h1 / s1
    if s2 > 0:
        h2 = h2 / s2
    return float(np.sum(np.minimum(h1, h2)))


def evaluate_directory(midi_dir):
    files = sorted([
        os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(".mid")
    ])
    all_dists = []
    for fp in files:
        notes = midi_to_notes(fp)
        d = compute_distributions(notes)
        if d:
            d["file"] = os.path.basename(fp)
            all_dists.append(d)
    return all_dists


def aggregate_metrics(dists):
    if not dists:
        return {}
    return {
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


def compute_oa_metrics(train_agg, gen_agg):
    """Compute OA between training and generated distributions."""
    return {
        "OA_pitch_class": round(overlapping_area(train_agg["agg_pc_hist"], gen_agg["agg_pc_hist"]), 4),
        "OA_duration": round(overlapping_area(train_agg["agg_dur_hist"], gen_agg["agg_dur_hist"]), 4),
        "OA_interval": round(overlapping_area(train_agg["agg_int_hist"], gen_agg["agg_int_hist"]), 4),
        "OA_ioi": round(overlapping_area(train_agg["agg_ioi_hist"], gen_agg["agg_ioi_hist"]), 4),
    }


def generate_report(train_agg, eval_results, out_path):
    """Generate evaluation report markdown."""
    lines = ["# Evaluation Report", "", f"Generated: 2026-03-25", ""]

    # Training data baseline
    lines.append("## Training Data Baseline")
    lines.append("")
    for k, v in train_agg.items():
        if not k.startswith("agg_"):
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Comparison table
    lines.append("## Model Comparison")
    lines.append("")
    metrics = ["mean_n_notes", "mean_density", "mean_duration", "mean_velocity",
               "mean_pc_entropy", "mean_penta_purity", "mean_large_leap_rate",
               "mean_interval", "mean_max_simul", "mean_pitch"]
    header = "| Metric | Training |"
    sep = "|--------|----------|"
    for name in eval_results:
        header += f" {name} |"
        sep += "----------|"
    lines.append(header)
    lines.append(sep)
    for m in metrics:
        row = f"| {m} | {train_agg.get(m, 'N/A')} |"
        for name, er in eval_results.items():
            row += f" {er['aggregate'].get(m, 'N/A')} |"
        lines.append(row)
    lines.append("")

    # OA Metrics
    lines.append("## Overlapping Area (OA) Metrics")
    lines.append("(Higher = more similar to training data, 1.0 = identical distribution)")
    lines.append("")
    oa_header = "| Distribution |"
    oa_sep = "|-------------|"
    for name in eval_results:
        oa_header += f" {name} |"
        oa_sep += "----------|"
    lines.append(oa_header)
    lines.append(oa_sep)
    for oa_key in ["OA_pitch_class", "OA_duration", "OA_interval", "OA_ioi"]:
        row = f"| {oa_key} |"
        for name, er in eval_results.items():
            row += f" {er['oa'].get(oa_key, 'N/A')} |"
        lines.append(row)
    lines.append("")

    # Recommendations
    lines.append("## Analysis")
    lines.append("")
    for name, er in eval_results.items():
        agg = er["aggregate"]
        oa = er["oa"]
        lines.append(f"### {name}")
        pp = agg.get("mean_penta_purity", 0)
        if pp > 0.95:
            lines.append(f"- Pentatonic purity: {pp} (excellent)")
        elif pp > 0.8:
            lines.append(f"- Pentatonic purity: {pp} (good, some chromatic notes)")
        else:
            lines.append(f"- Pentatonic purity: {pp} (POOR - constrained decoding needed)")
        oa_pc = oa.get("OA_pitch_class", 0)
        lines.append(f"- Pitch class OA: {oa_pc} ({'good' if oa_pc > 0.7 else 'needs improvement'})")
        oa_dur = oa.get("OA_duration", 0)
        lines.append(f"- Duration OA: {oa_dur} ({'good' if oa_dur > 0.5 else 'needs improvement'})")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dirs", type=str, nargs="+",
                        help="Directories to evaluate (name:path pairs)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "outputs/evaluation"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Default eval dirs if not specified
    if not args.eval_dirs:
        args.eval_dirs = []
        candidate_dirs = {
            "midirwkv_pretrained": os.path.join(ROOT, "archive/outputs/midirwkv_pretrained"),
            "midirwkv_state_tuned": os.path.join(ROOT, "archive/outputs/midirwkv_finetuned"),
            "midirwkv_lora_constrained": os.path.join(ROOT, "outputs/midirwkv_constrained"),
            "midirwkv_lora_unconstrained": os.path.join(ROOT, "outputs/midirwkv_constrained_unconstrained"),
            "moonbeam_pretrained": os.path.join(ROOT, "archive/outputs/moonbeam_pretrained"),
            "moonbeam_finetuned": os.path.join(ROOT, "archive/outputs/moonbeam_finetuned"),
        }
        for name, path in candidate_dirs.items():
            if os.path.isdir(path) and any(f.endswith(".mid") for f in os.listdir(path)):
                args.eval_dirs.append(f"{name}:{path}")

    # Evaluate training data
    print("Evaluating training data...")
    train_dists = evaluate_directory(TRAIN_DIR)
    train_agg = aggregate_metrics(train_dists)
    print(f"  {train_agg['n_files']} files, {train_agg['mean_n_notes']} mean notes")

    # Evaluate each variant
    eval_results = {}
    for spec in args.eval_dirs:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name = os.path.basename(spec)
            path = spec
        if not os.path.isdir(path):
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"Evaluating {name}...")
        dists = evaluate_directory(path)
        if not dists:
            print(f"  No valid MIDI files")
            continue
        agg = aggregate_metrics(dists)
        oa = compute_oa_metrics(train_agg, agg)
        eval_results[name] = {"aggregate": agg, "oa": oa, "per_file": dists}
        print(f"  {agg['n_files']} files, penta_purity={agg['mean_penta_purity']}, "
              f"OA_pc={oa['OA_pitch_class']}")

    # Save results
    results_json = {
        "training_data": train_agg,
        "variants": {n: {"aggregate": r["aggregate"], "oa": r["oa"]}
                     for n, r in eval_results.items()},
    }
    json_path = os.path.join(args.out_dir, "full_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nMetrics saved to {json_path}")

    # Generate report
    report = generate_report(train_agg, eval_results, args.out_dir)
    report_path = os.path.join(args.out_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Variant':<30} {'Files':<6} {'Penta%':<8} {'Density':<8} {'OA_PC':<8} {'OA_Dur':<8}")
    print("-" * 68)
    print(f"{'training_data':<30} {train_agg['n_files']:<6} {train_agg['mean_penta_purity']:<8} {train_agg['mean_density']:<8} {'1.0000':<8} {'1.0000':<8}")
    for name, er in eval_results.items():
        a = er["aggregate"]
        o = er["oa"]
        print(f"{name:<30} {a['n_files']:<6} {a['mean_penta_purity']:<8} {a['mean_density']:<8} {o['OA_pitch_class']:<8} {o['OA_duration']:<8}")


if __name__ == "__main__":
    main()

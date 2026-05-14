#!/usr/bin/env python3
"""
Step 5: Evaluate generated MIDI files against training data.

Trial 4 changes vs Trial 3:
- Evaluates post-processed files (from 04_postprocess.py)
- Per-category analysis (val, test, synthetic)
- Comparison against Trial 3 and original best targets

Metrics:
- OA: pitch class, duration, interval, IOI distribution overlaps
- Pentatonic purity, note density, pitch range, polyphony
- Self-repetition (4/8/12-gram)
- Per-category breakdown
"""

import os
import json
import numpy as np
import mido
from collections import Counter

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))

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
    for key in ("self_rep_4gram", "self_rep_8gram", "self_rep_12gram"):
        vals = [d[key] for d in dists if key in d]
        if vals:
            result[f"mean_{key}"] = round(np.mean(vals), 4)
    return result


def main():
    eval_dir = os.path.join(TRIAL_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate training data baseline (Trial 4's augmented training data)
    train_dir = os.path.join(TRIAL_ROOT, "data", "train")
    print("Evaluating training data baseline...")
    train_dists = evaluate_directory(train_dir)
    train_agg = aggregate_metrics(train_dists)
    print(f"  {train_agg.get('n_files', 0)} files")

    # Evaluate test data (Trial 5's own test split)
    test_dir = os.path.join(TRIAL_ROOT, "data", "test")
    print("Evaluating test data...")
    test_dists = evaluate_directory(test_dir)
    test_agg = aggregate_metrics(test_dists)
    print(f"  {test_agg.get('n_files', 0)} files")

    # Find all generated checkpoint variants (post-processed only)
    gen_dir = os.path.join(TRIAL_ROOT, "generated")
    all_variants = {}

    if os.path.isdir(gen_dir):
        for ckpt_name in sorted(os.listdir(gen_dir)):
            ckpt_gen_dir = os.path.join(gen_dir, ckpt_name)
            if not os.path.isdir(ckpt_gen_dir):
                continue

            # Evaluate per category (post-processed)
            for category in ["val_postprocessed", "test_postprocessed", "synthetic_postprocessed"]:
                var_dir = os.path.join(ckpt_gen_dir, category)
                if not os.path.isdir(var_dir):
                    continue

                variant_name = f"{ckpt_name}/{category}"
                print(f"Evaluating {variant_name}...")
                dists = evaluate_directory(var_dir)
                agg = aggregate_metrics(dists)

                oa = {}
                if train_agg and agg:
                    oa = {
                        "OA_pitch_class": round(overlapping_area(train_agg["agg_pc_hist"], agg["agg_pc_hist"]), 4),
                        "OA_duration": round(overlapping_area(train_agg["agg_dur_hist"], agg["agg_dur_hist"]), 4),
                        "OA_interval": round(overlapping_area(train_agg["agg_int_hist"], agg["agg_int_hist"]), 4),
                        "OA_ioi": round(overlapping_area(train_agg["agg_ioi_hist"], agg["agg_ioi_hist"]), 4),
                    }

                all_variants[variant_name] = {"aggregate": agg, "oa": oa, "per_file": dists}
                print(f"  {agg.get('n_files', 0)} files, "
                      f"purity={agg.get('mean_penta_purity', 'N/A')}, "
                      f"OA_pc={oa.get('OA_pitch_class', 'N/A')}")

            # Also evaluate combined (all categories together) for overall metrics
            combined_dists = []
            for category in ["val_postprocessed", "test_postprocessed", "synthetic_postprocessed"]:
                var_dir = os.path.join(ckpt_gen_dir, category)
                if os.path.isdir(var_dir):
                    combined_dists.extend(evaluate_directory(var_dir))

            if combined_dists:
                variant_name = f"{ckpt_name}/all_postprocessed"
                agg = aggregate_metrics(combined_dists)
                oa = {}
                if train_agg and agg:
                    oa = {
                        "OA_pitch_class": round(overlapping_area(train_agg["agg_pc_hist"], agg["agg_pc_hist"]), 4),
                        "OA_duration": round(overlapping_area(train_agg["agg_dur_hist"], agg["agg_dur_hist"]), 4),
                        "OA_interval": round(overlapping_area(train_agg["agg_int_hist"], agg["agg_int_hist"]), 4),
                        "OA_ioi": round(overlapping_area(train_agg["agg_ioi_hist"], agg["agg_ioi_hist"]), 4),
                    }
                all_variants[variant_name] = {"aggregate": agg, "oa": oa, "per_file": combined_dists}
                print(f"  Combined: {agg.get('n_files', 0)} files, OA_pc={oa.get('OA_pitch_class', 'N/A')}")

    # Save results
    results = {
        "training_data": train_agg,
        "test_data": test_agg,
        "variants": {n: {"aggregate": v["aggregate"], "oa": v["oa"]} for n, v in all_variants.items()},
    }
    json_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    lines = ["# Trial 4 Evaluation Report", ""]
    lines.append("## Training Data Baseline")
    for k, v in train_agg.items():
        if not k.startswith("agg_"):
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    if test_agg:
        lines.append("## Test Data (unseen)")
        for k, v in test_agg.items():
            if not k.startswith("agg_"):
                lines.append(f"- **{k}:** {v}")
        lines.append("")

    if all_variants:
        lines.append("## Generated Variants (Post-Processed)")
        lines.append("")

        # Comparison table
        header = "| Metric | Training |"
        sep = "|--------|----------|"
        for name in all_variants:
            short = name.split("/")[-1][:20]
            header += f" {short} |"
            sep += "----------|"
        lines.append(header)
        lines.append(sep)

        metrics_keys = ["mean_n_notes", "mean_density", "mean_duration", "mean_velocity",
                        "mean_pc_entropy", "mean_penta_purity", "mean_large_leap_rate",
                        "mean_interval", "mean_max_simul", "mean_pitch",
                        "mean_self_rep_4gram", "mean_self_rep_8gram", "mean_self_rep_12gram"]
        for m in metrics_keys:
            row = f"| {m} | {train_agg.get(m, 'N/A')} |"
            for name, v in all_variants.items():
                row += f" {v['aggregate'].get(m, 'N/A')} |"
            lines.append(row)
        lines.append("")

        lines.append("## OA Metrics (1.0 = identical to training)")
        lines.append("")
        oa_header = "| Distribution |"
        oa_sep = "|-------------|"
        for name in all_variants:
            short = name.split("/")[-1][:20]
            oa_header += f" {short} |"
            oa_sep += "----------|"
        lines.append(oa_header)
        lines.append(oa_sep)
        for oa_key in ["OA_pitch_class", "OA_duration", "OA_interval", "OA_ioi"]:
            row = f"| {oa_key} |"
            for name, v in all_variants.items():
                row += f" {v['oa'].get(oa_key, 'N/A')} |"
            lines.append(row)
        lines.append("")

        # Comparison to targets
        lines.append("## Target Comparison")
        lines.append("")
        lines.append("| Metric | Original Best | Trial 3 Best | Trial 4 Target | Trial 4 Best |")
        lines.append("|--------|--------------|-------------|---------------|-------------|")

        # Find best checkpoint (by OA_pitch_class in all_postprocessed)
        best_variant = None
        best_oa_pc = 0
        for name, v in all_variants.items():
            if "all_postprocessed" in name:
                oa_pc = v["oa"].get("OA_pitch_class", 0)
                if oa_pc > best_oa_pc:
                    best_oa_pc = oa_pc
                    best_variant = v

        if best_variant:
            ba = best_variant["aggregate"]
            bo = best_variant["oa"]
            lines.append(f"| OA pitch class | 0.918 | 0.797 | > 0.80 | {bo.get('OA_pitch_class', 'N/A')} |")
            lines.append(f"| OA duration | 0.839 | 0.642 | > 0.65 | {bo.get('OA_duration', 'N/A')} |")
            lines.append(f"| Note count | 178 | 49 | > 150 | {ba.get('mean_n_notes', 'N/A')} |")
            lines.append(f"| Density (n/s) | 3.45 | 2.11 | 3.0-4.0 | {ba.get('mean_density', 'N/A')} |")
            lines.append(f"| Pentatonic purity | 100% | 97.9% | 100% | {ba.get('mean_penta_purity', 'N/A')} |")
        lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(eval_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {json_path}")
    print(f"Report saved to: {report_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Variant':<40} {'Files':<6} {'Notes':<7} {'Penta%':<8} {'Dens':<7} {'OA_PC':<8} {'OA_Dur':<8} {'Rep4g':<8}")
    print("-" * 92)
    if train_agg:
        rep4 = train_agg.get('mean_self_rep_4gram', 'N/A')
        print(f"{'training':<40} {train_agg['n_files']:<6} {train_agg['mean_n_notes']:<7} "
              f"{train_agg['mean_penta_purity']:<8} {train_agg['mean_density']:<7} "
              f"{'1.0000':<8} {'1.0000':<8} {rep4:<8}")
    for name, v in all_variants.items():
        a, o = v["aggregate"], v["oa"]
        rep4 = a.get('mean_self_rep_4gram', 'N/A')
        print(f"{name:<40} {a.get('n_files', 0):<6} {a.get('mean_n_notes', 'N/A'):<7} "
              f"{a.get('mean_penta_purity', 'N/A'):<8} {a.get('mean_density', 'N/A'):<7} "
              f"{o.get('OA_pitch_class', 'N/A'):<8} {o.get('OA_duration', 'N/A'):<8} {rep4:<8}")


if __name__ == "__main__":
    main()

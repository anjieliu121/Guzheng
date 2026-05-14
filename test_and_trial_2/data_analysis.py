#!/usr/bin/env python3
"""
Phase 0: Deep analysis of all guzheng MIDI training data.
Produces docs/data_analysis_report.md and outputs/plots/ visualizations.

Run: python3 scripts/data_analysis.py
"""

import os, json, math, collections
import mido
import numpy as np

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"
MIDI_ORIG = os.path.join(ROOT, "MIDI")
MIDI_TRANS = os.path.join(ROOT, "MIDI_transposed")
OUT_DIR = os.path.join(ROOT, "outputs/plots")
REPORT_PATH = os.path.join(ROOT, "docs/data_analysis_report.md")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Pentatonic scales (pitch class sets)
PENTATONIC_SCALES = {
    "A": {9, 11, 1, 4, 6},   # A B C# E F#
    "C": {0, 2, 4, 7, 9},    # C D E G A
    "D": {2, 4, 6, 9, 11},   # D E F# A B
    "F": {5, 7, 9, 0, 2},    # F G A C D
    "G": {7, 9, 11, 2, 4},   # G A B D E
}
# Pressed string pitches (4 and 7 in jianpu) per scale
PRESSED_PCS = {
    "D": {7, 1},   # G, C#
    "G": {0, 6},   # C, F#
    "C": {5, 11},  # F, B
    "A": {2, 8},   # D, G#
    "F": {10, 4},  # A#, E
}
ALL_PENTA_PCS = set().union(*PENTATONIC_SCALES.values())

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_note_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def extract_notes(midi_path):
    """Extract (pitch, onset_sec, duration_sec, velocity) from MIDI."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        return []

    tempo = 500000
    tpb = mid.ticks_per_beat
    notes = []
    track_count = len([t for t in mid.tracks if any(m.type in ('note_on', 'note_off') for m in t)])

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
                        onset_s = mido.tick2second(active.pop(msg.note)[0], tpb, tempo)
                        off_s = mido.tick2second(abs_tick, tpb, tempo)
                        if off_s > onset_s:
                            notes.append((msg.note, onset_s, off_s - onset_s, vel if vel > 0 else 64))

    return notes, track_count


def detect_ornaments(notes):
    """Detect tremolo and glissando candidates."""
    if len(notes) < 3:
        return {"tremolo_count": 0, "glissando_count": 0, "tremolo_regions": [], "glissando_regions": []}

    sorted_notes = sorted(notes, key=lambda x: x[1])
    pitches = [n[0] for n in sorted_notes]
    onsets = [n[1] for n in sorted_notes]

    # Tremolo: rapid alternation between 2 notes (IOI < 150ms)
    tremolo_count = 0
    tremolo_regions = []
    i = 0
    while i < len(pitches) - 2:
        ioi1 = onsets[i+1] - onsets[i]
        ioi2 = onsets[i+2] - onsets[i+1]
        if ioi1 < 0.15 and ioi2 < 0.15:
            if abs(pitches[i+1] - pitches[i]) <= 2 and abs(pitches[i+2] - pitches[i+1]) <= 2:
                start = i
                while i < len(pitches) - 1 and (onsets[i+1] - onsets[i]) < 0.15:
                    i += 1
                tremolo_count += 1
                tremolo_regions.append((start, i))
        i += 1

    # Glissando: 5+ consecutive stepwise notes within short IOI
    glissando_count = 0
    glissando_regions = []
    i = 0
    while i < len(pitches) - 4:
        run_len = 1
        j = i
        while j < len(pitches) - 1:
            interval = abs(pitches[j+1] - pitches[j])
            ioi = onsets[j+1] - onsets[j]
            if 1 <= interval <= 3 and ioi < 0.2:
                run_len += 1
                j += 1
            else:
                break
        if run_len >= 5:
            glissando_count += 1
            glissando_regions.append((i, j))
            i = j + 1
        else:
            i += 1

    return {
        "tremolo_count": tremolo_count,
        "glissando_count": glissando_count,
        "tremolo_regions": tremolo_regions,
        "glissando_regions": glissando_regions,
    }


def detect_polyphony(notes):
    """Compute max simultaneous notes and fraction of time with >1 note."""
    if not notes:
        return 0, 0.0
    sorted_notes = sorted(notes, key=lambda x: x[1])
    events = []
    for pitch, onset, dur, vel in sorted_notes:
        events.append((onset, 1))
        events.append((onset + dur, -1))
    events.sort()

    max_simul = 0
    cur = 0
    polyphonic_time = 0.0
    last_t = events[0][0]
    for t, delta in events:
        if cur > 1:
            polyphonic_time += t - last_t
        last_t = t
        cur += delta
        max_simul = max(max_simul, cur)

    total_dur = max(n[1] + n[2] for n in sorted_notes) - min(n[1] for n in sorted_notes)
    poly_frac = polyphonic_time / max(total_dur, 0.001)
    return max_simul, poly_frac


def best_pentatonic_match(pitches):
    """Find the pentatonic scale with highest adherence."""
    if not pitches:
        return "?", 0.0
    pc_counts = collections.Counter(p % 12 for p in pitches)
    best_scale = "?"
    best_ratio = 0.0
    for name, pcs in PENTATONIC_SCALES.items():
        # Include pressed strings as valid
        extended = pcs | PRESSED_PCS.get(name, set())
        match = sum(pc_counts.get(pc, 0) for pc in extended)
        ratio = match / len(pitches)
        if ratio > best_ratio:
            best_ratio = ratio
            best_scale = name
    return best_scale, best_ratio


def analyze_file(midi_path):
    """Full analysis of one MIDI file."""
    notes_data = extract_notes(midi_path)
    if isinstance(notes_data, tuple):
        notes, track_count = notes_data
    else:
        notes = notes_data
        track_count = 1

    if not notes:
        return None

    pitches = [n[0] for n in notes]
    durations = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    onsets = sorted([n[1] for n in notes])

    # IOI (inter-onset interval)
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1) if onsets[i+1] > onsets[i]]

    # Intervals
    sorted_by_onset = sorted(notes, key=lambda x: x[1])
    melodic_intervals = [sorted_by_onset[i+1][0] - sorted_by_onset[i][0] for i in range(len(sorted_by_onset)-1)]

    # Pitch class distribution
    pc_counts = np.zeros(12)
    for p in pitches:
        pc_counts[p % 12] += 1
    pc_dist = pc_counts / max(pc_counts.sum(), 1)

    # Pentatonic analysis
    best_scale, penta_ratio = best_pentatonic_match(pitches)

    # Non-pentatonic notes
    scale_pcs = PENTATONIC_SCALES.get(best_scale, set()) | PRESSED_PCS.get(best_scale, set())
    non_penta = [p for p in pitches if p % 12 not in scale_pcs]

    # Ornaments
    ornaments = detect_ornaments(notes)

    # Polyphony
    max_simul, poly_frac = detect_polyphony(notes)

    # Duration stats
    total_time = max(n[1] + n[2] for n in notes) - min(n[1] for n in notes)

    # Pitch entropy
    eps = 1e-9
    pc_entropy = float(-np.sum(pc_dist * np.log2(pc_dist + eps)))

    return {
        "file": os.path.basename(midi_path),
        "track_count": track_count,
        "n_notes": len(notes),
        "total_duration_sec": round(total_time, 1),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "pitch_min_name": midi_note_name(min(pitches)),
        "pitch_max_name": midi_note_name(max(pitches)),
        "mean_pitch": round(np.mean(pitches), 1),
        "best_scale": best_scale,
        "penta_ratio": round(penta_ratio, 4),
        "non_penta_count": len(non_penta),
        "non_penta_pcs": sorted(set(p % 12 for p in non_penta)) if non_penta else [],
        "pc_distribution": pc_dist.tolist(),
        "pc_entropy": round(pc_entropy, 3),
        "mean_duration": round(np.mean(durations), 4),
        "median_duration": round(np.median(durations), 4),
        "min_duration": round(min(durations), 4),
        "max_duration": round(max(durations), 4),
        "mean_velocity": round(np.mean(velocities), 1),
        "min_velocity": int(min(velocities)),
        "max_velocity": int(max(velocities)),
        "velocity_std": round(np.std(velocities), 1),
        "note_density": round(len(notes) / max(total_time, 0.001), 2),
        "mean_ioi": round(np.mean(iois), 4) if iois else 0,
        "median_ioi": round(np.median(iois), 4) if iois else 0,
        "max_simultaneous": max_simul,
        "polyphonic_fraction": round(poly_frac, 3),
        "tremolo_count": ornaments["tremolo_count"],
        "glissando_count": ornaments["glissando_count"],
        "melodic_intervals_abs_mean": round(np.mean([abs(i) for i in melodic_intervals]), 2) if melodic_intervals else 0,
        "large_leap_rate": round(sum(1 for i in melodic_intervals if abs(i) > 12) / max(len(melodic_intervals), 1), 4),
    }


def generate_report(orig_analyses, trans_analyses):
    """Generate markdown report."""
    lines = []
    lines.append("# Guzheng MIDI Data Analysis Report")
    lines.append(f"\nGenerated: 2026-03-25")
    lines.append("")

    # Summary
    lines.append("## 1. Dataset Inventory")
    lines.append(f"- **Original files:** {len(orig_analyses)}")
    lines.append(f"- **Transposed files:** {len(trans_analyses)}")
    lines.append(f"- **Total training corpus:** {len(orig_analyses) + len(trans_analyses)} files")
    lines.append("")

    # Original files table
    lines.append("## 2. Original Files Overview")
    lines.append("")
    lines.append("| File | Notes | Duration | Pitch Range | Scale | Penta% | Density | Max Simul |")
    lines.append("|------|-------|----------|-------------|-------|--------|---------|-----------|")
    for a in sorted(orig_analyses, key=lambda x: x["file"]):
        lines.append(
            f"| {a['file']} | {a['n_notes']} | {a['total_duration_sec']}s | "
            f"{a['pitch_min_name']}–{a['pitch_max_name']} ({a['pitch_min']}–{a['pitch_max']}) | "
            f"{a['best_scale']} | {a['penta_ratio']*100:.1f}% | {a['note_density']} n/s | {a['max_simultaneous']} |"
        )
    lines.append("")

    # Aggregate stats
    all_a = orig_analyses + trans_analyses
    lines.append("## 3. Aggregate Statistics (All Files)")
    lines.append("")

    pitches_all = [a["pitch_min"] for a in all_a] + [a["pitch_max"] for a in all_a]
    lines.append(f"- **Global pitch range:** MIDI {min(a['pitch_min'] for a in all_a)}–{max(a['pitch_max'] for a in all_a)} "
                 f"({midi_note_name(min(a['pitch_min'] for a in all_a))}–{midi_note_name(max(a['pitch_max'] for a in all_a))})")
    lines.append(f"- **Mean note count:** {np.mean([a['n_notes'] for a in all_a]):.0f}")
    lines.append(f"- **Mean duration:** {np.mean([a['total_duration_sec'] for a in all_a]):.1f}s")
    lines.append(f"- **Mean note density:** {np.mean([a['note_density'] for a in all_a]):.2f} notes/sec")
    lines.append(f"- **Mean note duration:** {np.mean([a['mean_duration'] for a in all_a]):.3f}s")
    lines.append(f"- **Mean velocity:** {np.mean([a['mean_velocity'] for a in all_a]):.1f}")
    lines.append(f"- **Mean pentatonic adherence:** {np.mean([a['penta_ratio'] for a in all_a])*100:.1f}%")
    lines.append(f"- **Mean large leap rate (>12 semitones):** {np.mean([a['large_leap_rate'] for a in all_a]):.4f}")
    lines.append(f"- **Mean melodic interval:** {np.mean([a['melodic_intervals_abs_mean'] for a in all_a]):.2f} semitones")
    lines.append("")

    # Pentatonic scale distribution
    lines.append("## 4. Pentatonic Scale Distribution")
    lines.append("")
    scale_counts = collections.Counter(a["best_scale"] for a in all_a)
    for s, c in sorted(scale_counts.items()):
        lines.append(f"- **{s} major pentatonic:** {c} files")
    lines.append("")

    # Pitch class distribution (aggregate)
    lines.append("## 5. Aggregate Pitch Class Distribution")
    lines.append("")
    agg_pc = np.zeros(12)
    for a in all_a:
        agg_pc += np.array(a["pc_distribution"])
    agg_pc /= max(agg_pc.sum(), 1)
    lines.append("| Pitch Class | C | C# | D | D# | E | F | F# | G | G# | A | A# | B |")
    lines.append("|-------------|---|-----|---|-----|---|---|------|---|------|---|------|---|")
    row = "| Proportion |"
    for v in agg_pc:
        row += f" {v:.3f} |"
    lines.append(row)
    lines.append("")

    # Highlight which are pentatonic
    penta_pcs_d = PENTATONIC_SCALES["D"]
    lines.append("D pentatonic (D E F# A B = PCs 2,4,6,9,11) weight: "
                 f"{sum(agg_pc[pc] for pc in penta_pcs_d):.3f}")
    lines.append("")

    # Texture analysis
    lines.append("## 6. Texture Analysis")
    lines.append("")
    lines.append(f"- **Mean max simultaneous notes:** {np.mean([a['max_simultaneous'] for a in all_a]):.1f}")
    lines.append(f"- **Mean polyphonic fraction:** {np.mean([a['polyphonic_fraction'] for a in all_a]):.3f}")
    mono = sum(1 for a in all_a if a["max_simultaneous"] <= 2)
    lines.append(f"- **Predominantly monophonic files (max 2 simultaneous):** {mono}/{len(all_a)}")
    lines.append("")

    # Ornament detection
    lines.append("## 7. Ornament Detection")
    lines.append("")
    total_trem = sum(a["tremolo_count"] for a in all_a)
    total_gliss = sum(a["glissando_count"] for a in all_a)
    lines.append(f"- **Total tremolo regions detected:** {total_trem}")
    lines.append(f"- **Total glissando regions detected:** {total_gliss}")
    lines.append("")

    # Velocity analysis
    lines.append("## 8. Velocity Distribution")
    lines.append("")
    lines.append(f"- **Mean velocity:** {np.mean([a['mean_velocity'] for a in all_a]):.1f}")
    lines.append(f"- **Global velocity range:** {min(a['min_velocity'] for a in all_a)}–{max(a['max_velocity'] for a in all_a)}")
    lines.append(f"- **Mean velocity std:** {np.mean([a['velocity_std'] for a in all_a]):.1f}")
    has_velocity = sum(1 for a in all_a if a["velocity_std"] > 5)
    lines.append(f"- **Files with meaningful velocity variation (std>5):** {has_velocity}/{len(all_a)}")
    lines.append("")

    # Duration analysis
    lines.append("## 9. Note Duration Distribution")
    lines.append("")
    lines.append(f"- **Mean note duration:** {np.mean([a['mean_duration'] for a in all_a]):.4f}s")
    lines.append(f"- **Median note duration:** {np.mean([a['median_duration'] for a in all_a]):.4f}s")
    lines.append(f"- **Shortest note (any file):** {min(a['min_duration'] for a in all_a):.4f}s")
    lines.append(f"- **Longest note (any file):** {max(a['max_duration'] for a in all_a):.4f}s")
    lines.append("")

    # IOI analysis
    lines.append("## 10. Inter-Onset Interval (IOI)")
    lines.append("")
    lines.append(f"- **Mean IOI:** {np.mean([a['mean_ioi'] for a in all_a]):.4f}s")
    lines.append(f"- **Median IOI:** {np.mean([a['median_ioi'] for a in all_a]):.4f}s")
    lines.append("")

    # Detailed per-file table (original only)
    lines.append("## 11. Detailed Per-File Analysis (Original)")
    lines.append("")
    lines.append("| File | Tremolo | Glissando | Poly Frac | PC Entropy | Vel Mean±Std | Leap Rate |")
    lines.append("|------|---------|-----------|-----------|------------|-------------|-----------|")
    for a in sorted(orig_analyses, key=lambda x: x["file"]):
        lines.append(
            f"| {a['file']} | {a['tremolo_count']} | {a['glissando_count']} | "
            f"{a['polyphonic_fraction']:.3f} | {a['pc_entropy']:.2f} | "
            f"{a['mean_velocity']:.0f}±{a['velocity_std']:.0f} | {a['large_leap_rate']:.4f} |"
        )
    lines.append("")

    # Red flags
    lines.append("## 12. Red Flags & Quality Notes")
    lines.append("")
    issues = []
    for a in all_a:
        if a["penta_ratio"] < 0.9:
            issues.append(f"- `{a['file']}`: pentatonic adherence only {a['penta_ratio']*100:.1f}% (non-penta PCs: {a['non_penta_pcs']})")
        if a["max_simultaneous"] > 4:
            issues.append(f"- `{a['file']}`: max {a['max_simultaneous']} simultaneous notes (possible chord/error)")
        if a["large_leap_rate"] > 0.1:
            issues.append(f"- `{a['file']}`: large leap rate {a['large_leap_rate']:.3f} (>10% of intervals exceed octave)")
    if issues:
        for issue in issues:
            lines.append(issue)
    else:
        lines.append("No significant red flags detected.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=== Phase 0: Data Analysis ===")

    # Analyze original files
    print("\nAnalyzing original MIDI files...")
    orig_analyses = []
    orig_files = sorted([os.path.join(MIDI_ORIG, f) for f in os.listdir(MIDI_ORIG) if f.endswith(".mid")])
    for fp in orig_files:
        print(f"  {os.path.basename(fp)}")
        result = analyze_file(fp)
        if result:
            orig_analyses.append(result)

    # Analyze transposed files
    print(f"\nAnalyzing {len(os.listdir(MIDI_TRANS))} transposed MIDI files...")
    trans_analyses = []
    trans_files = sorted([os.path.join(MIDI_TRANS, f) for f in os.listdir(MIDI_TRANS) if f.endswith(".mid")])
    for fp in trans_files:
        result = analyze_file(fp)
        if result:
            trans_analyses.append(result)
    print(f"  Analyzed {len(trans_analyses)} transposed files.")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(orig_analyses, trans_analyses)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")

    # Save raw analysis JSON
    raw_path = os.path.join(ROOT, "outputs/evaluation/data_analysis_raw.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump({
            "original": orig_analyses,
            "transposed": trans_analyses,
        }, f, indent=2)
    print(f"Raw data saved to {raw_path}")

    # Print summary
    all_a = orig_analyses + trans_analyses
    print(f"\n=== Summary ===")
    print(f"Total files: {len(all_a)} ({len(orig_analyses)} original + {len(trans_analyses)} transposed)")
    print(f"Mean pentatonic adherence: {np.mean([a['penta_ratio'] for a in all_a])*100:.1f}%")
    print(f"Pitch range: {min(a['pitch_min'] for a in all_a)}-{max(a['pitch_max'] for a in all_a)}")
    print(f"Mean note density: {np.mean([a['note_density'] for a in all_a]):.2f} notes/sec")
    print(f"Mean note duration: {np.mean([a['mean_duration'] for a in all_a]):.3f}s")
    print(f"Mean max simultaneous: {np.mean([a['max_simultaneous'] for a in all_a]):.1f}")


if __name__ == "__main__":
    main()

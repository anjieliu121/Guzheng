#!/usr/bin/env python3
"""
Analyze 50 generated guzheng MIDI files for authenticity.
Ranks them by how convincingly they sound like real traditional Chinese
pentatonic guzheng music.
"""

import os
import glob
import mido
import numpy as np
from collections import Counter, defaultdict

MIDI_DIR = "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_batch2_ks/"
KS_THRESHOLD = 36  # pitches below this are keyswitches

# D-pentatonic: D E F# A B  =>  pitch classes 2, 4, 6, 9, 11
D_PENTA = {2, 4, 6, 9, 11}
PC_NAMES = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}

# Known keyswitch mappings (common for guzheng libraries)
KS_ARTICULATIONS = {
    24: "sustain/normal",
    25: "tremolo",
    26: "trill",
    27: "bend_up",
    28: "bend_down",
    29: "glissando_up",
    30: "glissando_down",
    31: "harmonic",
    32: "staccato",
    33: "portamento",
    34: "vibrato",
    35: "muted",
}


def parse_midi(filepath):
    """Parse a MIDI file and extract melodic notes and keyswitches."""
    mid = mido.MidiFile(filepath)
    tpb = mid.ticks_per_beat

    # Find tempo
    tempo = 500000  # default 120 BPM
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    # Collect all note_on events with absolute tick times
    melodic_notes = []
    keyswitches = []

    for track in mid.tracks:
        abs_tick = 0
        active_notes = {}  # pitch -> (start_tick, velocity)
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note < KS_THRESHOLD:
                    keyswitches.append({
                        'pitch': msg.note,
                        'tick': abs_tick,
                        'velocity': msg.velocity,
                    })
                else:
                    active_notes[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note >= KS_THRESHOLD and msg.note in active_notes:
                    start_tick, vel = active_notes.pop(msg.note)
                    dur_ticks = abs_tick - start_tick
                    melodic_notes.append({
                        'pitch': msg.note,
                        'start_tick': start_tick,
                        'dur_ticks': dur_ticks,
                        'velocity': vel,
                    })

    # Sort by start time
    melodic_notes.sort(key=lambda n: n['start_tick'])
    keyswitches.sort(key=lambda n: n['tick'])

    # Convert ticks to seconds
    tick_to_sec = tempo / (tpb * 1_000_000)
    for n in melodic_notes:
        n['start_sec'] = n['start_tick'] * tick_to_sec
        n['dur_sec'] = n['dur_ticks'] * tick_to_sec

    return melodic_notes, keyswitches, tempo, tpb


def analyze_file(filepath):
    """Compute all metrics for a single MIDI file."""
    melodic_notes, keyswitches, tempo, tpb = parse_midi(filepath)

    result = {'filename': os.path.basename(filepath)}

    if len(melodic_notes) < 3:
        # Too few notes to analyze meaningfully
        result['valid'] = False
        return result

    result['valid'] = True

    pitches = [n['pitch'] for n in melodic_notes]
    starts_sec = [n['start_sec'] for n in melodic_notes]
    durs_sec = [n['dur_sec'] for n in melodic_notes]

    # Basic metrics
    total_dur = max(n['start_sec'] + n['dur_sec'] for n in melodic_notes) - min(n['start_sec'] for n in melodic_notes)
    result['total_duration'] = total_dur
    result['num_notes'] = len(melodic_notes)
    result['note_density'] = len(melodic_notes) / total_dur if total_dur > 0 else 0

    # Pitch range
    result['pitch_min'] = min(pitches)
    result['pitch_max'] = max(pitches)
    result['pitch_range'] = max(pitches) - min(pitches)

    # Pitch classes
    pcs = [p % 12 for p in pitches]
    unique_pcs = set(pcs)
    result['unique_pcs'] = unique_pcs
    result['unique_pc_names'] = sorted([PC_NAMES[pc] for pc in unique_pcs])
    result['num_unique_pcs'] = len(unique_pcs)

    # D-pentatonic percentage
    penta_count = sum(1 for pc in pcs if pc in D_PENTA)
    result['penta_pct'] = penta_count / len(pcs) * 100

    # Intervals
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    if intervals:
        leaps = sum(1 for iv in intervals if iv > 5)
        steps = sum(1 for iv in intervals if 1 <= iv <= 3)
        result['leap_pct'] = leaps / len(intervals) * 100
        result['step_pct'] = steps / len(intervals) * 100
        result['avg_interval'] = np.mean(intervals)
    else:
        result['leap_pct'] = 0
        result['step_pct'] = 0
        result['avg_interval'] = 0

    # Average note duration
    result['avg_note_dur'] = np.mean(durs_sec)

    # IOI (inter-onset interval) stats
    ioi = [starts_sec[i+1] - starts_sec[i] for i in range(len(starts_sec)-1)]
    ioi = [x for x in ioi if x > 0]
    if ioi:
        result['ioi_std'] = np.std(ioi)
        result['ioi_mean'] = np.mean(ioi)
        result['ioi_cv'] = np.std(ioi) / np.mean(ioi) if np.mean(ioi) > 0 else 0
    else:
        result['ioi_std'] = 0
        result['ioi_mean'] = 0
        result['ioi_cv'] = 0

    # Density variation across 4-bar windows
    # Approximate 4 bars as ~8 seconds at 120 BPM (or scaled by tempo)
    bpm = 60_000_000 / tempo
    bar_dur = 4 * (60 / bpm) * 4  # 4 bars of 4/4
    if total_dur > bar_dur:
        n_windows = max(1, int(total_dur / bar_dur))
        window_counts = [0] * n_windows
        for s in starts_sec:
            idx = min(int((s - starts_sec[0]) / bar_dur), n_windows - 1)
            window_counts[idx] += 1
        wc = np.array(window_counts, dtype=float)
        result['density_cv'] = np.std(wc) / np.mean(wc) if np.mean(wc) > 0 else 0
    else:
        result['density_cv'] = 0

    # Phrase structure: density in thirds of the piece
    third = total_dur / 3
    start_offset = starts_sec[0]
    thirds_counts = [0, 0, 0]
    for s in starts_sec:
        idx = min(int((s - start_offset) / third), 2)
        thirds_counts[idx] += 1
    result['thirds_counts'] = thirds_counts
    # Check for arc shape (beginning moderate, middle dense, end moderate/sparse)
    # or any clear structure
    tc = thirds_counts
    has_structure = (tc[0] != tc[1] or tc[1] != tc[2]) and max(tc) > 1.3 * min(tc)
    result['has_structure'] = has_structure

    # Melodic contour repetition (3-4 note patterns)
    # Encode contour as up/down/same
    contour = []
    for i in range(len(pitches) - 1):
        diff = pitches[i+1] - pitches[i]
        if diff > 0:
            contour.append('U')
        elif diff < 0:
            contour.append('D')
        else:
            contour.append('S')

    # Count 3-gram and 4-gram patterns
    pattern_counts_3 = Counter()
    pattern_counts_4 = Counter()
    for i in range(len(contour) - 2):
        pattern_counts_3[tuple(contour[i:i+3])] += 1
    for i in range(len(contour) - 3):
        pattern_counts_4[tuple(contour[i:i+4])] += 1

    # Repetition score: fraction of patterns that appear more than once
    total_3 = sum(pattern_counts_3.values())
    repeated_3 = sum(v for v in pattern_counts_3.values() if v > 1)
    total_4 = sum(pattern_counts_4.values())
    repeated_4 = sum(v for v in pattern_counts_4.values() if v > 1)

    rep_3 = repeated_3 / total_3 if total_3 > 0 else 0
    rep_4 = repeated_4 / total_4 if total_4 > 0 else 0
    result['repetition_score'] = (rep_3 + rep_4) / 2

    # Keyswitches / articulations
    ks_pitches = set(k['pitch'] for k in keyswitches)
    result['num_ks_events'] = len(keyswitches)
    result['unique_ks'] = len(ks_pitches)
    result['ks_types'] = sorted([KS_ARTICULATIONS.get(p, f"ks_{p}") for p in ks_pitches])

    return result


def compute_score(r):
    """Compute composite authenticity score (0-100)."""
    if not r.get('valid', False):
        return 0.0

    score = 0.0

    # 1. Pentatonic (MANDATORY gate) - 25 points
    if r['penta_pct'] < 95:
        return 0.0  # Hard fail
    penta_score = 25.0 if r['penta_pct'] == 100 else 25.0 * (r['penta_pct'] - 95) / 5
    score += penta_score

    # 2. Stepwise motion (step% > 60% ideal) - 15 points
    step_score = min(r['step_pct'] / 60, 1.0) * 15
    score += step_score

    # 3. Low leaps (< 10% ideal) - 10 points
    if r['leap_pct'] <= 10:
        leap_score = 10.0
    elif r['leap_pct'] <= 20:
        leap_score = 10.0 * (1 - (r['leap_pct'] - 10) / 10)
    else:
        leap_score = 0.0
    score += leap_score

    # 4. Note density (2-6 n/s ideal) - 10 points
    d = r['note_density']
    if 2 <= d <= 6:
        density_score = 10.0
    elif 1 <= d < 2:
        density_score = 10.0 * (d - 1)
    elif 6 < d <= 10:
        density_score = 10.0 * (1 - (d - 6) / 4)
    else:
        density_score = 0.0
    score += density_score

    # 5. Duration (15-60s ideal) - 8 points
    dur = r['total_duration']
    if 15 <= dur <= 60:
        dur_score = 8.0
    elif 10 <= dur < 15:
        dur_score = 8.0 * (dur - 10) / 5
    elif 60 < dur <= 90:
        dur_score = 8.0 * (1 - (dur - 60) / 30)
    elif 5 <= dur < 10:
        dur_score = 4.0 * (dur - 5) / 5
    else:
        dur_score = 0.0
    score += dur_score

    # 6. Rhythmic variety (IOI CV 0.3-1.0 ideal) - 8 points
    cv = r['ioi_cv']
    if 0.3 <= cv <= 1.0:
        rhythm_score = 8.0
    elif 0.1 <= cv < 0.3:
        rhythm_score = 8.0 * (cv - 0.1) / 0.2
    elif 1.0 < cv <= 1.5:
        rhythm_score = 8.0 * (1 - (cv - 1.0) / 0.5)
    else:
        rhythm_score = max(0, 2.0)  # some base if not chaotic
    score += rhythm_score

    # 7. Melodic repetition (0.3-0.8 ideal) - 8 points
    rep = r['repetition_score']
    if 0.3 <= rep <= 0.8:
        rep_score = 8.0
    elif 0.15 <= rep < 0.3:
        rep_score = 8.0 * (rep - 0.15) / 0.15
    elif 0.8 < rep <= 0.95:
        rep_score = 8.0 * (1 - (rep - 0.8) / 0.15)
    else:
        rep_score = max(0, 2.0)
    score += rep_score

    # 8. Phrase structure / density variation (CV 0.2-0.8) - 6 points
    dcv = r['density_cv']
    if 0.2 <= dcv <= 0.8:
        struct_score = 6.0
    elif 0.1 <= dcv < 0.2:
        struct_score = 6.0 * (dcv - 0.1) / 0.1
    elif 0.8 < dcv <= 1.2:
        struct_score = 6.0 * (1 - (dcv - 0.8) / 0.4)
    else:
        struct_score = 0.0
    score += struct_score

    # 9. Articulation variety - 5 points
    n_art = r['unique_ks']
    if n_art >= 4:
        art_score = 5.0
    elif n_art >= 2:
        art_score = 5.0 * (n_art - 1) / 3
    elif n_art == 1:
        art_score = 1.0
    else:
        art_score = 0.0
    score += art_score

    # 10. Pitch range (15-30 semitones ideal) - 5 points
    pr = r['pitch_range']
    if 15 <= pr <= 30:
        range_score = 5.0
    elif 10 <= pr < 15:
        range_score = 5.0 * (pr - 10) / 5
    elif 30 < pr <= 40:
        range_score = 5.0 * (1 - (pr - 30) / 10)
    else:
        range_score = 0.0
    score += range_score

    return round(score, 1)


def main():
    files = sorted(glob.glob(os.path.join(MIDI_DIR, "*_ks.mid")))
    print(f"Found {len(files)} files to analyze.\n")

    results = []
    for f in files:
        r = analyze_file(f)
        r['score'] = compute_score(r)
        results.append(r)

    # Sort by score descending
    results.sort(key=lambda r: r['score'], reverse=True)

    # Print ranked table
    print("=" * 140)
    print(f"{'Rank':>4} {'File':<24} {'Score':>5} {'Dur(s)':>7} {'Notes':>5} {'N/s':>5} "
          f"{'Range':>5} {'Penta%':>7} {'Step%':>6} {'Leap%':>6} {'IOI_CV':>6} "
          f"{'RepSc':>5} {'DenCV':>5} {'KS#':>3} {'PCs':<20}")
    print("-" * 140)

    for i, r in enumerate(results, 1):
        if not r.get('valid', False):
            print(f"{i:>4} {r['filename']:<24} {'N/A':>5}  (too few notes)")
            continue

        pc_str = ",".join(r['unique_pc_names'])
        print(f"{i:>4} {r['filename']:<24} {r['score']:>5.1f} {r['total_duration']:>7.1f} "
              f"{r['num_notes']:>5} {r['note_density']:>5.1f} {r['pitch_range']:>5} "
              f"{r['penta_pct']:>6.1f}% {r['step_pct']:>5.1f}% {r['leap_pct']:>5.1f}% "
              f"{r['ioi_cv']:>6.2f} {r['repetition_score']:>5.2f} {r['density_cv']:>5.2f} "
              f"{r['unique_ks']:>3} {pc_str:<20}")

    print("=" * 140)

    # Top 10 detailed analysis
    top10 = [r for r in results if r.get('valid', False)][:10]

    print(f"\n{'='*80}")
    print("TOP 10 MOST AUTHENTIC FILES - DETAILED ANALYSIS")
    print(f"{'='*80}\n")

    for rank, r in enumerate(top10, 1):
        print(f"--- #{rank}: {r['filename']}  (Score: {r['score']}/100) ---")
        print(f"  Duration: {r['total_duration']:.1f}s | Notes: {r['num_notes']} | "
              f"Density: {r['note_density']:.2f} n/s")
        print(f"  Pitch range: {r['pitch_range']} semitones "
              f"(MIDI {r['pitch_min']}-{r['pitch_max']})")
        print(f"  Pitch classes: {', '.join(r['unique_pc_names'])} "
              f"({r['num_unique_pcs']} unique)")
        print(f"  D-pentatonic: {r['penta_pct']:.1f}%")
        print(f"  Motion: step={r['step_pct']:.1f}%, leap={r['leap_pct']:.1f}%, "
              f"avg interval={r['avg_interval']:.1f} semitones")
        print(f"  Avg note dur: {r['avg_note_dur']:.3f}s | IOI CV: {r['ioi_cv']:.2f}")
        print(f"  Rhythmic variety (IOI std): {r['ioi_std']:.3f}s")
        print(f"  Phrase density (thirds): {r['thirds_counts']} "
              f"| Structure: {'Yes' if r['has_structure'] else 'No'}")
        print(f"  Density CV (4-bar windows): {r['density_cv']:.2f}")
        print(f"  Repetition score: {r['repetition_score']:.2f}")
        print(f"  Articulations ({r['unique_ks']}): {', '.join(r['ks_types'])}")

        # Qualitative assessment
        strengths = []
        weaknesses = []

        if r['penta_pct'] == 100:
            strengths.append("Perfect D-pentatonic adherence")
        elif r['penta_pct'] >= 95:
            strengths.append(f"Near-perfect pentatonic ({r['penta_pct']:.1f}%)")

        if r['step_pct'] >= 60:
            strengths.append(f"Strong conjunct motion ({r['step_pct']:.0f}% stepwise)")
        elif r['step_pct'] >= 45:
            weaknesses.append(f"Moderate stepwise motion ({r['step_pct']:.0f}%)")
        else:
            weaknesses.append(f"Low stepwise motion ({r['step_pct']:.0f}%)")

        if r['leap_pct'] <= 10:
            strengths.append(f"Appropriately few leaps ({r['leap_pct']:.0f}%)")
        elif r['leap_pct'] <= 20:
            weaknesses.append(f"Somewhat leapy ({r['leap_pct']:.0f}%)")
        else:
            weaknesses.append(f"Too many leaps ({r['leap_pct']:.0f}%)")

        if 2 <= r['note_density'] <= 6:
            strengths.append(f"Ideal guzheng density ({r['note_density']:.1f} n/s)")
        elif r['note_density'] < 2:
            weaknesses.append(f"Sparse ({r['note_density']:.1f} n/s)")
        else:
            weaknesses.append(f"Too dense ({r['note_density']:.1f} n/s)")

        if 15 <= r['total_duration'] <= 60:
            strengths.append(f"Good duration ({r['total_duration']:.0f}s)")
        elif r['total_duration'] < 15:
            weaknesses.append(f"Short ({r['total_duration']:.0f}s)")
        else:
            weaknesses.append(f"Long ({r['total_duration']:.0f}s)")

        if 0.3 <= r['ioi_cv'] <= 1.0:
            strengths.append("Good rhythmic variety")
        elif r['ioi_cv'] < 0.3:
            weaknesses.append("Rhythmically monotonous")
        else:
            weaknesses.append("Rhythmically chaotic")

        if 0.3 <= r['repetition_score'] <= 0.8:
            strengths.append(f"Good motif development (rep={r['repetition_score']:.2f})")
        elif r['repetition_score'] > 0.8:
            weaknesses.append("Overly repetitive")
        else:
            weaknesses.append("Lacks motivic repetition")

        if r['has_structure']:
            strengths.append("Clear phrase structure")
        else:
            weaknesses.append("Flat phrase structure")

        if r['unique_ks'] >= 3:
            strengths.append(f"Rich articulations ({', '.join(r['ks_types'])})")
        elif r['unique_ks'] >= 2:
            strengths.append(f"Some articulation variety ({', '.join(r['ks_types'])})")
        else:
            weaknesses.append("Limited articulations")

        if 15 <= r['pitch_range'] <= 30:
            strengths.append(f"Idiomatic range ({r['pitch_range']} semitones)")
        elif r['pitch_range'] < 15:
            weaknesses.append(f"Narrow range ({r['pitch_range']} semitones)")
        else:
            weaknesses.append(f"Wide range ({r['pitch_range']} semitones)")

        print(f"  STRENGTHS: {'; '.join(strengths)}")
        if weaknesses:
            print(f"  WEAKNESSES: {'; '.join(weaknesses)}")
        print()


if __name__ == '__main__':
    main()

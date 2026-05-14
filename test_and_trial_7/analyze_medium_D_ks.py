#!/usr/bin/env python3
"""Analyze MIDI files in medium_D_ks/ for musical coherence."""

import os
import glob
import math
import mido
import numpy as np
from collections import Counter

MIDI_DIR = "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_ks"
KEYSWITCH_THRESHOLD = 36  # pitches below this are keyswitches

def analyze_midi(filepath):
    mid = mido.MidiFile(filepath)

    # Collect all note events with absolute times in seconds
    notes = []  # list of (onset_sec, offset_sec, pitch, velocity)

    for track in mid.tracks:
        abs_time = 0  # in ticks
        pending = {}  # pitch -> (onset_tick, velocity)

        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note >= KEYSWITCH_THRESHOLD:
                    pending[msg.note] = (abs_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in pending:
                    onset_tick, vel = pending.pop(msg.note)
                    onset_sec = mido.tick2second(onset_tick, mid.ticks_per_beat, 500000)
                    offset_sec = mido.tick2second(abs_time, mid.ticks_per_beat, 500000)
                    # Check for tempo changes - use a simpler approach
                    notes.append((onset_sec, offset_sec, msg.note, vel))

    # Re-parse with proper tempo handling
    notes = []
    for track in mid.tracks:
        # Build tempo map from all tracks
        pass

    # Better approach: use mid.length and compute times properly
    # Rebuild with tempo-aware timing
    tempo_map = []  # (tick, tempo)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))

    if not tempo_map:
        tempo_map = [(0, 500000)]  # default 120 BPM
    tempo_map.sort(key=lambda x: x[0])

    def tick_to_sec(tick):
        """Convert tick to seconds using tempo map."""
        sec = 0.0
        prev_tick = 0
        prev_tempo = 500000
        for t_tick, t_tempo in tempo_map:
            if t_tick >= tick:
                break
            sec += mido.tick2second(t_tick - prev_tick, mid.ticks_per_beat, prev_tempo)
            prev_tick = t_tick
            prev_tempo = t_tempo
        sec += mido.tick2second(tick - prev_tick, mid.ticks_per_beat, prev_tempo)
        return sec

    notes = []
    for track in mid.tracks:
        abs_tick = 0
        pending = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note >= KEYSWITCH_THRESHOLD:
                    pending[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in pending:
                    onset_tick, vel = pending.pop(msg.note)
                    onset_sec = tick_to_sec(onset_tick)
                    offset_sec = tick_to_sec(abs_tick)
                    notes.append((onset_sec, offset_sec, msg.note, vel))

    if not notes:
        return None

    notes.sort(key=lambda x: x[0])

    # Basic metrics
    total_duration = max(n[1] for n in notes) - min(n[0] for n in notes)
    if total_duration <= 0:
        total_duration = 0.01

    num_notes = len(notes)
    density = num_notes / total_duration

    pitches = [n[2] for n in notes]
    pitch_min = min(pitches)
    pitch_max = max(pitches)
    pitch_range = pitch_max - pitch_min

    pitch_classes = set(p % 12 for p in pitches)
    num_pitch_classes = len(pitch_classes)

    # Intervals and leaps
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    leaps = sum(1 for iv in intervals if iv > 5)
    leap_pct = (leaps / len(intervals) * 100) if intervals else 0

    # Note durations
    note_durs = [n[1] - n[0] for n in notes]
    avg_note_dur = np.mean(note_durs)

    # Gaps/rests between consecutive notes
    onsets = [n[0] for n in notes]
    offsets = [n[1] for n in notes]
    gaps = []
    for i in range(len(notes) - 1):
        gap = onsets[i+1] - offsets[i]
        if gap > 0:
            gaps.append(gap)
    longest_gap = max(gaps) if gaps else 0

    # Inter-onset intervals
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    iois = [x for x in iois if x > 0]
    ioi_std = np.std(iois) if len(iois) > 1 else 0
    ioi_mean = np.mean(iois) if iois else 0
    ioi_cv = (ioi_std / ioi_mean) if ioi_mean > 0 else 0  # coefficient of variation

    # Phrase structure detection: split into 2-second windows, check density variation
    window_size = 2.0
    start_time = min(n[0] for n in notes)
    end_time = max(n[1] for n in notes)
    window_densities = []
    t = start_time
    while t < end_time:
        count = sum(1 for n in notes if n[0] >= t and n[0] < t + window_size)
        window_densities.append(count / window_size)
        t += window_size

    if len(window_densities) > 2:
        density_std = np.std(window_densities)
        density_mean = np.mean(window_densities)
        density_cv = density_std / density_mean if density_mean > 0 else 0
        has_phrases = density_cv > 0.4  # reasonable variation suggests phrasing
    else:
        density_cv = 0
        has_phrases = False

    # Melodic repetition score: count recurring 3-note and 4-note pitch-class patterns
    pc_seq = [p % 12 for p in pitches]

    trigrams = Counter()
    for i in range(len(pc_seq) - 2):
        trigrams[tuple(pc_seq[i:i+3])] += 1

    quadgrams = Counter()
    for i in range(len(pc_seq) - 3):
        quadgrams[tuple(pc_seq[i:i+4])] += 1

    # Repetition = fraction of n-grams that appear more than once
    repeated_tri = sum(c for _, c in trigrams.items() if c > 1)
    total_tri = sum(trigrams.values())
    repeated_quad = sum(c for _, c in quadgrams.items() if c > 1)
    total_quad = sum(quadgrams.values())

    rep_score_tri = repeated_tri / total_tri if total_tri > 0 else 0
    rep_score_quad = repeated_quad / total_quad if total_quad > 0 else 0
    repetition_score = 0.5 * rep_score_tri + 0.5 * rep_score_quad

    # Also track interval-based repetition (relative patterns)
    interval_seq = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    int_trigrams = Counter()
    for i in range(len(interval_seq) - 2):
        int_trigrams[tuple(interval_seq[i:i+3])] += 1
    repeated_int_tri = sum(c for _, c in int_trigrams.items() if c > 1)
    total_int_tri = sum(int_trigrams.values())
    int_rep_score = repeated_int_tri / total_int_tri if total_int_tri > 0 else 0

    repetition_score = 0.4 * rep_score_tri + 0.3 * rep_score_quad + 0.3 * int_rep_score

    return {
        'filename': os.path.basename(filepath),
        'duration': total_duration,
        'num_notes': num_notes,
        'density': density,
        'pitch_min': pitch_min,
        'pitch_max': pitch_max,
        'pitch_range': pitch_range,
        'num_pitch_classes': num_pitch_classes,
        'leap_pct': leap_pct,
        'avg_note_dur': avg_note_dur,
        'longest_gap': longest_gap,
        'ioi_std': ioi_std,
        'ioi_cv': ioi_cv,
        'density_cv': density_cv,
        'has_phrases': has_phrases,
        'repetition_score': repetition_score,
    }


def compute_musicality(r):
    """Composite musicality score (0-100)."""
    score = 0.0

    # 1. Density score (weight: 20) - 3-8 notes/sec is ideal
    d = r['density']
    if 3 <= d <= 8:
        density_score = 1.0
    elif d < 3:
        density_score = max(0, d / 3.0)
    else:  # d > 8
        density_score = max(0, 1.0 - (d - 8) / 8.0)
    score += 20 * density_score

    # 2. Leap percentage (weight: 15) - lower is better, < 15% ideal
    lp = r['leap_pct']
    if lp <= 15:
        leap_score = 1.0
    elif lp <= 30:
        leap_score = 1.0 - (lp - 15) / 15.0
    else:
        leap_score = max(0, 0.5 - (lp - 30) / 40.0)
    score += 15 * leap_score

    # 3. Duration (weight: 10) - > 10 sec good, > 30 sec even better
    dur = r['duration']
    if dur >= 30:
        dur_score = 1.0
    elif dur >= 10:
        dur_score = 0.5 + 0.5 * (dur - 10) / 20.0
    else:
        dur_score = max(0, dur / 20.0)
    score += 10 * dur_score

    # 4. Pitch class variety (weight: 15) - 4-5 ideal for pentatonic
    pc = r['num_pitch_classes']
    if 4 <= pc <= 6:
        pc_score = 1.0
    elif pc == 3 or pc == 7:
        pc_score = 0.7
    elif pc == 2 or pc == 8:
        pc_score = 0.4
    else:
        pc_score = 0.2
    score += 15 * pc_score

    # 5. Rhythmic variety (weight: 15) - moderate IOI CV is good
    cv = r['ioi_cv']
    if 0.4 <= cv <= 1.2:
        rhythm_score = 1.0
    elif cv < 0.4:
        rhythm_score = cv / 0.4
    else:
        rhythm_score = max(0, 1.0 - (cv - 1.2) / 1.0)
    score += 15 * rhythm_score

    # 6. Phrase structure (weight: 10)
    if r['has_phrases']:
        phrase_score = 1.0
    else:
        # partial credit based on density CV
        phrase_score = min(1.0, r['density_cv'] / 0.4)
    score += 10 * phrase_score

    # 7. Repetition (weight: 15) - some repetition is good (0.2-0.6)
    rep = r['repetition_score']
    if 0.2 <= rep <= 0.6:
        rep_score = 1.0
    elif rep < 0.2:
        rep_score = rep / 0.2
    else:
        rep_score = max(0, 1.0 - (rep - 0.6) / 0.4)
    score += 15 * rep_score

    # Penalty: very long gaps suggest broken/incomplete pieces
    if r['longest_gap'] > 5.0:
        score *= max(0.5, 1.0 - (r['longest_gap'] - 5.0) / 20.0)

    return round(score, 1)


def main():
    files = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
    print(f"Found {len(files)} MIDI files in {MIDI_DIR}\n")

    results = []
    for f in files:
        r = analyze_midi(f)
        if r is None:
            print(f"WARNING: {os.path.basename(f)} has no notes (excluding keyswitches)")
            continue
        r['musicality'] = compute_musicality(r)
        results.append(r)

    # Sort by musicality score descending
    results.sort(key=lambda x: x['musicality'], reverse=True)

    # Print table
    header = (
        f"{'Rank':>4} {'Filename':<30} {'Dur(s)':>7} {'Notes':>6} "
        f"{'Dens':>6} {'PitRng':>6} {'#PC':>4} {'Leap%':>6} "
        f"{'AvgDur':>7} {'MaxGap':>7} {'IOI_CV':>7} {'DenCV':>6} "
        f"{'Phrase':>6} {'RepScr':>7} {'SCORE':>7}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        phrase_str = "Yes" if r['has_phrases'] else "No"
        print(
            f"{i:>4} {r['filename']:<30} {r['duration']:>7.1f} {r['num_notes']:>6} "
            f"{r['density']:>6.2f} {r['pitch_range']:>6} {r['num_pitch_classes']:>4} {r['leap_pct']:>6.1f} "
            f"{r['avg_note_dur']:>7.3f} {r['longest_gap']:>7.2f} {r['ioi_cv']:>7.3f} {r['density_cv']:>6.2f} "
            f"{phrase_str:>6} {r['repetition_score']:>7.3f} {r['musicality']:>7.1f}"
        )

    # Summary statistics
    scores = [r['musicality'] for r in results]
    print(f"\n--- Summary ---")
    print(f"Files analyzed: {len(results)}")
    print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"Mean score: {np.mean(scores):.1f}")
    print(f"Median score: {np.median(scores):.1f}")

    # Top tier
    top = [r for r in results if r['musicality'] >= 70]
    good = [r for r in results if 50 <= r['musicality'] < 70]
    mediocre = [r for r in results if 30 <= r['musicality'] < 50]
    poor = [r for r in results if r['musicality'] < 30]

    print(f"\nTier breakdown:")
    print(f"  Excellent (>=70): {len(top)} files")
    print(f"  Good (50-69):     {len(good)} files")
    print(f"  Mediocre (30-49): {len(mediocre)} files")
    print(f"  Poor (<30):       {len(poor)} files")

    if top:
        print(f"\nTop picks:")
        for r in top:
            print(f"  {r['filename']} (score={r['musicality']}): "
                  f"{r['num_notes']} notes, {r['duration']:.1f}s, "
                  f"density={r['density']:.1f}, leaps={r['leap_pct']:.0f}%, "
                  f"repetition={r['repetition_score']:.2f}")


if __name__ == "__main__":
    main()

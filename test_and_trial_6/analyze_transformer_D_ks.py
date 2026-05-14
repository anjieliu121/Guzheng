"""
Analyze 50 transformer-generated MIDI files (D pentatonic, ACZ4 keyswitches)
for musical coherence and quality metrics.
"""

import os
import glob
from collections import Counter
import mido
import numpy as np

MIDI_DIR = "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_6/generated/transformer_D_ks/"
KS_THRESHOLD = 36  # pitches below this are keyswitches, exclude them


def analyze_midi(filepath):
    mid = mido.MidiFile(filepath)

    # Collect note-on events with absolute time in seconds
    notes = []  # list of (onset_sec, pitch, duration_sec)

    for track in mid.tracks:
        abs_time = 0.0
        tempo = 500000  # default 120 BPM
        active = {}  # pitch -> onset_time

        for msg in track:
            # Convert delta ticks to seconds
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.note >= KS_THRESHOLD:
                    active[msg.note] = abs_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note >= KS_THRESHOLD and msg.note in active:
                    onset = active.pop(msg.note)
                    dur = abs_time - onset
                    notes.append((onset, msg.note, dur))

    if len(notes) == 0:
        return None

    notes.sort(key=lambda x: x[0])

    onsets = np.array([n[0] for n in notes])
    pitches = np.array([n[1] for n in notes])
    durations = np.array([n[2] for n in notes])

    total_duration = max(onsets[-1] + durations[-1], onsets[-1] + 0.1)
    num_notes = len(notes)
    note_density = num_notes / total_duration if total_duration > 0 else 0

    pitch_range_low = int(pitches.min())
    pitch_range_high = int(pitches.max())
    pitch_range = pitch_range_high - pitch_range_low

    pitch_classes = set(p % 12 for p in pitches)
    num_pitch_classes = len(pitch_classes)

    # Intervals and leaps
    intervals = np.abs(np.diff(pitches))
    leap_pct = (np.sum(intervals > 5) / len(intervals) * 100) if len(intervals) > 0 else 0

    avg_note_dur = float(np.mean(durations))

    # Gaps between consecutive note onsets
    ioi = np.diff(onsets)
    longest_gap = float(np.max(ioi)) if len(ioi) > 0 else 0
    ioi_std = float(np.std(ioi)) if len(ioi) > 0 else 0

    # Melodic fragment repetition (3-note and 4-note interval patterns)
    interval_seq = list(np.diff(pitches).astype(int))

    def count_repeated_patterns(seq, length):
        if len(seq) < length:
            return 0, 0
        patterns = []
        for i in range(len(seq) - length + 1):
            patterns.append(tuple(seq[i:i+length]))
        counter = Counter(patterns)
        repeated = sum(c for c in counter.values() if c > 1)
        return repeated, len(patterns)

    rep3, total3 = count_repeated_patterns(interval_seq, 3)
    rep4, total4 = count_repeated_patterns(interval_seq, 4)
    rep_ratio = 0
    if total3 > 0:
        rep_ratio = (rep3 / total3 * 0.6 + (rep4 / total4 if total4 > 0 else 0) * 0.4)

    return {
        'total_duration': total_duration,
        'num_notes': num_notes,
        'note_density': note_density,
        'pitch_range_low': pitch_range_low,
        'pitch_range_high': pitch_range_high,
        'pitch_range': pitch_range,
        'num_pitch_classes': num_pitch_classes,
        'leap_pct': leap_pct,
        'avg_note_dur': avg_note_dur,
        'longest_gap': longest_gap,
        'ioi_std': ioi_std,
        'rep_ratio': rep_ratio,
    }


def compute_score(m):
    """Composite musicality score (0-100)."""
    score = 0

    # 1. Note density: ideal 3-8 n/s (weight: 20)
    d = m['note_density']
    if 3 <= d <= 8:
        density_score = 20
    elif 2 <= d < 3 or 8 < d <= 10:
        density_score = 14
    elif 1 <= d < 2 or 10 < d <= 14:
        density_score = 8
    else:
        density_score = 2
    score += density_score

    # 2. Leap percentage: lower is better, <15% ideal (weight: 15)
    lp = m['leap_pct']
    if lp <= 10:
        leap_score = 15
    elif lp <= 15:
        leap_score = 12
    elif lp <= 25:
        leap_score = 8
    elif lp <= 35:
        leap_score = 4
    else:
        leap_score = 1
    score += leap_score

    # 3. Duration: >10s reasonable (weight: 15)
    dur = m['total_duration']
    if dur >= 30:
        dur_score = 15
    elif dur >= 20:
        dur_score = 13
    elif dur >= 10:
        dur_score = 10
    elif dur >= 5:
        dur_score = 5
    else:
        dur_score = 1
    score += dur_score

    # 4. Pitch class variety: 4-5 ideal for pentatonic (weight: 15)
    pc = m['num_pitch_classes']
    if pc == 5:
        pc_score = 15
    elif pc == 4:
        pc_score = 13
    elif pc == 3 or pc == 6:
        pc_score = 9
    elif pc == 2 or pc == 7:
        pc_score = 5
    else:
        pc_score = 2
    score += pc_score

    # 5. Rhythmic variety (IOI std dev): some variety good, not too chaotic (weight: 15)
    # Ideal range: 0.1 - 0.5s std dev
    ioi = m['ioi_std']
    if 0.1 <= ioi <= 0.5:
        rhy_score = 15
    elif 0.05 <= ioi < 0.1 or 0.5 < ioi <= 0.8:
        rhy_score = 11
    elif 0.8 < ioi <= 1.2:
        rhy_score = 7
    elif ioi < 0.05:
        rhy_score = 4  # too mechanical
    else:
        rhy_score = 3  # too erratic
    score += rhy_score

    # 6. Melodic repetition (some = structure): ideal 0.1-0.4 ratio (weight: 10)
    rr = m['rep_ratio']
    if 0.1 <= rr <= 0.4:
        rep_score = 10
    elif 0.05 <= rr < 0.1 or 0.4 < rr <= 0.6:
        rep_score = 7
    elif rr > 0.6:
        rep_score = 4  # too repetitive
    else:
        rep_score = 3  # no structure
    score += rep_score

    # 7. Longest gap penalty: large gaps break coherence (weight: 10)
    gap = m['longest_gap']
    if gap <= 1.0:
        gap_score = 10
    elif gap <= 2.0:
        gap_score = 8
    elif gap <= 4.0:
        gap_score = 5
    else:
        gap_score = 2
    score += gap_score

    return score


def main():
    files = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
    print(f"Found {len(files)} MIDI files\n")

    results = []
    for f in files:
        name = os.path.basename(f)
        metrics = analyze_midi(f)
        if metrics is None:
            print(f"WARNING: {name} has no melodic notes, skipping")
            continue
        metrics['score'] = compute_score(metrics)
        metrics['name'] = name
        results.append(metrics)

    # Sort by score descending
    results.sort(key=lambda x: -x['score'])

    # Print full table
    header = (
        f"{'Rank':>4}  {'File':<30}  {'Score':>5}  {'Dur(s)':>6}  {'Notes':>5}  "
        f"{'N/s':>5}  {'PitchLo':>7}  {'PitchHi':>7}  {'Range':>5}  {'PC#':>3}  "
        f"{'Leap%':>5}  {'AvgDur':>6}  {'MaxGap':>6}  {'IOI_SD':>6}  {'RepR':>5}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        print(
            f"{i+1:>4}  {r['name']:<30}  {r['score']:>5}  {r['total_duration']:>6.1f}  "
            f"{r['num_notes']:>5}  {r['note_density']:>5.1f}  {r['pitch_range_low']:>7}  "
            f"{r['pitch_range_high']:>7}  {r['pitch_range']:>5}  {r['num_pitch_classes']:>3}  "
            f"{r['leap_pct']:>5.1f}  {r['avg_note_dur']:>6.3f}  {r['longest_gap']:>6.2f}  "
            f"{r['ioi_std']:>6.3f}  {r['rep_ratio']:>5.3f}"
        )

    print("\n" + "=" * 80)
    print("TOP 5 FILES:")
    print("=" * 80)
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']}")
        print(f"       Score: {r['score']}/100")
        print(f"       Duration: {r['total_duration']:.1f}s  |  Notes: {r['num_notes']}  |  Density: {r['note_density']:.1f} n/s")
        print(f"       Pitch range: {r['pitch_range_low']}-{r['pitch_range_high']} ({r['pitch_range']} semitones)  |  Pitch classes: {r['num_pitch_classes']}")
        print(f"       Leaps: {r['leap_pct']:.1f}%  |  Avg note dur: {r['avg_note_dur']:.3f}s  |  Max gap: {r['longest_gap']:.2f}s")
        print(f"       Rhythmic variety (IOI SD): {r['ioi_std']:.3f}s  |  Repetition ratio: {r['rep_ratio']:.3f}")

    # Summary stats
    scores = [r['score'] for r in results]
    print(f"\n{'='*80}")
    print(f"SUMMARY: mean score = {np.mean(scores):.1f}, median = {np.median(scores):.1f}, "
          f"min = {min(scores)}, max = {max(scores)}")

    # Constrained vs unconstrained comparison
    c_scores = [r['score'] for r in results if r['name'].startswith('constrained')]
    u_scores = [r['score'] for r in results if r['name'].startswith('unconstrained')]
    print(f"Constrained mean: {np.mean(c_scores):.1f}  |  Unconstrained mean: {np.mean(u_scores):.1f}")


if __name__ == '__main__':
    main()

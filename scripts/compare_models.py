"""
Compare two guzheng music generation models by analyzing their MIDI outputs.
Model 1: NotaGen medium (244M params, fine-tuned)
Model 2: Decoder-only transformer (3.5M params, trained from scratch)
"""

import os
import glob
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu, entropy
import mido

# ── Configuration ──────────────────────────────────────────────────────────

MODEL1_DIRS = [
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_batch2_ks/",
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_ks/",
]
MODEL2_DIRS = [
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_6/generated/transformer_D_ks/",
]

MODEL1_NAME = "NotaGen-medium (244M)"
MODEL2_NAME = "Transformer (3.5M)"

KS_THRESHOLD = 36          # pitches below this are keyswitches
D_PENTATONIC = {2, 4, 6, 9, 11}  # D E F# A B  (pitch classes)

# ── MIDI analysis ──────────────────────────────────────────────────────────

def analyze_midi(filepath):
    """Parse a MIDI file and return a dict of metrics."""
    mid = mido.MidiFile(filepath)

    # Collect note events (absolute time in seconds)
    notes = []  # list of (onset_sec, offset_sec, pitch)
    for track in mid.tracks:
        abs_time = 0
        active = {}  # pitch -> onset_time
        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat,
                                         mido.bpm2tempo(120))  # default tempo
        # Redo with proper tempo map
    # Build a tempo map first
    tempo_map = []  # (abs_tick, tempo)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))
    if not tempo_map:
        tempo_map = [(0, mido.bpm2tempo(120))]
    tempo_map.sort()

    def tick_to_sec(tick):
        """Convert absolute tick to seconds using the tempo map."""
        sec = 0.0
        prev_tick = 0
        prev_tempo = tempo_map[0][1]
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
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note >= KS_THRESHOLD:
                    active[msg.note] = abs_tick
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    onset_tick = active.pop(msg.note)
                    if msg.note >= KS_THRESHOLD:
                        notes.append((tick_to_sec(onset_tick),
                                      tick_to_sec(abs_tick),
                                      msg.note))
        # Close any still-active notes
        for pitch, onset_tick in active.items():
            if pitch >= KS_THRESHOLD:
                notes.append((tick_to_sec(onset_tick), tick_to_sec(abs_tick), pitch))

    if len(notes) < 2:
        return None  # skip degenerate files

    notes.sort(key=lambda x: x[0])
    onsets = np.array([n[0] for n in notes])
    offsets = np.array([n[1] for n in notes])
    pitches = np.array([n[2] for n in notes])
    durations_note = offsets - onsets

    duration_sec = max(offsets) - min(onsets)
    if duration_sec <= 0:
        return None

    note_count = len(notes)
    density = note_count / duration_sec

    pitch_range = int(pitches.max() - pitches.min())
    pitch_classes = set(p % 12 for p in pitches)
    unique_pc = len(pitch_classes)
    pc_list = sorted(pitch_classes)

    # D-pentatonic %
    in_penta = sum(1 for p in pitches if p % 12 in D_PENTATONIC)
    penta_pct = in_penta / note_count * 100

    # Intervals (melodic, consecutive notes by onset)
    intervals = np.abs(np.diff(pitches))
    mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0
    leap_pct = float(np.sum(intervals > 5) / len(intervals) * 100) if len(intervals) > 0 else 0
    step_pct = float(np.sum((intervals >= 1) & (intervals <= 3)) / len(intervals) * 100) if len(intervals) > 0 else 0
    unison_pct = float(np.sum(intervals == 0) / len(intervals) * 100) if len(intervals) > 0 else 0

    # IOI
    ioi = np.diff(onsets)
    ioi = ioi[ioi >= 0]
    if len(ioi) > 0:
        mean_ioi = float(np.mean(ioi))
        std_ioi = float(np.std(ioi))
        cv_ioi = std_ioi / mean_ioi if mean_ioi > 0 else 0
    else:
        mean_ioi = std_ioi = cv_ioi = 0

    # Note duration
    mean_note_dur = float(np.mean(durations_note))
    std_note_dur = float(np.std(durations_note))

    # Repetition: 3-note pitch-interval contours
    if len(intervals) >= 2:
        contours = []
        for i in range(len(intervals) - 1):
            # direction: up(+1), down(-1), same(0) for two consecutive intervals
            d1 = int(np.sign(np.diff(pitches)[i])) if i < len(np.diff(pitches)) else 0
            d2 = int(np.sign(np.diff(pitches)[i+1])) if i+1 < len(np.diff(pitches)) else 0
            s1 = min(int(intervals[i]), 12)
            s2 = min(int(intervals[i+1]), 12) if i+1 < len(intervals) else 0
            contours.append((d1, s1, d2, s2))
        contour_counts = Counter(contours)
        repeated = sum(1 for c in contours if contour_counts[c] > 1)
        repetition_pct = repeated / len(contours) * 100
    else:
        repetition_pct = 0

    # Longest gap (rest)
    # A gap exists when the next onset is after all previous offsets have ended
    # Simplified: gap between consecutive notes = next_onset - prev_offset (if positive)
    gaps = []
    for i in range(len(notes) - 1):
        gap = notes[i+1][0] - notes[i][1]
        if gap > 0:
            gaps.append(gap)
    longest_gap = max(gaps) if gaps else 0

    # Pitch entropy (Shannon, base 2)
    pc_counts = Counter(p % 12 for p in pitches)
    total = sum(pc_counts.values())
    probs = np.array([pc_counts[pc] / total for pc in range(12) if pc in pc_counts])
    pitch_entropy = float(entropy(probs, base=2))

    return {
        'duration': duration_sec,
        'note_count': note_count,
        'density': density,
        'pitch_range': pitch_range,
        'unique_pc': unique_pc,
        'pc_list': pc_list,
        'penta_pct': penta_pct,
        'mean_interval': mean_interval,
        'leap_pct': leap_pct,
        'step_pct': step_pct,
        'unison_pct': unison_pct,
        'mean_ioi': mean_ioi,
        'std_ioi': std_ioi,
        'cv_ioi': cv_ioi,
        'mean_note_dur': mean_note_dur,
        'std_note_dur': std_note_dur,
        'repetition_pct': repetition_pct,
        'longest_gap': longest_gap,
        'pitch_entropy': pitch_entropy,
    }


def collect_files(dirs):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*.mid'))))
    return files


def analyze_all(files, label):
    results = []
    skipped = 0
    for f in files:
        r = analyze_midi(f)
        if r is None:
            skipped += 1
        else:
            results.append(r)
    print(f"  {label}: {len(results)} files analyzed, {skipped} skipped (degenerate)")
    return results


# ── Main ───────────────────────────────────────────────────────────────────

METRICS = [
    ('duration',       'Duration (s)',          '{:.1f}'),
    ('note_count',     'Note count',            '{:.0f}'),
    ('density',        'Note density (n/s)',     '{:.2f}'),
    ('pitch_range',    'Pitch range (st)',       '{:.1f}'),
    ('unique_pc',      'Unique pitch classes',   '{:.1f}'),
    ('penta_pct',      'D-pentatonic %',         '{:.1f}'),
    ('mean_interval',  'Mean |interval| (st)',   '{:.2f}'),
    ('leap_pct',       'Leap % (>5 st)',         '{:.1f}'),
    ('step_pct',       'Step % (1-3 st)',        '{:.1f}'),
    ('unison_pct',     'Unison % (0 st)',        '{:.1f}'),
    ('mean_ioi',       'Mean IOI (s)',           '{:.3f}'),
    ('std_ioi',        'Std IOI (s)',            '{:.3f}'),
    ('cv_ioi',         'IOI CV',                 '{:.3f}'),
    ('mean_note_dur',  'Mean note dur (s)',      '{:.3f}'),
    ('std_note_dur',   'Std note dur (s)',       '{:.3f}'),
    ('repetition_pct', 'Repetition %',           '{:.1f}'),
    ('longest_gap',    'Longest gap (s)',         '{:.2f}'),
    ('pitch_entropy',  'Pitch entropy (bits)',   '{:.3f}'),
]


def main():
    print("=" * 80)
    print("GUZHENG GENERATION MODEL COMPARISON")
    print("=" * 80)

    print(f"\nModel 1: {MODEL1_NAME}")
    print(f"Model 2: {MODEL2_NAME}\n")

    files1 = collect_files(MODEL1_DIRS)
    files2 = collect_files(MODEL2_DIRS)
    print(f"Files found: Model 1 = {len(files1)}, Model 2 = {len(files2)}")

    print("\nAnalyzing...")
    res1 = analyze_all(files1, MODEL1_NAME)
    res2 = analyze_all(files2, MODEL2_NAME)

    if not res1 or not res2:
        print("ERROR: One model has no valid results.")
        return

    # ── Summary statistics table ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (mean +/- std)")
    print("=" * 80)

    header = f"{'Metric':<25s} | {'Model 1 (NotaGen)':>22s} | {'Model 2 (Transf.)':>22s} | {'p-value':>10s} | {'Sig.':>4s}"
    print(header)
    print("-" * len(header))

    for key, label, fmt in METRICS:
        vals1 = np.array([r[key] for r in res1])
        vals2 = np.array([r[key] for r in res2])

        m1, s1 = np.mean(vals1), np.std(vals1)
        m2, s2 = np.mean(vals2), np.std(vals2)

        str1 = fmt.format(m1) + " +/- " + fmt.format(s1)
        str2 = fmt.format(m2) + " +/- " + fmt.format(s2)

        # Mann-Whitney U
        try:
            stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
        except ValueError:
            pval = 1.0

        sig = ""
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"

        print(f"{label:<25s} | {str1:>22s} | {str2:>22s} | {pval:>10.4f} | {sig:>4s}")

    print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001 (Mann-Whitney U, two-sided)")

    # ── Pitch class distributions ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("PITCH CLASS USAGE (% of files using each pitch class)")
    print("=" * 80)
    pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(f"{'PC':<5s} {'Name':<5s} {'Penta?':<7s} {'Model 1 %':>10s} {'Model 2 %':>10s}")
    print("-" * 40)
    for pc in range(12):
        count1 = sum(1 for r in res1 if pc in r['pc_list'])
        count2 = sum(1 for r in res2 if pc in r['pc_list'])
        pct1 = count1 / len(res1) * 100
        pct2 = count2 / len(res2) * 100
        penta = "YES" if pc in D_PENTATONIC else ""
        print(f"{pc:<5d} {pc_names[pc]:<5s} {penta:<7s} {pct1:>9.1f}% {pct2:>9.1f}%")

    # ── Qualitative assessment ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("QUALITATIVE ASSESSMENT")
    print("=" * 80)

    criteria = [
        ('penta_pct',    'D-pentatonic adherence', '~100%',         95, 100,  True),
        ('step_pct',     'Stepwise motion',        'high',          40, 70,   True),
        ('leap_pct',     'Leap rate',              '<15%',           0, 15,   False),
        ('density',      'Note density',           '2-6 n/s',        2,  6,   None),
        ('duration',     'Duration',               '15-60 s',       15, 60,   None),
        ('pitch_range',  'Pitch range',            '15-30 st',      15, 30,   None),
        ('cv_ioi',       'Rhythmic variety (CV)',   '0.3-0.8',     0.3, 0.8,  None),
        ('repetition_pct','Repetition',            'moderate',      20, 60,   None),
    ]

    for key, label, ideal, lo, hi, higher_better in criteria:
        vals1 = np.array([r[key] for r in res1])
        vals2 = np.array([r[key] for r in res2])
        m1, m2 = np.mean(vals1), np.mean(vals2)

        def grade(v, lo, hi, higher_better):
            if higher_better is None:
                # range-based
                if lo <= v <= hi:
                    return "GOOD"
                elif v < lo:
                    return f"LOW (want >={lo})"
                else:
                    return f"HIGH (want <={hi})"
            elif higher_better:
                if v >= lo:
                    return "GOOD"
                else:
                    return f"LOW (want >={lo})"
            else:
                if v <= hi:
                    return "GOOD"
                else:
                    return f"HIGH (want <={hi})"

        g1 = grade(m1, lo, hi, higher_better)
        g2 = grade(m2, lo, hi, higher_better)

        winner = ""
        if g1 == "GOOD" and g2 != "GOOD":
            winner = "<-- Model 1 better"
        elif g2 == "GOOD" and g1 != "GOOD":
            winner = "<-- Model 2 better"
        elif g1 == "GOOD" and g2 == "GOOD":
            winner = "(both good)"

        fmt_val = '{:.1f}'
        print(f"\n  {label} (ideal: {ideal}):")
        print(f"    Model 1: {fmt_val.format(m1):>8s}  {g1}")
        print(f"    Model 2: {fmt_val.format(m2):>8s}  {g2}")
        if winner:
            print(f"    {winner}")

    # ── Overall verdict ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    score1 = 0
    score2 = 0
    for key, label, ideal, lo, hi, higher_better in criteria:
        vals1 = np.array([r[key] for r in res1])
        vals2 = np.array([r[key] for r in res2])
        m1, m2 = np.mean(vals1), np.mean(vals2)

        def in_range(v):
            if higher_better is None:
                return lo <= v <= hi
            elif higher_better:
                return v >= lo
            else:
                return v <= hi

        if in_range(m1):
            score1 += 1
        if in_range(m2):
            score2 += 1

    print(f"\n  Criteria met (out of {len(criteria)}):")
    print(f"    Model 1 ({MODEL1_NAME}): {score1}/{len(criteria)}")
    print(f"    Model 2 ({MODEL2_NAME}): {score2}/{len(criteria)}")

    # File-level stats
    print(f"\n  File-level ranges:")
    for key, label, fmt in [('duration', 'Duration', '{:.1f}'), ('note_count', 'Notes', '{:.0f}'), ('density', 'Density', '{:.2f}')]:
        v1 = [r[key] for r in res1]
        v2 = [r[key] for r in res2]
        print(f"    {label}:")
        print(f"      Model 1: min={fmt.format(min(v1))}, max={fmt.format(max(v1))}, median={fmt.format(np.median(v1))}")
        print(f"      Model 2: min={fmt.format(min(v2))}, max={fmt.format(max(v2))}, median={fmt.format(np.median(v2))}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

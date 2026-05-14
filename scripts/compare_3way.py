"""
4-way comparison of guzheng generation models:
  Model 1: NotaGen medium (244M, fine-tuned hierarchical, pre-trained)
  Model 2: MIDI Decoder-only Transformer (3.5M, trained from scratch)
  Model 3: CharGPT baseline (3.46M, flat char-level GPT on ABC)
  Model 4: PatchGPT (242M, hierarchical like NotaGen, trained from scratch)
"""

import os
import glob
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu, entropy, kruskal
import mido

# ── Configuration ──────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "NotaGen-medium (244M)",
        "short": "NotaGen",
        "dirs": [
            "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_batch2_ks/",
            "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_ks/",
        ],
    },
    {
        "name": "MIDI Transformer (3.5M)",
        "short": "MIDITrans",
        "dirs": [
            "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_6/generated/transformer_D_ks/",
        ],
    },
    {
        "name": "CharGPT baseline (3.46M)",
        "short": "CharGPT",
        "dirs": [
            "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_8/generated/chargpt_D_ks/",
        ],
    },
    {
        "name": "PatchGPT (242M, no pretrain)",
        "short": "PatchGPT",
        "dirs": [
            "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_9/generated/patchgpt_D_ks/",
        ],
    },
]

KS_THRESHOLD = 36
D_PENTATONIC = {2, 4, 6, 9, 11}

# ── MIDI analysis (same as compare_models.py) ──────────────────────────────

def analyze_midi(filepath):
    mid = mido.MidiFile(filepath)
    tempo_map = []
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
                        notes.append((tick_to_sec(onset_tick), tick_to_sec(abs_tick), msg.note))
        for pitch, onset_tick in active.items():
            if pitch >= KS_THRESHOLD:
                notes.append((tick_to_sec(onset_tick), tick_to_sec(abs_tick), pitch))

    if len(notes) < 2:
        return None

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

    in_penta = sum(1 for p in pitches if p % 12 in D_PENTATONIC)
    penta_pct = in_penta / note_count * 100

    intervals = np.abs(np.diff(pitches))
    mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0
    leap_pct = float(np.sum(intervals > 5) / len(intervals) * 100) if len(intervals) > 0 else 0
    step_pct = float(np.sum((intervals >= 1) & (intervals <= 3)) / len(intervals) * 100) if len(intervals) > 0 else 0
    unison_pct = float(np.sum(intervals == 0) / len(intervals) * 100) if len(intervals) > 0 else 0

    ioi = np.diff(onsets)
    ioi = ioi[ioi >= 0]
    if len(ioi) > 0:
        mean_ioi = float(np.mean(ioi))
        std_ioi = float(np.std(ioi))
        cv_ioi = std_ioi / mean_ioi if mean_ioi > 0 else 0
    else:
        mean_ioi = std_ioi = cv_ioi = 0

    mean_note_dur = float(np.mean(durations_note))
    std_note_dur = float(np.std(durations_note))

    if len(intervals) >= 2:
        contours = []
        for i in range(len(intervals) - 1):
            d1 = int(np.sign(np.diff(pitches)[i]))
            d2 = int(np.sign(np.diff(pitches)[i+1])) if i+1 < len(np.diff(pitches)) else 0
            s1 = min(int(intervals[i]), 12)
            s2 = min(int(intervals[i+1]), 12) if i+1 < len(intervals) else 0
            contours.append((d1, s1, d2, s2))
        contour_counts = Counter(contours)
        repeated = sum(1 for c in contours if contour_counts[c] > 1)
        repetition_pct = repeated / len(contours) * 100
    else:
        repetition_pct = 0

    gaps = []
    for i in range(len(notes) - 1):
        gap = notes[i+1][0] - notes[i][1]
        if gap > 0:
            gaps.append(gap)
    longest_gap = max(gaps) if gaps else 0

    pc_counts = Counter(p % 12 for p in pitches)
    total = sum(pc_counts.values())
    probs = np.array([pc_counts[pc] / total for pc in range(12) if pc in pc_counts])
    pitch_entropy = float(entropy(probs, base=2))

    return {
        'duration': duration_sec, 'note_count': note_count, 'density': density,
        'pitch_range': pitch_range, 'unique_pc': unique_pc, 'pc_list': pc_list,
        'penta_pct': penta_pct, 'mean_interval': mean_interval,
        'leap_pct': leap_pct, 'step_pct': step_pct, 'unison_pct': unison_pct,
        'mean_ioi': mean_ioi, 'std_ioi': std_ioi, 'cv_ioi': cv_ioi,
        'mean_note_dur': mean_note_dur, 'std_note_dur': std_note_dur,
        'repetition_pct': repetition_pct, 'longest_gap': longest_gap,
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
    print(f"  {label}: {len(results)} files analyzed, {skipped} skipped")
    return results


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
    ('cv_ioi',         'IOI CV',                 '{:.3f}'),
    ('mean_note_dur',  'Mean note dur (s)',      '{:.3f}'),
    ('repetition_pct', 'Repetition %',           '{:.1f}'),
    ('longest_gap',    'Longest gap (s)',         '{:.2f}'),
    ('pitch_entropy',  'Pitch entropy (bits)',   '{:.3f}'),
]


def main():
    print("=" * 100)
    print("3-WAY GUZHENG GENERATION MODEL COMPARISON")
    print("=" * 100)

    all_results = []
    for m in MODELS:
        files = collect_files(m["dirs"])
        print(f"\n{m['name']}: {len(files)} files")
        res = analyze_all(files, m["name"])
        all_results.append(res)

    # ── Summary statistics ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS (mean +/- std)")
    print("=" * 100)

    col_w = 24
    shorts = [m["short"] for m in MODELS]
    header = f"{'Metric':<25s}"
    for s in shorts:
        header += f" | {s:>{col_w}s}"
    header += f" | {'Kruskal p':>10s} | {'Sig':>4s}"
    print(header)
    print("-" * len(header))

    for key, label, fmt in METRICS:
        row = f"{label:<25s}"
        vals_list = []
        for res in all_results:
            vals = np.array([r[key] for r in res])
            vals_list.append(vals)
            m, s = np.mean(vals), np.std(vals)
            cell = fmt.format(m) + " +/- " + fmt.format(s)
            row += f" | {cell:>{col_w}s}"

        # Kruskal-Wallis (3-way non-parametric)
        try:
            stat, pval = kruskal(*vals_list)
        except ValueError:
            pval = 1.0
        sig = ""
        if pval < 0.001: sig = "***"
        elif pval < 0.01: sig = "**"
        elif pval < 0.05: sig = "*"
        row += f" | {pval:>10.4f} | {sig:>4s}"
        print(row)

    print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001 (Kruskal-Wallis)")

    # ── Pairwise Mann-Whitney U ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("PAIRWISE COMPARISONS (Mann-Whitney U, two-sided)")
    print("=" * 100)

    pairs = [
        (0, 1, "NotaGen vs MIDITrans"),
        (0, 2, "NotaGen vs CharGPT"),
        (0, 3, "NotaGen vs PatchGPT"),
        (1, 2, "MIDITrans vs CharGPT"),
        (1, 3, "MIDITrans vs PatchGPT"),
        (2, 3, "CharGPT vs PatchGPT"),
    ]
    for i, j, pair_label in pairs:
        print(f"\n  --- {pair_label} ---")
        header2 = f"  {'Metric':<25s} | {'Mean diff':>12s} | {'p-value':>10s} | {'Sig':>4s}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for key, label, fmt in METRICS:
            v1 = np.array([r[key] for r in all_results[i]])
            v2 = np.array([r[key] for r in all_results[j]])
            diff = np.mean(v1) - np.mean(v2)
            try:
                _, pval = mannwhitneyu(v1, v2, alternative='two-sided')
            except ValueError:
                pval = 1.0
            sig = ""
            if pval < 0.001: sig = "***"
            elif pval < 0.01: sig = "**"
            elif pval < 0.05: sig = "*"
            print(f"  {label:<25s} | {diff:>+12.2f} | {pval:>10.4f} | {sig:>4s}")

    # ── Qualitative assessment ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("QUALITATIVE ASSESSMENT")
    print("=" * 100)

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

    def grade(v, lo, hi, higher_better):
        if higher_better is None:
            if lo <= v <= hi: return "GOOD"
            elif v < lo: return f"LOW"
            else: return f"HIGH"
        elif higher_better:
            return "GOOD" if v >= lo else "LOW"
        else:
            return "GOOD" if v <= hi else "HIGH"

    scores = [0] * len(MODELS)
    for key, label, ideal, lo, hi, higher_better in criteria:
        print(f"\n  {label} (ideal: {ideal}):")
        for mi, res in enumerate(all_results):
            vals = np.array([r[key] for r in res])
            m = np.mean(vals)
            g = grade(m, lo, hi, higher_better)
            if g == "GOOD":
                scores[mi] += 1
            print(f"    {MODELS[mi]['short']:>12s}: {m:>8.1f}  {g}")

    # ── Overall ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"\n  Criteria met (out of {len(criteria)}):")
    for mi, m in enumerate(MODELS):
        print(f"    {m['name']}: {scores[mi]}/{len(criteria)}")

    print("\n  File-level ranges:")
    for key, label, fmt in [('duration', 'Duration', '{:.1f}'), ('note_count', 'Notes', '{:.0f}'), ('density', 'Density', '{:.2f}')]:
        print(f"    {label}:")
        for mi, m in enumerate(MODELS):
            v = [r[key] for r in all_results[mi]]
            print(f"      {m['short']:>12s}: min={fmt.format(min(v))}, max={fmt.format(max(v))}, median={fmt.format(np.median(v))}")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()

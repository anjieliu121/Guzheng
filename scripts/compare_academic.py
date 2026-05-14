"""
Academic comparison: NotaGen vs PatchGPT
Metrics from established symbolic music generation literature.

References:
  [1] Dong et al. (2018). "MuseGAN: Multi-track Sequential GANs for
      Symbolic Music Generation and Accompaniment." AAAI.
      → Pitch Class Entropy, Groove Consistency, Empty Bar Rate
  [2] Yang & Lerch (2020). "On the evaluation of generative models in music."
      Neural Computing and Applications.
      → Overlapping Area, pitch/rhythm distribution comparison
  [3] Wu & Yang (2020). "The Jazz Transformer on the Front Line:
      Exploring the Shortcomings of AI-composed Music through
      Quantitative Measures." ISMIR.
      → Pitch Class Transition Matrix, self-similarity
  [4] Ji et al. (2023). "A Survey on Deep Learning for Symbolic Music
      Generation: Representations, Algorithms, Evaluations, and Challenges."
      ACM Computing Surveys.
      → Comprehensive metric taxonomy
  [5] Huang & Yang (2020). "Pop Music Transformer: Beat-based Modeling and
      Generation of Expressive Pop Piano Compositions." ACM Multimedia.
      → Groove Consistency, pitch histogram metrics
  [6] Ren et al. (2020). "PopMAG: Pop Music Accompaniment Generation."
      ACM Multimedia.
      → Pitch/rhythm distribution metrics
"""

import os
import glob
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu, entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import mido

# ── Configuration ──────────────────────────────────────────────────────────

TRAINING_DIR = "/Users/anjie/Documents/MyGuzheng/Guzheng/MIDI_transposed/"
TRAINING_SUFFIX = "_D.mid"

NOTAGEN_DIRS = [
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_batch2_ks/",
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_7/generated/medium_D_ks/",
]
PATCHGPT_DIRS = [
    "/Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_9/generated/patchgpt_D_ks/",
]

KS_THRESHOLD = 36
D_PENTATONIC = {2, 4, 6, 9, 11}
TICKS_PER_BAR = None  # set per file from time signature


# ══════════════════════════════════════════════════════════════════════════
# MIDI Parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_midi(filepath):
    """Parse MIDI → list of (onset_sec, offset_sec, pitch, velocity) tuples."""
    try:
        mid = mido.MidiFile(filepath)
    except Exception:
        return None, None, None

    # Tempo map
    tempo_map = []
    ticks_per_beat = mid.ticks_per_beat
    time_sig_num, time_sig_den = 4, 4  # default
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))
            if msg.type == 'time_signature':
                time_sig_num = msg.numerator
                time_sig_den = msg.denominator
    if not tempo_map:
        tempo_map = [(0, mido.bpm2tempo(120))]
    tempo_map.sort()

    ticks_per_bar = ticks_per_beat * time_sig_num * (4 // time_sig_den)

    def tick_to_sec(tick):
        sec, pt, ptem = 0.0, 0, tempo_map[0][1]
        for tt, tem in tempo_map:
            if tt >= tick:
                break
            sec += mido.tick2second(tt - pt, ticks_per_beat, ptem)
            pt, ptem = tt, tem
        sec += mido.tick2second(tick - pt, ticks_per_beat, ptem)
        return sec

    notes = []
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note >= KS_THRESHOLD:
                    active[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active and msg.note >= KS_THRESHOLD:
                    onset_tick, vel = active.pop(msg.note)
                    notes.append((tick_to_sec(onset_tick), tick_to_sec(abs_tick),
                                  msg.note, vel, onset_tick))
        for pitch, (onset_tick, vel) in active.items():
            if pitch >= KS_THRESHOLD:
                notes.append((tick_to_sec(onset_tick), tick_to_sec(abs_tick),
                              pitch, vel, onset_tick))

    if len(notes) < 4:
        return None, None, None

    notes.sort(key=lambda x: x[0])
    return notes, ticks_per_bar, ticks_per_beat


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(notes, ticks_per_bar, ticks_per_beat):
    """Compute all academic metrics for a single MIDI file."""
    onsets = np.array([n[0] for n in notes])
    offsets = np.array([n[1] for n in notes])
    pitches = np.array([n[2] for n in notes])
    velocities = np.array([n[3] for n in notes])
    onset_ticks = np.array([n[4] for n in notes])
    durations = offsets - onsets

    total_dur = max(offsets) - min(onsets)
    if total_dur <= 0:
        return None
    N = len(notes)

    # ── 1. Pitch Class Entropy (PCE) [1][4] ──────────────────────────
    # Shannon entropy of the pitch class histogram (base 2)
    pc_counts = Counter(p % 12 for p in pitches)
    total = sum(pc_counts.values())
    pc_probs = np.array([pc_counts.get(pc, 0) / total for pc in range(12)])
    pc_probs = pc_probs[pc_probs > 0]
    pce = float(entropy(pc_probs, base=2))

    # ── 2. Scale Consistency (SC) [1][4] ──────────────────────────────
    # Fraction of notes belonging to the target scale
    in_scale = sum(1 for p in pitches if p % 12 in D_PENTATONIC)
    sc = in_scale / N

    # ── 3. Pitch Range (PR) [2][4] ────────────────────────────────────
    pr = int(pitches.max() - pitches.min())

    # ── 4. Note Density (ND) [1][4] ──────────────────────────────────
    # Notes per second
    nd = N / total_dur

    # ── 5. Pitch Interval Class Entropy (PICE) [3][4] ────────────────
    # Shannon entropy of the distribution of melodic interval classes (0-11)
    intervals_signed = np.diff(pitches)
    interval_classes = np.abs(intervals_signed) % 12
    ic_counts = Counter(int(ic) for ic in interval_classes)
    ic_total = sum(ic_counts.values())
    ic_probs = np.array([ic_counts.get(ic, 0) / ic_total for ic in range(12)])
    ic_probs = ic_probs[ic_probs > 0]
    pice = float(entropy(ic_probs, base=2))

    # ── 6. Groove Consistency (GC) [1][5] ─────────────────────────────
    # Measures how similar the rhythmic pattern is across bars.
    # For each bar, create a binary onset vector (quantized to 16th notes).
    # GC = mean pairwise cosine similarity across bars.
    if ticks_per_bar and ticks_per_bar > 0:
        steps_per_bar = 16  # quantize to 16th notes
        tick_per_step = ticks_per_bar / steps_per_bar
        max_tick = int(onset_ticks.max())
        n_bars = max(1, int(np.ceil(max_tick / ticks_per_bar)))

        bar_vectors = []
        for bar_idx in range(n_bars):
            vec = np.zeros(steps_per_bar)
            bar_start = bar_idx * ticks_per_bar
            bar_end = bar_start + ticks_per_bar
            for ot in onset_ticks:
                if bar_start <= ot < bar_end:
                    step = min(int((ot - bar_start) / tick_per_step), steps_per_bar - 1)
                    vec[step] = 1.0
            if vec.sum() > 0:
                bar_vectors.append(vec)

        if len(bar_vectors) >= 2:
            sims = []
            for i in range(len(bar_vectors)):
                for j in range(i + 1, len(bar_vectors)):
                    dot = np.dot(bar_vectors[i], bar_vectors[j])
                    norm = np.linalg.norm(bar_vectors[i]) * np.linalg.norm(bar_vectors[j])
                    if norm > 0:
                        sims.append(dot / norm)
            gc = float(np.mean(sims)) if sims else 0.0
        else:
            gc = 0.0
    else:
        gc = 0.0

    # ── 7. Duration Distribution Entropy (DDE) [4][6] ─────────────────
    # Quantize note durations to bins, compute entropy
    dur_bins = np.array([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])  # in seconds
    dur_indices = np.digitize(durations, dur_bins)
    dur_counts = Counter(int(d) for d in dur_indices)
    dur_total = sum(dur_counts.values())
    dur_probs = np.array([dur_counts.get(i, 0) / dur_total for i in range(len(dur_bins) + 1)])
    dur_probs = dur_probs[dur_probs > 0]
    dde = float(entropy(dur_probs, base=2))

    # ── 8. Pitch Transition Matrix Entropy (PTME) [3] ─────────────────
    # Entropy of the first-order Markov pitch class transition matrix
    pc_seq = [int(p % 12) for p in pitches]
    trans_counts = np.zeros((12, 12))
    for i in range(len(pc_seq) - 1):
        trans_counts[pc_seq[i]][pc_seq[i + 1]] += 1
    # Normalize rows
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_probs = trans_counts / row_sums
    # Entropy per row, weighted by row frequency
    row_entropies = []
    row_weights = []
    for r in range(12):
        if trans_counts[r].sum() > 0:
            p_row = trans_probs[r]
            p_row = p_row[p_row > 0]
            row_entropies.append(float(entropy(p_row, base=2)))
            row_weights.append(trans_counts[r].sum())
    if row_weights:
        total_w = sum(row_weights)
        ptme = sum(e * w / total_w for e, w in zip(row_entropies, row_weights))
    else:
        ptme = 0.0

    # ── 9. Melodic Contour Metrics [4] ────────────────────────────────
    # Step %: intervals 1-3 semitones (conjunct motion)
    abs_intervals = np.abs(intervals_signed)
    if len(abs_intervals) > 0:
        step_pct = float(np.sum((abs_intervals >= 1) & (abs_intervals <= 3)) / len(abs_intervals))
        leap_pct = float(np.sum(abs_intervals > 5) / len(abs_intervals))
        mean_abs_interval = float(np.mean(abs_intervals))
    else:
        step_pct = leap_pct = mean_abs_interval = 0.0

    # ── 10. IOI CV [2][4] ─────────────────────────────────────────────
    # Coefficient of variation of inter-onset intervals
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0]
    if len(ioi) > 0:
        ioi_cv = float(np.std(ioi) / np.mean(ioi))
    else:
        ioi_cv = 0.0

    # ── 11. Total Duration ────────────────────────────────────────────
    duration_sec = total_dur

    # ── 12. Pitch Class Histogram (for distributional comparison) ─────
    pc_hist = np.array([pc_counts.get(pc, 0) for pc in range(12)], dtype=float)
    pc_hist_norm = pc_hist / pc_hist.sum() if pc_hist.sum() > 0 else pc_hist

    # ── 13. Interval Histogram (for distributional comparison) ────────
    int_hist = np.zeros(25)  # intervals 0-24
    for iv in abs_intervals:
        idx = min(int(iv), 24)
        int_hist[idx] += 1
    int_hist_norm = int_hist / int_hist.sum() if int_hist.sum() > 0 else int_hist

    # ── 14. Duration Histogram (for distributional comparison) ────────
    dur_hist = np.array([dur_counts.get(i, 0) for i in range(len(dur_bins) + 1)], dtype=float)
    dur_hist_norm = dur_hist / dur_hist.sum() if dur_hist.sum() > 0 else dur_hist

    return {
        'pce': pce,
        'sc': sc,
        'pr': pr,
        'nd': nd,
        'pice': pice,
        'gc': gc,
        'dde': dde,
        'ptme': ptme,
        'step_pct': step_pct,
        'leap_pct': leap_pct,
        'mean_interval': mean_abs_interval,
        'ioi_cv': ioi_cv,
        'duration': duration_sec,
        'n_notes': N,
        'pc_hist': pc_hist_norm,
        'int_hist': int_hist_norm,
        'dur_hist': dur_hist_norm,
    }


def analyze_dir(dirs, label):
    """Analyze all MIDI files in given directories."""
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*.mid'))))
    results = []
    skipped = 0
    for f in files:
        notes, tpb, tpbeat = parse_midi(f)
        if notes is None:
            skipped += 1
            continue
        m = compute_metrics(notes, tpb, tpbeat)
        if m is not None:
            results.append(m)
        else:
            skipped += 1
    print(f"  {label}: {len(results)} analyzed, {skipped} skipped")
    return results


def analyze_training(midi_dir, suffix):
    """Analyze training data (D-key transpositions) as reference."""
    files = sorted(glob.glob(os.path.join(midi_dir, f"*{suffix}")))
    results = []
    skipped = 0
    for f in files:
        notes, tpb, tpbeat = parse_midi(f)
        if notes is None:
            skipped += 1
            continue
        m = compute_metrics(notes, tpb, tpbeat)
        if m is not None:
            results.append(m)
        else:
            skipped += 1
    print(f"  Training data: {len(results)} analyzed, {skipped} skipped")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Distributional Metrics (comparing generated vs training data)
# ══════════════════════════════════════════════════════════════════════════

def aggregate_histogram(results, key):
    """Aggregate histograms across all files by averaging."""
    hists = np.array([r[key] for r in results])
    mean_hist = hists.mean(axis=0)
    if mean_hist.sum() > 0:
        mean_hist = mean_hist / mean_hist.sum()
    return mean_hist


def overlapping_area(p, q):
    """Overlapping Area (OA) [2]: sum of min(p_i, q_i). Range [0, 1]."""
    return float(np.sum(np.minimum(p, q)))


def kl_divergence(p, q, eps=1e-10):
    """KL divergence D_KL(P || Q) with smoothing [2]."""
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

SCALAR_METRICS = [
    ('pce',           'Pitch Class Entropy (bits)',            '{:.3f}', '[1][4]'),
    ('sc',            'Scale Consistency',                     '{:.3f}', '[1][4]'),
    ('pr',            'Pitch Range (semitones)',               '{:.1f}', '[2][4]'),
    ('nd',            'Note Density (notes/s)',                '{:.2f}', '[1][4]'),
    ('pice',          'Pitch Interval Entropy (bits)',         '{:.3f}', '[3][4]'),
    ('gc',            'Groove Consistency',                    '{:.3f}', '[1][5]'),
    ('dde',           'Duration Distribution Entropy (bits)',  '{:.3f}', '[4][6]'),
    ('ptme',          'Pitch Transition Entropy (bits)',       '{:.3f}', '[3]'),
    ('step_pct',      'Conjunct Motion Ratio',                '{:.3f}', '[4]'),
    ('leap_pct',      'Disjunct Motion Ratio (>5 st)',        '{:.3f}', '[4]'),
    ('mean_interval',  'Mean Melodic Interval (st)',           '{:.2f}', '[4]'),
    ('ioi_cv',        'IOI Coefficient of Variation',         '{:.3f}', '[2][4]'),
    ('duration',      'Total Duration (s)',                    '{:.1f}', ''),
    ('n_notes',       'Note Count',                           '{:.0f}', ''),
]


def main():
    print("=" * 95)
    print("ACADEMIC COMPARISON: NotaGen vs PatchGPT")
    print("Metrics from Symbolic Music Generation Literature")
    print("=" * 95)

    print("\nAnalyzing...")
    ref = analyze_training(TRAINING_DIR, TRAINING_SUFFIX)
    nota = analyze_dir(NOTAGEN_DIRS, "NotaGen")
    patch = analyze_dir(PATCHGPT_DIRS, "PatchGPT")

    # ══════════════════════════════════════════════════════════════════
    # Table 1: Scalar Metrics (mean ± std)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("TABLE 1: INTRA-SET METRICS (mean +/- std)")
    print("Computed per generated piece, then aggregated.")
    print("=" * 95)

    header = (f"{'Metric':<35s} | {'Training':>18s} | {'NotaGen':>18s} | "
              f"{'PatchGPT':>18s} | {'p-val':>7s} | {'Ref':>5s}")
    print(header)
    print("-" * len(header))

    for key, label, fmt, ref_str in SCALAR_METRICS:
        vr = np.array([r[key] for r in ref])
        vn = np.array([r[key] for r in nota])
        vp = np.array([r[key] for r in patch])

        sr = fmt.format(np.mean(vr)) + " +/- " + fmt.format(np.std(vr))
        sn = fmt.format(np.mean(vn)) + " +/- " + fmt.format(np.std(vn))
        sp = fmt.format(np.mean(vp)) + " +/- " + fmt.format(np.std(vp))

        # Mann-Whitney U: NotaGen vs PatchGPT
        try:
            _, pval = mannwhitneyu(vn, vp, alternative='two-sided')
        except ValueError:
            pval = 1.0
        sig = ""
        if pval < 0.001: sig = "***"
        elif pval < 0.01: sig = "**"
        elif pval < 0.05: sig = "*"

        print(f"{label:<35s} | {sr:>18s} | {sn:>18s} | {sp:>18s} | {pval:>6.4f}{sig:1s} | {ref_str:>5s}")

    print("\nSignificance: Mann-Whitney U, two-sided. * p<.05, ** p<.01, *** p<.001")

    # ══════════════════════════════════════════════════════════════════
    # Table 2: Distributional Similarity to Training Data
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("TABLE 2: DISTRIBUTIONAL SIMILARITY TO TRAINING DATA")
    print("Lower KL/JSD = closer to training; Higher OA = more overlap.")
    print("=" * 95)

    ref_pc = aggregate_histogram(ref, 'pc_hist')
    ref_int = aggregate_histogram(ref, 'int_hist')
    ref_dur = aggregate_histogram(ref, 'dur_hist')

    nota_pc = aggregate_histogram(nota, 'pc_hist')
    nota_int = aggregate_histogram(nota, 'int_hist')
    nota_dur = aggregate_histogram(nota, 'dur_hist')

    patch_pc = aggregate_histogram(patch, 'pc_hist')
    patch_int = aggregate_histogram(patch, 'int_hist')
    patch_dur = aggregate_histogram(patch, 'dur_hist')

    dist_metrics = [
        ('Pitch Class', ref_pc, nota_pc, patch_pc, '[2]'),
        ('Interval',    ref_int, nota_int, patch_int, '[2][3]'),
        ('Duration',    ref_dur, nota_dur, patch_dur, '[2][6]'),
    ]

    header2 = (f"{'Distribution':<15s} | {'Metric':<20s} | {'NotaGen':>10s} | "
               f"{'PatchGPT':>10s} | {'Closer':>8s} | {'Ref':>5s}")
    print(header2)
    print("-" * len(header2))

    nota_wins = 0
    patch_wins = 0
    for dist_name, ref_h, nota_h, patch_h, ref_str in dist_metrics:
        # Overlapping Area [2]
        oa_n = overlapping_area(ref_h, nota_h)
        oa_p = overlapping_area(ref_h, patch_h)
        closer_oa = "NotaGen" if oa_n > oa_p else "PatchGPT" if oa_p > oa_n else "Tie"
        if closer_oa == "NotaGen": nota_wins += 1
        elif closer_oa == "PatchGPT": patch_wins += 1
        print(f"{dist_name:<15s} | {'Overlap Area (OA)':<20s} | {oa_n:>10.4f} | {oa_p:>10.4f} | {closer_oa:>8s} | {ref_str:>5s}")

        # KL Divergence [2]
        kl_n = kl_divergence(nota_h, ref_h)
        kl_p = kl_divergence(patch_h, ref_h)
        closer_kl = "NotaGen" if kl_n < kl_p else "PatchGPT" if kl_p < kl_n else "Tie"
        if closer_kl == "NotaGen": nota_wins += 1
        elif closer_kl == "PatchGPT": patch_wins += 1
        print(f"{'':15s} | {'KL Divergence':<20s} | {kl_n:>10.4f} | {kl_p:>10.4f} | {closer_kl:>8s} |")

        # Jensen-Shannon Divergence
        jsd_n = float(jensenshannon(nota_h + 1e-10, ref_h + 1e-10, base=2))
        jsd_p = float(jensenshannon(patch_h + 1e-10, ref_h + 1e-10, base=2))
        closer_jsd = "NotaGen" if jsd_n < jsd_p else "PatchGPT" if jsd_p < jsd_n else "Tie"
        if closer_jsd == "NotaGen": nota_wins += 1
        elif closer_jsd == "PatchGPT": patch_wins += 1
        print(f"{'':15s} | {'JSD':<20s} | {jsd_n:>10.4f} | {jsd_p:>10.4f} | {closer_jsd:>8s} |")

        # Earth Mover's Distance (Wasserstein) [2]
        emd_n = wasserstein_distance(range(len(ref_h)), range(len(nota_h)), ref_h, nota_h)
        emd_p = wasserstein_distance(range(len(ref_h)), range(len(patch_h)), ref_h, patch_h)
        closer_emd = "NotaGen" if emd_n < emd_p else "PatchGPT" if emd_p < emd_n else "Tie"
        if closer_emd == "NotaGen": nota_wins += 1
        elif closer_emd == "PatchGPT": patch_wins += 1
        print(f"{'':15s} | {'EMD (Wasserstein)':<20s} | {emd_n:>10.4f} | {emd_p:>10.4f} | {closer_emd:>8s} |")

    print(f"\nDistributional metric wins: NotaGen={nota_wins}, PatchGPT={patch_wins} (out of {nota_wins+patch_wins})")

    # ══════════════════════════════════════════════════════════════════
    # Table 3: Distance from Training Data (per-metric)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("TABLE 3: ABSOLUTE DISTANCE FROM TRAINING DATA (|mean_gen - mean_train|)")
    print("Smaller distance = closer to real music characteristics.")
    print("=" * 95)

    header3 = f"{'Metric':<35s} | {'Train mean':>10s} | {'NotaGen':>10s} | {'PatchGPT':>10s} | {'N dist':>8s} | {'P dist':>8s} | {'Closer':>8s}"
    print(header3)
    print("-" * len(header3))

    nota_closer = 0
    patch_closer = 0
    for key, label, fmt, ref_str in SCALAR_METRICS:
        mr = np.mean([r[key] for r in ref])
        mn = np.mean([r[key] for r in nota])
        mp = np.mean([r[key] for r in patch])
        dn = abs(mn - mr)
        dp = abs(mp - mr)
        closer = "NotaGen" if dn < dp else "PatchGPT" if dp < dn else "Tie"
        if closer == "NotaGen": nota_closer += 1
        elif closer == "PatchGPT": patch_closer += 1
        print(f"{label:<35s} | {fmt.format(mr):>10s} | {fmt.format(mn):>10s} | {fmt.format(mp):>10s} | {dn:>8.3f} | {dp:>8.3f} | {closer:>8s}")

    print(f"\nCloser to training: NotaGen={nota_closer}, PatchGPT={patch_closer} (out of {nota_closer+patch_closer})")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)

    print(f"""
Models compared:
  NotaGen   : 244M params, hierarchical (patch+char), pre-trained on 1M+ scores
  PatchGPT  : 242M params, hierarchical (patch+char), trained from scratch

Training data: {len(ref)} guzheng pieces in D pentatonic
Generated: NotaGen {len(nota)} files, PatchGPT {len(patch)} files

Distributional similarity wins:  NotaGen {nota_wins} / PatchGPT {patch_wins}
Distance-from-training wins:     NotaGen {nota_closer} / PatchGPT {patch_closer}

References:
  [1] Dong et al. (2018) "MuseGAN" AAAI
  [2] Yang & Lerch (2020) "On the evaluation of generative models in music" NCA
  [3] Wu & Yang (2020) "The Jazz Transformer" ISMIR
  [4] Ji et al. (2023) "A Survey on Deep Learning for Symbolic Music Generation" ACM CS
  [5] Huang & Yang (2020) "Pop Music Transformer" ACM MM
  [6] Ren et al. (2020) "PopMAG" ACM MM
""")


if __name__ == '__main__':
    main()

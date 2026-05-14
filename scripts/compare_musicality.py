"""
Musicality comparison: NotaGen vs PatchGPT
Higher-level metrics capturing melodic coherence, phrase structure,
motif development, and large-scale form — not just statistical distributions.

Metric categories (following the taxonomy in [4]):
  A. Melodic Quality    — coherence, smoothness, phrase arcs
  B. Rhythmic Quality   — regularity-variety balance, syncopation
  C. Structural Quality — form, motif repetition, self-similarity
  D. Distributional     — distance from training data (as reference baseline)

References:
  [1] Dong et al. (2018) "MuseGAN" AAAI
  [2] Yang & Lerch (2020) "On the evaluation of generative models in music" NCA
  [3] Wu & Yang (2020) "The Jazz Transformer on the Front Line" ISMIR
  [4] Ji et al. (2023) "A Survey on Deep Learning for Symbolic Music Generation" ACM CS
  [5] Huang & Yang (2020) "Pop Music Transformer" ACM MM
  [6] Conklin (2003) "Music Generation from Statistical Models" AISB Symposium
  [7] Pearce & Wiggins (2012) "Auditory Expectation: The Information Dynamics
      of Music Perception and Cognition" Topics in Cognitive Science
  [8] Lerdahl & Jackendoff (1983) "A Generative Theory of Tonal Music" MIT Press
  [9] Meredith et al. (2002) "Algorithms for Discovering Repeated Patterns
      in Multidimensional Representations of Polyphonic Music" J New Music Res
"""

import os, glob, zlib, struct
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu, entropy
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


# ══════════════════════════════════════════════════════════════════════════
# MIDI Parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_midi(filepath):
    """Parse MIDI → (notes, ticks_per_bar, ticks_per_beat) or None."""
    try:
        mid = mido.MidiFile(filepath)
    except Exception:
        return None

    ticks_per_beat = mid.ticks_per_beat
    tempo_map = []
    time_sig_num, time_sig_den = 4, 4
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
    ticks_per_bar = ticks_per_beat * time_sig_num * (4 // max(time_sig_den, 1))

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
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note >= KS_THRESHOLD:
                active[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active and msg.note >= KS_THRESHOLD:
                    ot, vel = active.pop(msg.note)
                    notes.append({
                        'onset': tick_to_sec(ot),
                        'offset': tick_to_sec(abs_tick),
                        'pitch': msg.note,
                        'velocity': vel,
                        'onset_tick': ot,
                    })
        for pitch, (ot, vel) in active.items():
            if pitch >= KS_THRESHOLD:
                notes.append({
                    'onset': tick_to_sec(ot), 'offset': tick_to_sec(abs_tick),
                    'pitch': pitch, 'velocity': vel, 'onset_tick': ot,
                })

    if len(notes) < 8:
        return None
    notes.sort(key=lambda n: n['onset'])
    return {'notes': notes, 'ticks_per_bar': ticks_per_bar, 'ticks_per_beat': ticks_per_beat}


# ══════════════════════════════════════════════════════════════════════════
# A. MELODIC QUALITY
# ══════════════════════════════════════════════════════════════════════════

def melodic_metrics(data):
    notes = data['notes']
    pitches = np.array([n['pitch'] for n in notes])
    onsets = np.array([n['onset'] for n in notes])
    offsets = np.array([n['offset'] for n in notes])
    intervals = np.diff(pitches)
    abs_intervals = np.abs(intervals)
    N = len(pitches)

    # ── A1. Pitch Autocorrelation (melodic coherence) [7] ────────────
    # High autocorrelation = pitch at time t predicts pitch at t+k
    # Coherent melodies have high short-lag autocorrelation
    centered = pitches - pitches.mean()
    var = np.var(centered)
    if var > 0:
        autocorr_1 = float(np.mean(centered[:-1] * centered[1:]) / var)
        autocorr_4 = float(np.mean(centered[:-4] * centered[4:]) / var) if N > 8 else 0.0
        autocorr_8 = float(np.mean(centered[:-8] * centered[8:]) / var) if N > 16 else 0.0
    else:
        autocorr_1 = autocorr_4 = autocorr_8 = 0.0

    # ── A2. Smoothness Score [8] ─────────────────────────────────────
    # Weighted penalty: 0 for steps (1-3st), mild for 4-5st, heavy for >5st
    # Inspired by Lerdahl & Jackendoff's preference for conjunct motion
    if len(abs_intervals) > 0:
        penalties = np.zeros_like(abs_intervals, dtype=float)
        penalties[abs_intervals == 0] = 0.0      # unison: neutral
        penalties[(abs_intervals >= 1) & (abs_intervals <= 3)] = 0.0  # step: good
        penalties[(abs_intervals >= 4) & (abs_intervals <= 5)] = 0.5  # skip: mild
        penalties[(abs_intervals >= 6) & (abs_intervals <= 7)] = 1.0  # leap: moderate
        penalties[abs_intervals >= 8] = 2.0       # large leap: heavy
        smoothness = 1.0 - float(np.mean(penalties) / 2.0)  # normalize to [0, 1]
        smoothness = max(0.0, smoothness)

        conjunct_ratio = float(np.mean((abs_intervals >= 1) & (abs_intervals <= 3)))
        leap_ratio = float(np.mean(abs_intervals > 5))
    else:
        smoothness = 0.0
        conjunct_ratio = 0.0
        leap_ratio = 0.0

    # ── A3. Phrase Detection & Phrase Arc [8] ─────────────────────────
    # Detect phrase boundaries via IOI gaps > 2× median IOI
    ioi = np.diff(onsets)
    ioi_pos = ioi[ioi > 0]
    if len(ioi_pos) > 0:
        gap_threshold = max(np.median(ioi_pos) * 2.0, 0.5)  # at least 0.5s
    else:
        gap_threshold = 1.0

    phrase_boundaries = [0]
    for i, gap in enumerate(ioi):
        if gap > gap_threshold:
            phrase_boundaries.append(i + 1)
    phrase_boundaries.append(N)

    phrases = []
    for i in range(len(phrase_boundaries) - 1):
        start, end = phrase_boundaries[i], phrase_boundaries[i + 1]
        if end - start >= 3:  # phrases need at least 3 notes
            phrases.append(pitches[start:end])

    n_phrases = len(phrases)
    if n_phrases > 0:
        phrase_lengths = [len(p) for p in phrases]
        mean_phrase_len = float(np.mean(phrase_lengths))

        # Phrase arc score: does pitch rise then fall within phrases?
        # A good arc has the peak roughly in the middle third
        arc_scores = []
        for phrase in phrases:
            if len(phrase) < 4:
                continue
            peak_pos = np.argmax(phrase) / (len(phrase) - 1)  # normalized [0,1]
            # Peak in middle 60% of phrase = good arc
            if 0.2 <= peak_pos <= 0.8:
                arc_scores.append(1.0)
            elif 0.1 <= peak_pos <= 0.9:
                arc_scores.append(0.5)
            else:
                arc_scores.append(0.0)
        phrase_arc = float(np.mean(arc_scores)) if arc_scores else 0.0
    else:
        mean_phrase_len = 0.0
        phrase_arc = 0.0

    return {
        'autocorr_1': autocorr_1,
        'autocorr_4': autocorr_4,
        'autocorr_8': autocorr_8,
        'smoothness': smoothness,
        'conjunct_ratio': conjunct_ratio,
        'leap_ratio': leap_ratio,
        'n_phrases': n_phrases,
        'mean_phrase_len': mean_phrase_len,
        'phrase_arc': phrase_arc,
    }


# ══════════════════════════════════════════════════════════════════════════
# B. RHYTHMIC QUALITY
# ══════════════════════════════════════════════════════════════════════════

def rhythm_metrics(data):
    notes = data['notes']
    onsets = np.array([n['onset'] for n in notes])
    onset_ticks = np.array([n['onset_tick'] for n in notes])
    tpb = data['ticks_per_beat']
    tpbar = data['ticks_per_bar']

    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0]

    # ── B1. Rhythmic Regularity-Variety Balance [1][5] ────────────────
    # Use IOI entropy: very low = too regular (boring), very high = chaotic
    # Ideal: moderate entropy relative to the number of distinct IOI values
    if len(ioi) > 0:
        # Quantize IOI to 32nd-note grid for cleaner histogram
        ioi_quantized = np.round(ioi * 16) / 16  # ~62ms resolution
        ioi_counts = Counter(tuple([round(x, 3)]) for x in ioi_quantized)
        total = sum(ioi_counts.values())
        ioi_probs = np.array([c / total for c in ioi_counts.values()])
        ioi_entropy = float(entropy(ioi_probs, base=2))
        ioi_cv = float(np.std(ioi) / np.mean(ioi))
    else:
        ioi_entropy = 0.0
        ioi_cv = 0.0

    # ── B2. Syncopation Index [5] ────────────────────────────────────
    # Fraction of note onsets on weak metric positions
    # In 4/4: beats 1,3 are strong; beats 2,4 are weak;
    # off-beat positions (between beats) are weakest
    if tpb > 0:
        beat_positions = (onset_ticks % tpb) / tpb  # 0.0 = on beat
        # On-beat: position < 0.1 or > 0.9 of beat
        on_beat = np.sum((beat_positions < 0.1) | (beat_positions > 0.9))
        off_beat = len(beat_positions) - on_beat
        syncopation = float(off_beat / len(beat_positions)) if len(beat_positions) > 0 else 0.0
    else:
        syncopation = 0.0

    # ── B3. Groove Consistency [1][5] ─────────────────────────────────
    # Bar-level rhythmic pattern similarity (cosine similarity)
    if tpbar > 0:
        steps_per_bar = 16
        tick_per_step = tpbar / steps_per_bar
        max_tick = int(onset_ticks.max()) if len(onset_ticks) > 0 else 0
        n_bars = max(1, int(np.ceil(max_tick / tpbar)))

        bar_vecs = []
        for b in range(n_bars):
            vec = np.zeros(steps_per_bar)
            b_start = b * tpbar
            b_end = b_start + tpbar
            for ot in onset_ticks:
                if b_start <= ot < b_end:
                    step = min(int((ot - b_start) / tick_per_step), steps_per_bar - 1)
                    vec[step] = 1.0
            if vec.sum() > 0:
                bar_vecs.append(vec)

        if len(bar_vecs) >= 2:
            sims = []
            for i in range(len(bar_vecs)):
                for j in range(i + 1, min(i + 5, len(bar_vecs))):  # nearby bars
                    dot = np.dot(bar_vecs[i], bar_vecs[j])
                    norm = np.linalg.norm(bar_vecs[i]) * np.linalg.norm(bar_vecs[j])
                    if norm > 0:
                        sims.append(dot / norm)
            groove_consistency = float(np.mean(sims)) if sims else 0.0
        else:
            groove_consistency = 0.0
    else:
        groove_consistency = 0.0

    return {
        'ioi_entropy': ioi_entropy,
        'ioi_cv': ioi_cv,
        'syncopation': syncopation,
        'groove_consistency': groove_consistency,
    }


# ══════════════════════════════════════════════════════════════════════════
# C. STRUCTURAL QUALITY
# ══════════════════════════════════════════════════════════════════════════

def structure_metrics(data):
    notes = data['notes']
    pitches = np.array([n['pitch'] for n in notes])
    onsets = np.array([n['onset'] for n in notes])
    intervals = np.diff(pitches)
    N = len(pitches)

    # ── C1. Motif Repetition Rate (n-gram analysis) [6][9] ───────────
    # What fraction of n-note pitch-interval patterns appear more than once?
    # Higher = more motivic development; very high = too repetitive
    def ngram_repetition(seq, n):
        if len(seq) < n:
            return 0.0, 0
        grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
        counts = Counter(grams)
        unique = len(counts)
        repeated = sum(1 for g in grams if counts[g] > 1)
        return float(repeated / len(grams)) if grams else 0.0, unique

    # Use pitch intervals (relative, not absolute) to be transposition-invariant
    int_seq = list(intervals)
    motif_rep_3, n_unique_3 = ngram_repetition(int_seq, 3)
    motif_rep_5, n_unique_5 = ngram_repetition(int_seq, 5)
    motif_rep_8, n_unique_8 = ngram_repetition(int_seq, 8)

    # ── C2. Compression Ratio (algorithmic complexity) [3][4] ─────────
    # Lempel-Ziv complexity proxy via zlib compression ratio
    # Lower ratio = more repetitive structure; higher = more random
    pitch_bytes = bytes([min(max(p, 0), 127) for p in pitches])
    compressed = zlib.compress(pitch_bytes, level=9)
    compression_ratio = len(compressed) / max(len(pitch_bytes), 1)

    # Also compress the interval sequence
    int_bytes = bytes([min(max(i + 64, 0), 127) for i in intervals])
    if len(int_bytes) > 0:
        int_compressed = zlib.compress(int_bytes, level=9)
        int_compression = len(int_compressed) / max(len(int_bytes), 1)
    else:
        int_compression = 1.0

    # ── C3. Self-Similarity (large-scale form) [3][4] ────────────────
    # Divide piece into 8 equal segments, compute pitch histogram per segment
    # Then compute pairwise cosine similarity → detect block structure
    n_segments = 8
    if N >= n_segments * 4:
        seg_size = N // n_segments
        seg_hists = []
        for s in range(n_segments):
            seg_pitches = pitches[s * seg_size:(s + 1) * seg_size]
            hist = np.zeros(12)
            for p in seg_pitches:
                hist[p % 12] += 1
            if hist.sum() > 0:
                hist = hist / hist.sum()
            seg_hists.append(hist)

        # Self-similarity matrix
        ssm = np.zeros((n_segments, n_segments))
        for i in range(n_segments):
            for j in range(n_segments):
                dot = np.dot(seg_hists[i], seg_hists[j])
                ni = np.linalg.norm(seg_hists[i])
                nj = np.linalg.norm(seg_hists[j])
                ssm[i, j] = dot / (ni * nj) if ni > 0 and nj > 0 else 0.0

        # Block diagonal score: how much more similar are nearby segments
        # than distant ones? (measures coherent sections)
        near_sims = []
        far_sims = []
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                if abs(i - j) <= 2:
                    near_sims.append(ssm[i, j])
                else:
                    far_sims.append(ssm[i, j])

        mean_near = float(np.mean(near_sims)) if near_sims else 0.0
        mean_far = float(np.mean(far_sims)) if far_sims else 0.0
        # Form score: near should be higher than far for structured music
        form_score = mean_near - mean_far  # positive = good structure

        # Overall self-similarity (mean of full matrix, excluding diagonal)
        off_diag = ssm[~np.eye(n_segments, dtype=bool)]
        self_sim_mean = float(np.mean(off_diag))
    else:
        form_score = 0.0
        self_sim_mean = 0.0

    # ── C4. Pitch Contour Direction Changes [7] ──────────────────────
    # How often does the melodic direction change? (up→down or down→up)
    # Moderate changes = musical; too many = erratic; too few = monotonous
    if len(intervals) > 1:
        directions = np.sign(intervals)
        direction_changes = np.sum(directions[:-1] != directions[1:])
        direction_change_rate = float(direction_changes / (len(directions) - 1))
    else:
        direction_change_rate = 0.0

    # ── C5. Long-Range Recurrence [9] ────────────────────────────────
    # What fraction of 4-note motifs in the first half appear again in the second half?
    half = len(int_seq) // 2
    if half >= 4:
        first_half = [tuple(int_seq[i:i + 4]) for i in range(half - 3)]
        second_half = [tuple(int_seq[i:i + 4]) for i in range(half, len(int_seq) - 3)]
        first_set = set(first_half)
        second_set = set(second_half)
        if len(first_set) > 0:
            recurrence = float(len(first_set & second_set) / len(first_set))
        else:
            recurrence = 0.0
    else:
        recurrence = 0.0

    return {
        'motif_rep_3': motif_rep_3,
        'motif_rep_5': motif_rep_5,
        'motif_rep_8': motif_rep_8,
        'n_unique_3': n_unique_3,
        'n_unique_5': n_unique_5,
        'compression_ratio': compression_ratio,
        'int_compression': int_compression,
        'form_score': form_score,
        'self_sim_mean': self_sim_mean,
        'direction_change_rate': direction_change_rate,
        'long_range_recurrence': recurrence,
    }


# ══════════════════════════════════════════════════════════════════════════
# Analysis pipeline
# ══════════════════════════════════════════════════════════════════════════

def analyze_file(filepath):
    data = parse_midi(filepath)
    if data is None:
        return None
    m = {}
    m.update(melodic_metrics(data))
    m.update(rhythm_metrics(data))
    m.update(structure_metrics(data))
    # Basic stats
    notes = data['notes']
    onsets = np.array([n['onset'] for n in notes])
    offsets = np.array([n['offset'] for n in notes])
    m['duration'] = float(max(offsets) - min(onsets))
    m['n_notes'] = len(notes)
    m['density'] = m['n_notes'] / m['duration'] if m['duration'] > 0 else 0
    return m


def analyze_set(dirs, label, suffix=None):
    files = []
    for d in dirs:
        if suffix:
            files.extend(sorted(glob.glob(os.path.join(d, f"*{suffix}"))))
        else:
            files.extend(sorted(glob.glob(os.path.join(d, "*.mid"))))
    results = []
    for f in files:
        r = analyze_file(f)
        if r is not None:
            results.append(r)
    print(f"  {label}: {len(results)}/{len(files)} analyzed")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════

METRICS = [
    # (key, label, format, higher_is_better_or_None, category, references)

    # A. Pitch / Melodic Structure
    ('autocorr_1',       'Pitch Autocorr. rho_1',            '{:.3f}', True,  'A', '[7]'),
    ('autocorr_4',       'Pitch Autocorr. rho_4',            '{:.3f}', True,  'A', '[7]'),
    ('autocorr_8',       'Pitch Autocorr. rho_8',            '{:.3f}', True,  'A', '[7]'),
    ('smoothness',       'Weighted Interval Penalty (inv.)', '{:.3f}', True,  'A', '[8]'),
    ('conjunct_ratio',   'P(|Delta|<=3 semitones)',          '{:.3f}', True,  'A', '[4][8]'),
    ('leap_ratio',       'P(|Delta|>5 semitones)',           '{:.3f}', False, 'A', '[4]'),
    ('n_phrases',        'Phrase Count (IOI-gap seg.)',      '{:.1f}', None,  'A', '[8]'),
    ('mean_phrase_len',  'Mean Phrase Length (notes)',       '{:.1f}', None,  'A', '[8]'),
    ('phrase_arc',       'Arch-Contour Peak-Position Idx',   '{:.3f}', True,  'A', '[8]'),

    # B. Rhythmic / Temporal Structure
    ('ioi_entropy',      'H(IOI) (bits)',                    '{:.3f}', None,  'B', '[2][4]'),
    ('ioi_cv',           'CV(IOI) = sigma/mu',               '{:.3f}', None,  'B', '[2]'),
    ('syncopation',      'P(off-beat onset)',                '{:.3f}', None,  'B', '[5]'),
    ('groove_consistency','Grooving Similarity GS',          '{:.3f}', True,  'B', '[1][5]'),

    # C. Sequence / Structural Complexity
    ('motif_rep_3',      'Interval 3-gram Recurrence',       '{:.3f}', None,  'C', '[6][9]'),
    ('motif_rep_5',      'Interval 5-gram Recurrence',       '{:.3f}', None,  'C', '[6][9]'),
    ('motif_rep_8',      'Interval 8-gram Recurrence',       '{:.3f}', None,  'C', '[6][9]'),
    ('compression_ratio','LZ Compression Ratio (pitch)',     '{:.3f}', None,  'C', '[3]'),
    ('int_compression',  'LZ Compression Ratio (interval)',  '{:.3f}', None,  'C', '[3]'),
    ('form_score',       'SSM Block-Diagonal Contrast',      '{:.3f}', True,  'C', '[3][4]'),
    ('self_sim_mean',    'Mean PCH-SSM (off-diagonal)',      '{:.3f}', None,  'C', '[3]'),
    ('direction_change_rate', 'Contour Reversal Rate',       '{:.3f}', None,  'C', '[7]'),
    ('long_range_recurrence', 'Cross-Half 4-gram Overlap',   '{:.3f}', True,  'C', '[9]'),

    # Basic
    ('duration',         'Duration (s)',                     '{:.1f}', None,  '-', ''),
    ('n_notes',          'Note Count |N|',                   '{:.0f}', None,  '-', ''),
    ('density',          'Onset Density (Hz)',               '{:.2f}', None,  '-', ''),
]


def main():
    print("=" * 105)
    print("MUSICALITY COMPARISON: NotaGen vs PatchGPT")
    print("Higher-level metrics: melodic coherence, phrase structure, motif development, form")
    print("=" * 105)

    print("\nAnalyzing...")
    ref = analyze_set([TRAINING_DIR], "Training", suffix=TRAINING_SUFFIX)
    nota = analyze_set(NOTAGEN_DIRS, "NotaGen")
    patch = analyze_set(PATCHGPT_DIRS, "PatchGPT")

    # ══════════════════════════════════════════════════════════════════
    # MAIN TABLE
    # ══════════════════════════════════════════════════════════════════
    current_cat = None
    cat_names = {'A': 'A. MELODIC QUALITY', 'B': 'B. RHYTHMIC QUALITY',
                 'C': 'C. STRUCTURAL QUALITY', '-': 'BASIC STATISTICS'}

    for key, label, fmt, hib, cat, refs in METRICS:
        if cat != current_cat:
            current_cat = cat
            print(f"\n{'=' * 105}")
            print(f"  {cat_names[cat]}")
            print(f"{'=' * 105}")
            header = (f"  {'Metric':<32s} | {'Training':>18s} | {'NotaGen':>18s} | "
                      f"{'PatchGPT':>18s} | {'p':>7s} | {'Closer':>8s} | {'Ref':>5s}")
            print(header)
            print("  " + "-" * (len(header) - 2))

        vr = np.array([r[key] for r in ref])
        vn = np.array([r[key] for r in nota])
        vp = np.array([r[key] for r in patch])

        mr, mn, mp = np.mean(vr), np.mean(vn), np.mean(vp)
        sr = fmt.format(mr) + " +/- " + fmt.format(np.std(vr))
        sn = fmt.format(mn) + " +/- " + fmt.format(np.std(vn))
        sp = fmt.format(mp) + " +/- " + fmt.format(np.std(vp))

        try:
            _, pval = mannwhitneyu(vn, vp, alternative='two-sided')
        except ValueError:
            pval = 1.0
        sig = ""
        if pval < 0.001: sig = "***"
        elif pval < 0.01: sig = "**"
        elif pval < 0.05: sig = "*"

        # Which model is closer to training data?
        dn = abs(mn - mr)
        dp = abs(mp - mr)
        closer = "NotaGen" if dn < dp else "PatchGPT" if dp < dn else "Tie"

        pstr = f"{pval:.3f}" if pval >= 0.001 else "<.001"
        print(f"  {label:<32s} | {sr:>18s} | {sn:>18s} | {sp:>18s} | {pstr:>6s}{sig:1s} | {closer:>8s} | {refs:>5s}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 105}")
    print("  SUMMARY: DISTANCE FROM TRAINING DATA")
    print(f"{'=' * 105}")

    nota_wins = {'A': 0, 'B': 0, 'C': 0}
    patch_wins = {'A': 0, 'B': 0, 'C': 0}
    nota_total = {'A': 0, 'B': 0, 'C': 0}

    for key, label, fmt, hib, cat, refs in METRICS:
        if cat == '-':
            continue
        mr = np.mean([r[key] for r in ref])
        mn = np.mean([r[key] for r in nota])
        mp = np.mean([r[key] for r in patch])
        dn, dp = abs(mn - mr), abs(mp - mr)
        nota_total[cat] = nota_total.get(cat, 0) + 1
        if dn < dp:
            nota_wins[cat] += 1
        elif dp < dn:
            patch_wins[cat] += 1

    for cat, name in [('A', 'Melodic Quality'), ('B', 'Rhythmic Quality'), ('C', 'Structural Quality')]:
        total = nota_total[cat]
        nw = nota_wins[cat]
        pw = patch_wins[cat]
        ties = total - nw - pw
        print(f"  {name:<25s}: NotaGen {nw}/{total}, PatchGPT {pw}/{total}" +
              (f", Ties {ties}" if ties > 0 else ""))

    total_all = sum(nota_total.values())
    nw_all = sum(nota_wins.values())
    pw_all = sum(patch_wins.values())
    ties_all = total_all - nw_all - pw_all
    print(f"  {'TOTAL':<25s}: NotaGen {nw_all}/{total_all}, PatchGPT {pw_all}/{total_all}, Ties {ties_all}")

    print(f"""
{'=' * 105}
  INTERPRETATION
{'=' * 105}

  Both models share the same hierarchical patch+character architecture (242M params).
  The only difference: NotaGen was pre-trained on 1M+ Western scores; PatchGPT was not.

  Pre-training provides:
  {'✓' if nota_wins['A'] > patch_wins['A'] else '✗'} Better melodic quality (NotaGen {nota_wins['A']}/{nota_total['A']} metrics closer to training)
    → Higher pitch autocorrelation = more coherent melodic contours
    → Higher conjunct motion = smoother, more singable melodies
    → Better phrase arcs = natural rise-fall within phrases

  {'✓' if nota_wins['B'] > patch_wins['B'] else '✗'} Better rhythmic quality (NotaGen {nota_wins['B']}/{nota_total['B']} metrics closer to training)
    → More stable rhythmic patterns, less erratic timing
    → Groove consistency closer to real guzheng performances

  {'✓' if nota_wins['C'] > patch_wins['C'] else '✗'} Better structural quality (NotaGen {nota_wins['C']}/{nota_total['C']} metrics closer to training)
    → Motif repetition rates closer to real music (neither too rigid nor too random)
    → Better long-range coherence across the piece

  Limitation: These computational metrics are proxies. Human listening evaluations
  (e.g., MOS scores for naturalness, expressiveness, and structural coherence)
  would provide stronger evidence but are beyond the scope of this comparison.

References:
  [1] Dong et al. (2018) "MuseGAN" AAAI
  [2] Yang & Lerch (2020) "On the evaluation of generative models in music" NCA
  [3] Wu & Yang (2020) "The Jazz Transformer on the Front Line" ISMIR
  [4] Ji et al. (2023) "A Survey on Deep Learning for Symbolic Music Generation" ACM CS
  [5] Huang & Yang (2020) "Pop Music Transformer" ACM MM
  [6] Conklin (2003) "Music Generation from Statistical Models" AISB Symposium
  [7] Pearce & Wiggins (2012) "Auditory Expectation" Topics in Cognitive Science
  [8] Lerdahl & Jackendoff (1983) "A Generative Theory of Tonal Music" MIT Press
  [9] Meredith et al. (2002) "Algorithms for Discovering Repeated Patterns" J New Music Res
""")


if __name__ == '__main__':
    main()

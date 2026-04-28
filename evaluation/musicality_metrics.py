"""
Musicality metrics for symbolic MIDI evaluation.

Implements:
1. Compression ratio (gzip on token sequence) — Pearce & Wiggins 2012
   Captures structural repetition. Higher = more repetition/structure.
2. Structureness Indicator (Wu & Yang 2020) — fitness scape over self-similarity
   Captures presence of long-range repeated sections.
3. 2nd-order pitch transition entropy
   Measures whether note-to-note moves are typical, not just notes.
4. Groove consistency (Wu & Yang 2020)
   1 - mean Hamming distance between bar-level rhythm patterns.

Usage:
    python3 musicality_metrics.py <dir_of_midis> [--label LABEL]
    python3 musicality_metrics.py --compare DIR1=label1 DIR2=label2 ...
"""
import os
import sys
import gzip
import math
import argparse
from glob import glob
from collections import Counter

import numpy as np
import mido


# ----------------------------- MIDI loading -----------------------------

def load_notes(path):
    """Return list of (onset_sec, dur_sec, pitch) sorted by onset."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM
    # find first set_tempo
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        else:
            continue
        break

    def t2s(t):
        return mido.tick2second(t, tpb, tempo)

    notes = []
    for tr in mid.tracks:
        abs_t = 0
        on = {}
        for msg in tr:
            abs_t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on[msg.note] = abs_t
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in on:
                    start = on.pop(msg.note)
                    notes.append((t2s(start), t2s(abs_t - start), msg.note))
    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


# --------------------------- Compression ratio --------------------------

def compression_ratio(notes, quantize_time=True):
    """Gzip-based compression ratio of the token sequence.
    Higher = more structural repetition. Random walks compress poorly.

    Tokens encode (delta_onset_quantized, pitch, dur_quantized).
    """
    if len(notes) < 2:
        return 1.0
    tokens = []
    prev_t = notes[0][0]
    for (t, d, p) in notes:
        dt = t - prev_t
        prev_t = t
        if quantize_time:
            dt_q = int(round(dt * 16))   # 16 bins/sec
            d_q = int(round(d * 16))
        else:
            dt_q = int(round(dt * 1000))
            d_q = int(round(d * 1000))
        tokens.append(f"{dt_q},{p},{d_q}")
    blob = " ".join(tokens).encode("utf-8")
    comp = gzip.compress(blob, compresslevel=9)
    return len(blob) / max(len(comp), 1)


# ------------------------ Structureness Indicator -----------------------

def structureness_indicator(notes, lower_sec=3.0, upper_sec=8.0):
    """Wu & Yang 2020 structureness indicator.
    Builds a self-similarity matrix from chroma per-beat-like windows
    and looks for the strongest off-diagonal repeated segment whose length
    falls within [lower_sec, upper_sec]. Returns a score in [0, 1].

    Simplified, faithful to the paper's intent: detects repeated sections.
    """
    if len(notes) < 8:
        return 0.0

    total = notes[-1][0] + notes[-1][1]
    if total < lower_sec * 2:
        return 0.0

    # Build chroma frames at 4 Hz
    fps = 4.0
    n_frames = max(int(math.ceil(total * fps)), 4)
    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    for (t, d, p) in notes:
        f0 = int(t * fps)
        f1 = max(int((t + d) * fps), f0 + 1)
        for f in range(f0, min(f1, n_frames)):
            chroma[f, p % 12] += 1.0

    # Normalize each frame
    norms = np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9
    chroma_n = chroma / norms

    # Self-similarity matrix
    S = chroma_n @ chroma_n.T   # n_frames x n_frames

    # Look for repeated segments: high values on off-diagonals
    L_lo = int(lower_sec * fps)
    L_hi = int(upper_sec * fps)
    n = n_frames
    best = 0.0
    # diag offset = how far apart the two segments are
    for offset in range(L_lo, n - L_lo):
        # extract diag
        diag = np.array([S[i, i + offset] for i in range(n - offset)])
        if len(diag) < L_lo:
            continue
        # sliding mean over window L_lo..L_hi
        for L in (L_lo, (L_lo + L_hi) // 2, L_hi):
            if L > len(diag):
                continue
            # max mean over windows of length L
            csum = np.cumsum(diag)
            window_means = (csum[L:] - csum[:-L]) / L if len(diag) > L else np.array([diag.mean()])
            if len(window_means) > 0:
                m = float(window_means.max())
                if m > best:
                    best = m
    return best


# ------------------- 2nd-order pitch transition entropy -----------------

def pitch_transition_entropy(notes):
    """Entropy of P(next_pitch_class | current_pitch_class).
    Lower than 1st-order entropy for structured music; near 1st-order entropy
    for random walks.
    """
    if len(notes) < 3:
        return 0.0
    pcs = [p % 12 for (_, _, p) in notes]
    bigram = Counter(zip(pcs[:-1], pcs[1:]))
    unigram = Counter(pcs[:-1])
    H = 0.0
    total = sum(bigram.values())
    for (a, b), c in bigram.items():
        p_ab = c / total
        p_b_given_a = c / unigram[a]
        H -= p_ab * math.log2(p_b_given_a + 1e-12)
    return H


# ------------------------- Groove consistency ---------------------------

def groove_consistency(notes, bar_sec=2.0, subdiv=16):
    """Wu & Yang 2020. Compares onset patterns across bars.
    1 - mean Hamming distance between bar onset vectors.
    Higher = more consistent groove.
    """
    if len(notes) < 8:
        return 0.0
    total = notes[-1][0] + notes[-1][1]
    n_bars = int(total // bar_sec)
    if n_bars < 2:
        return 0.0
    bars = np.zeros((n_bars, subdiv), dtype=np.int8)
    for (t, _, _) in notes:
        b = int(t // bar_sec)
        if b >= n_bars:
            continue
        s = int(((t % bar_sec) / bar_sec) * subdiv)
        if s >= subdiv:
            s = subdiv - 1
        bars[b, s] = 1
    # pairwise hamming
    dists = []
    for i in range(n_bars):
        for j in range(i + 1, n_bars):
            dists.append(np.mean(bars[i] != bars[j]))
    if not dists:
        return 0.0
    return float(1.0 - np.mean(dists))


# --------------------------- Driver ------------------------------------

def evaluate_dir(directory, label=None):
    files = sorted(glob(os.path.join(directory, "*.mid")))
    if not files:
        return None
    crs, sis, h2s, gcs, n_notes = [], [], [], [], []
    for f in files:
        try:
            notes = load_notes(f)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        if len(notes) < 4:
            continue
        crs.append(compression_ratio(notes))
        sis.append(structureness_indicator(notes))
        h2s.append(pitch_transition_entropy(notes))
        gcs.append(groove_consistency(notes))
        n_notes.append(len(notes))
    if not crs:
        return None
    return {
        "label": label or directory,
        "n_files": len(crs),
        "mean_notes": float(np.mean(n_notes)),
        "compression_ratio": float(np.mean(crs)),
        "structureness": float(np.mean(sis)),
        "trans_entropy_2nd": float(np.mean(h2s)),
        "groove_consistency": float(np.mean(gcs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+",
                    help="dir or label=dir entries")
    args = ap.parse_args()

    results = []
    for t in args.targets:
        if "=" in t:
            label, d = t.split("=", 1)
        else:
            label, d = os.path.basename(os.path.normpath(t)), t
        print(f"Evaluating {label} ({d}) ...")
        r = evaluate_dir(d, label)
        if r:
            results.append(r)
            print(f"  files={r['n_files']}  notes={r['mean_notes']:.0f}  "
                  f"CR={r['compression_ratio']:.3f}  SI={r['structureness']:.3f}  "
                  f"H2={r['trans_entropy_2nd']:.3f}  GC={r['groove_consistency']:.3f}")
        else:
            print("  (no files)")

    print("\n" + "=" * 92)
    print(f"{'Variant':<45} {'Files':>5} {'Notes':>6} {'CR':>7} {'SI':>7} {'H2':>7} {'GC':>7}")
    print("-" * 92)
    for r in results:
        print(f"{r['label']:<45} {r['n_files']:>5} {r['mean_notes']:>6.0f} "
              f"{r['compression_ratio']:>7.3f} {r['structureness']:>7.3f} "
              f"{r['trans_entropy_2nd']:>7.3f} {r['groove_consistency']:>7.3f}")
    print("=" * 92)
    print("CR = compression ratio (higher = more structural repetition)")
    print("SI = structureness indicator [0..1] (higher = stronger long-range repeats)")
    print("H2 = 2nd-order pitch transition entropy (lower vs 1st-order = more structure)")
    print("GC = groove consistency [0..1] (higher = more consistent rhythmic patterns)")


if __name__ == "__main__":
    main()

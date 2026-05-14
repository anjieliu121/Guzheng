#!/usr/bin/env python3
"""Transpose MIDI files from various pentatonic keys to D pentatonic.

Key detection from filename suffix (_A, _C, _D, _F, _G) and transposition:
  A → D: +5 semitones
  C → D: +2 semitones
  D → D:  0 semitones
  F → D: -3 semitones  (equivalently +9)
  G → D: -5 semitones  (equivalently +7)
"""

import argparse, os, sys
import mido

TRANSPOSE_MAP = {
    'A': 5,
    'C': 2,
    'D': 0,
    'F': -3,
    'G': -5,
}

GUZHENG_MIN = 38  # D1
GUZHENG_MAX = 86  # D6

D_PENTA_PCS = {2, 4, 6, 9, 11}

def nearest_d_penta(pitch):
    """Snap a pitch to nearest D-pentatonic note."""
    if pitch % 12 in D_PENTA_PCS:
        return pitch
    best = pitch
    best_dist = 999
    for offset in range(-2, 3):
        c = pitch + offset
        if c % 12 in D_PENTA_PCS and abs(offset) < best_dist:
            best = c
            best_dist = abs(offset)
    return max(GUZHENG_MIN, min(GUZHENG_MAX, best))


def detect_key(filename):
    """Extract key letter from filename like 'constrained_A_00.mid'."""
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    for p in parts:
        if p in TRANSPOSE_MAP:
            return p
    return None


def transpose_file(in_path, out_path, semitones):
    mid = mido.MidiFile(in_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type in ('note_on', 'note_off'):
                new_pitch = msg.note + semitones
                new_pitch = nearest_d_penta(new_pitch)
                new_pitch = max(GUZHENG_MIN, min(GUZHENG_MAX, new_pitch))
                msg.note = new_pitch
    mid.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.mid'))

    print(f"Transposing {len(files)} files to D pentatonic: {args.input_dir} -> {args.output_dir}")
    for fn in files:
        key = detect_key(fn)
        if key is None:
            print(f"  SKIP {fn}: cannot detect key from filename")
            continue
        semitones = TRANSPOSE_MAP[key]
        out_fn = fn  # keep original name
        transpose_file(os.path.join(args.input_dir, fn),
                       os.path.join(args.output_dir, out_fn),
                       semitones)
        print(f"  {fn}: key={key}, transpose={semitones:+d} semitones")

    print("Done.")


if __name__ == "__main__":
    main()

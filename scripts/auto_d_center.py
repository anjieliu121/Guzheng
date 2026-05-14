#!/usr/bin/env python3
"""Auto-shift D-pentatonic MIDI files to be D-centered (tonic = D).

If the most frequent pitch class is not D, shift every note down by one
pentatonic scale degree (E→D, F#→E, A→F#, B→A) until D becomes dominant.
This preserves the D-pentatonic string set and all rhythm/timing.

Also quantizes onsets and durations to a 16th-note grid at 120 BPM.

Usage:
    python3 scripts/auto_d_center.py --input_dir IN --output_dir OUT
"""
import argparse
import os
import sys
from collections import Counter

import mido

D_PENTA_PCS = {2, 4, 6, 9, 11}  # D E F# A B
SCALE = [2, 4, 6, 9, 11]  # pitch classes in order within one octave
GUZHENG_MIN = 38
GUZHENG_MAX = 86


def penta_note_below(midi_note):
    """Return MIDI note one pentatonic scale degree below."""
    pc = midi_note % 12
    octave = midi_note // 12
    if pc not in D_PENTA_PCS:
        return midi_note
    idx = SCALE.index(pc)
    if idx == 0:
        return (octave - 1) * 12 + SCALE[-1]
    return octave * 12 + SCALE[idx - 1]


def shift_all(notes, steps):
    """Shift every note down `steps` pentatonic scale degrees."""
    out = []
    for n, v in notes:
        cur = n
        for _ in range(steps):
            cur = penta_note_below(cur)
        out.append((max(GUZHENG_MIN, min(GUZHENG_MAX, cur)), v))
    return out


def count_pcs(notes):
    return Counter(n % 12 for n, _ in notes)


def load_notes(mid):
    """Return list of (on_tick, off_tick, pitch, vel)."""
    notes = []
    for track in mid.tracks:
        at = 0
        active = {}
        for msg in track:
            at += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (at, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    on, vel = active.pop(msg.note)
                    notes.append([on, at, msg.note, vel])
        for p, (on, vel) in active.items():
            notes.append([on, at, p, vel])
    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


def quantize(on_tick, off_tick, grid, min_dur_ticks):
    q_on = round(on_tick / grid) * grid
    dur = off_tick - on_tick
    q_dur = round(dur / grid) * grid
    if q_dur < min_dur_ticks:
        q_dur = min_dur_ticks
    return q_on, q_on + q_dur


def process_file(in_path, out_path, quantize_grid=True):
    mid = mido.MidiFile(in_path)
    tpb = mid.ticks_per_beat
    notes = load_notes(mid)
    if not notes:
        return None

    # Find best scale-degree shift: minimize E/A/B dominance, maximize D
    tnotes = [(p, v) for _, _, p, v in notes]
    best_steps = 0
    best_d_ratio = 0
    for steps in range(5):  # pentatonic has 5 positions
        shifted = shift_all(tnotes, steps)
        pcs = count_pcs(shifted)
        total = sum(pcs.values())
        d_ratio = pcs.get(2, 0) / total
        # Prefer D as most common
        most_common_pc = pcs.most_common(1)[0][0]
        score = d_ratio + (1.0 if most_common_pc == 2 else 0)
        if score > best_d_ratio:
            best_d_ratio = score
            best_steps = steps

    # Apply shift
    if best_steps > 0:
        shifted = shift_all(tnotes, best_steps)
        for i, (p, v) in enumerate(shifted):
            notes[i][2] = p
            notes[i][3] = v

    # Quantize to 16th-note grid (tpb/4), preserve long notes (>1.4s at 120 BPM)
    if quantize_grid:
        grid = tpb // 4  # 16th note
        tremolo_threshold_ticks = int(1.4 * tpb * 2)  # 1.4s at 120 BPM
        for i, n in enumerate(notes):
            dur = n[1] - n[0]
            q_on = round(n[0] / grid) * grid
            if dur >= tremolo_threshold_ticks:
                q_dur = dur  # preserve tremolo
            else:
                q_dur = round(dur / grid) * grid
                if q_dur < grid:
                    q_dur = grid
            notes[i][0] = q_on
            notes[i][1] = q_on + q_dur

    # Build output MIDI
    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()

    # Preserve tempo from first track if any
    tempo = 500000
    for src_track in mid.tracks:
        for msg in src_track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500000:
            break
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    events = []
    for on, off, p, v in notes:
        p = max(GUZHENG_MIN, min(GUZHENG_MAX, p))
        events.append((on, 0, mido.Message('note_on', note=p, velocity=v, time=0)))
        events.append((off, 1, mido.Message('note_off', note=p, velocity=64, time=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    last = 0
    for at, _, msg in events:
        delta = max(0, at - last)
        track.append(msg.copy(time=delta))
        last = at
    track.append(mido.MetaMessage('end_of_track', time=0))

    new_mid.tracks.append(track)
    new_mid.save(out_path)

    # Report
    final_pcs = count_pcs([(n[2], n[3]) for n in notes])
    total = sum(final_pcs.values())
    return {
        'steps_shifted': best_steps,
        'n_notes': len(notes),
        'd_pct': final_pcs.get(2, 0) / total * 100,
        'e_pct': final_pcs.get(4, 0) / total * 100,
        'fsharp_pct': final_pcs.get(6, 0) / total * 100,
        'a_pct': final_pcs.get(9, 0) / total * 100,
        'b_pct': final_pcs.get(11, 0) / total * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--no_quantize', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.mid'))
    print(f"Processing {len(files)} files → {args.output_dir}")
    totals = Counter()

    for fn in files:
        src = os.path.join(args.input_dir, fn)
        dst = os.path.join(args.output_dir, fn.replace('.mid', '_D.mid'))
        try:
            r = process_file(src, dst, quantize_grid=not args.no_quantize)
        except Exception as e:
            print(f"  {fn}: ERROR {e}")
            continue
        if r is None:
            print(f"  {fn}: SKIP (no notes)")
            continue
        totals['shifted_' + str(r['steps_shifted'])] += 1
        print(f"  {fn}: shift={r['steps_shifted']}  N={r['n_notes']:>3}  "
              f"D={r['d_pct']:.0f}% E={r['e_pct']:.0f}% F#={r['fsharp_pct']:.0f}% "
              f"A={r['a_pct']:.0f}% B={r['b_pct']:.0f}%")

    print(f"\nShift distribution: {dict(totals)}")


if __name__ == '__main__':
    main()

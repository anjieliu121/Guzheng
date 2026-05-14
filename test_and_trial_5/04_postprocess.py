#!/usr/bin/env python3
"""
Step 4: Post-process generated MIDI files.

Post-processing pipeline (matching the best original pipeline):
1. Merge all tracks to single track (single instrument)
2. Snap pitches to pentatonic scale
3. Clamp pitch range to MIDI 38-86 (guzheng range)
4. Limit polyphony to 4 simultaneous notes

Processes all generated MIDI in generated/<checkpoint>/<category>/ and writes
to generated/<checkpoint>/<category>_postprocessed/.
"""

import os
import mido
import numpy as np

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))

PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},
    "G": {7, 9, 11, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "A": {9, 11, 1, 4, 6},
    "F": {5, 7, 9, 0, 2},
}
PRESSED_PCS = {
    "D": {7, 1}, "G": {0, 6}, "C": {5, 11}, "A": {2, 8}, "F": {10, 4},
}
GUZHENG_PITCH_MIN = 38
GUZHENG_PITCH_MAX = 86
MAX_POLYPHONY = 4


def extract_notes(midi_path):
    """Extract all notes across all tracks as (pitch, onset_tick, duration_tick, velocity)."""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000

    notes = []
    for track in mid.tracks:
        abs_time = 0
        pending = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.note, msg.channel)] = (abs_time, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in pending:
                    onset, vel = pending.pop(key)
                    dur = abs_time - onset
                    if dur > 0:
                        notes.append([msg.note, onset, dur, vel])
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes, tpb, tempo


def detect_scale_from_filename(filename):
    """Detect scale from filename suffix."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in PENTATONIC_SCALES:
        return parts[1]
    return "D"


def snap_to_pentatonic(pitch, scale_name):
    """Snap a pitch to the nearest pentatonic note (including pressed notes)."""
    pcs = PENTATONIC_SCALES.get(scale_name, PENTATONIC_SCALES["D"])
    allowed_pcs = pcs | PRESSED_PCS.get(scale_name, set())

    pc = pitch % 12
    if pc in allowed_pcs:
        return pitch

    # Find nearest allowed pitch class
    best_dist = 999
    best_pc = pc
    for apc in allowed_pcs:
        dist = min(abs(pc - apc), 12 - abs(pc - apc))
        if dist < best_dist:
            best_dist = dist
            best_pc = apc

    # Reconstruct pitch with nearest pc
    octave = pitch // 12
    candidates = [octave * 12 + best_pc, (octave - 1) * 12 + best_pc, (octave + 1) * 12 + best_pc]
    candidates = [c for c in candidates if GUZHENG_PITCH_MIN <= c <= GUZHENG_PITCH_MAX]
    if not candidates:
        return max(GUZHENG_PITCH_MIN, min(GUZHENG_PITCH_MAX, pitch))
    return min(candidates, key=lambda c: abs(c - pitch))


def clamp_pitch(pitch):
    """Clamp pitch to guzheng range."""
    if pitch < GUZHENG_PITCH_MIN:
        # Shift up by octaves until in range
        while pitch < GUZHENG_PITCH_MIN:
            pitch += 12
    elif pitch > GUZHENG_PITCH_MAX:
        while pitch > GUZHENG_PITCH_MAX:
            pitch -= 12
    return pitch


def limit_polyphony(notes, max_poly=MAX_POLYPHONY):
    """Keep at most max_poly simultaneous notes by removing lowest-velocity overlapping notes."""
    if not notes:
        return notes

    # Build event list
    events = []
    for i, (pitch, onset, dur, vel) in enumerate(notes):
        events.append((onset, 1, vel, i))         # note on
        events.append((onset + dur, 0, vel, i))   # note off
    events.sort(key=lambda e: (e[0], e[1]))  # offs before ons at same time

    active = {}  # note_idx -> velocity
    removed = set()

    for tick, is_on, vel, idx in events:
        if idx in removed:
            continue
        if is_on:
            active[idx] = vel
            if len(active) > max_poly:
                # Remove lowest-velocity active note
                min_idx = min(active, key=lambda k: active[k])
                removed.add(min_idx)
                del active[min_idx]
        else:
            active.pop(idx, None)

    return [n for i, n in enumerate(notes) if i not in removed]


def postprocess_midi(input_path, output_path, scale_name):
    """Apply full post-processing pipeline to a MIDI file."""
    notes, tpb, tempo = extract_notes(input_path)

    if not notes:
        return 0

    # 1. All notes already merged across tracks by extract_notes

    # 2. Snap pitches to pentatonic
    for n in notes:
        n[0] = snap_to_pentatonic(n[0], scale_name)

    # 3. Clamp pitch range
    for n in notes:
        n[0] = clamp_pitch(n[0])

    # 4. Limit polyphony
    notes = limit_polyphony(notes, MAX_POLYPHONY)

    # Write to single-track MIDI
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    track.append(mido.MetaMessage("track_name", name="Guzheng"))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))

    events = []
    for pitch, onset, dur, vel in notes:
        events.append((onset, 1, pitch, vel))
        events.append((onset + dur, 0, pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))

    prev = 0
    for abs_t, is_on, pitch, vel in events:
        delta = max(0, abs_t - prev)
        kind = "note_on" if is_on else "note_off"
        track.append(mido.Message(kind, note=pitch, velocity=vel, time=delta, channel=0))
        prev = abs_t

    track.append(mido.MetaMessage("end_of_track"))
    mid.save(output_path)
    return len(notes)


def main():
    print("=" * 60)
    print("STEP 4: POST-PROCESSING (Trial 4)")
    print("=" * 60)

    gen_dir = os.path.join(TRIAL_ROOT, "generated")
    if not os.path.isdir(gen_dir):
        print(f"ERROR: generated/ directory not found. Run 03_generate.py first!")
        return

    total_processed = 0
    total_skipped = 0

    for ckpt_name in sorted(os.listdir(gen_dir)):
        ckpt_dir = os.path.join(gen_dir, ckpt_name)
        if not os.path.isdir(ckpt_dir):
            continue

        for category in ["val", "test", "synthetic"]:
            src_dir = os.path.join(ckpt_dir, category)
            if not os.path.isdir(src_dir):
                continue

            dst_dir = os.path.join(ckpt_dir, f"{category}_postprocessed")
            os.makedirs(dst_dir, exist_ok=True)

            midi_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".mid"))
            print(f"\n{ckpt_name}/{category}: {len(midi_files)} files")

            for fname in midi_files:
                scale = detect_scale_from_filename(fname)
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)

                n_notes = postprocess_midi(src_path, dst_path, scale)
                if n_notes > 0:
                    print(f"  {fname}: {n_notes} notes (scale: {scale})")
                    total_processed += 1
                else:
                    print(f"  {fname}: SKIPPED (no notes)")
                    total_skipped += 1

    print(f"\n{'='*60}")
    print(f"Post-processing complete: {total_processed} files processed, {total_skipped} skipped")


if __name__ == "__main__":
    main()

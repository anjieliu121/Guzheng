#!/usr/bin/env python3
"""
Post-process generated MIDI files: snap to pentatonic scale, constrain pitch range,
limit polyphony. Much faster than constrained decoding.

Run: python3 scripts/postprocess_midi.py --input_dir outputs/midirwkv_finetuned --scale D
"""

import os, argparse, copy
import mido

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"

PENTATONIC_SCALES = {
    "D": [2, 4, 6, 9, 11],    # D E F# A B
    "G": [7, 9, 11, 2, 4],    # G A B D E
    "C": [0, 2, 4, 7, 9],     # C D E G A
    "A": [9, 11, 1, 4, 6],    # A B C# E F#
    "F": [5, 7, 9, 0, 2],     # F G A C D
}
PRESSED_PCS = {
    "D": [7, 1], "G": [0, 6], "C": [5, 11], "A": [2, 8], "F": [10, 4],
}

GUZHENG_PITCH_MIN = 38
GUZHENG_PITCH_MAX = 86


def get_valid_pitches(scale_name, include_pressed=True):
    """Get all valid MIDI pitches for a scale within guzheng range."""
    pcs = set(PENTATONIC_SCALES.get(scale_name, PENTATONIC_SCALES["D"]))
    if include_pressed:
        pcs |= set(PRESSED_PCS.get(scale_name, []))
    valid = []
    for midi in range(GUZHENG_PITCH_MIN, GUZHENG_PITCH_MAX + 1):
        if midi % 12 in pcs:
            valid.append(midi)
    return sorted(valid)


def snap_to_nearest(pitch, valid_pitches):
    """Snap a pitch to the nearest valid pitch."""
    if pitch in valid_pitches:
        return pitch
    distances = [(abs(pitch - v), v) for v in valid_pitches]
    distances.sort()
    return distances[0][1]


def detect_scale(midi_path):
    """Detect scale from filename or content."""
    base = os.path.splitext(os.path.basename(midi_path))[0]
    for scale in PENTATONIC_SCALES:
        if f"_{scale}" in base or base.endswith(f"_{scale}"):
            return scale
    # Analyze pitch content
    try:
        mid = mido.MidiFile(midi_path)
        pc_counts = [0] * 12
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    pc_counts[msg.note % 12] += 1
        best_scale = "D"
        best_score = 0
        for name, pcs in PENTATONIC_SCALES.items():
            score = sum(pc_counts[pc] for pc in pcs)
            if score > best_score:
                best_score = score
                best_scale = name
        return best_scale
    except Exception:
        return "D"


def postprocess_midi(input_path, output_path, scale_name="D",
                     max_simultaneous=4, include_pressed=True):
    """Post-process a MIDI file with pentatonic and range constraints."""
    try:
        mid = mido.MidiFile(input_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    valid_pitches = get_valid_pitches(scale_name, include_pressed)
    stats = {"snapped": 0, "clamped": 0, "removed_poly": 0, "total": 0}

    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        active_notes = set()

        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                stats["total"] += 1
                original = msg.note

                # Snap to pentatonic scale within guzheng range
                new_pitch = snap_to_nearest(original, valid_pitches)
                if new_pitch != original:
                    stats["snapped"] += 1

                # Check polyphony limit
                if len(active_notes) >= max_simultaneous:
                    stats["removed_poly"] += 1
                    # Convert to note_off (skip this note)
                    new_track.append(mido.Message('note_off', note=new_pitch,
                                                   velocity=0, time=msg.time,
                                                   channel=msg.channel))
                    continue

                active_notes.add(new_pitch)
                new_msg = msg.copy(note=new_pitch)
                new_track.append(new_msg)

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # Find the closest active note to turn off
                original = msg.note
                snapped = snap_to_nearest(original, valid_pitches)
                if snapped in active_notes:
                    active_notes.discard(snapped)
                new_track.append(msg.copy(note=snapped))
            else:
                new_track.append(msg.copy())

        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", default=None)
    parser.add_argument("--out_suffix", default="_constrained")
    parser.add_argument("--scale", default=None, help="Force scale (D/G/C/A/F)")
    parser.add_argument("--max_simultaneous", type=int, default=4)
    args = parser.parse_args()

    if not args.input_dirs:
        # Default: process all existing generated output directories
        candidates = [
            os.path.join(ROOT, "archive/outputs/midirwkv_pretrained"),
            os.path.join(ROOT, "archive/outputs/midirwkv_finetuned"),
            os.path.join(ROOT, "archive/outputs/moonbeam_pretrained"),
            os.path.join(ROOT, "archive/outputs/moonbeam_finetuned"),
            os.path.join(ROOT, "archive/outputs/moonbeam_finetuned_t95"),
        ]
        args.input_dirs = [d for d in candidates if os.path.isdir(d)]

    for input_dir in args.input_dirs:
        dir_name = os.path.basename(input_dir)
        out_dir = os.path.join(ROOT, f"outputs/{dir_name}{args.out_suffix}")
        os.makedirs(out_dir, exist_ok=True)

        midi_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])
        if not midi_files:
            continue

        print(f"\n=== Processing {dir_name} ({len(midi_files)} files) ===")
        total_stats = {"snapped": 0, "clamped": 0, "removed_poly": 0, "total": 0}

        for fn in midi_files:
            input_path = os.path.join(input_dir, fn)
            output_path = os.path.join(out_dir, fn)

            scale = args.scale or detect_scale(input_path)
            stats = postprocess_midi(input_path, output_path, scale_name=scale,
                                     max_simultaneous=args.max_simultaneous)
            if stats:
                for k in total_stats:
                    total_stats[k] += stats[k]
                pct = stats["snapped"] / max(stats["total"], 1) * 100
                print(f"  {fn} (scale: {scale}): {stats['snapped']}/{stats['total']} snapped ({pct:.0f}%), "
                      f"{stats['removed_poly']} poly removed")

        pct = total_stats["snapped"] / max(total_stats["total"], 1) * 100
        print(f"  Total: {total_stats['snapped']}/{total_stats['total']} notes snapped ({pct:.0f}%)")
        print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()

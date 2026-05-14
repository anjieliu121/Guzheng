#!/usr/bin/env python3
"""
Step 3: Data augmentation for training data.

Augmentation strategies:
1. Tempo jitter: scale all onset/duration times by a random factor (0.85-1.15)
2. Velocity humanization: add Gaussian noise to velocities
3. Micro-timing: small random onset perturbations

Creates 2 augmented copies per training file -> ~3x training data.
Augmented files go into data/train/ alongside originals (prefixed with aug_).
"""

import os
import random
import mido
import numpy as np

TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(TRIAL_ROOT, "data", "train")
SEED = 42
N_AUGMENTATIONS = 2


def extract_notes(midi_path):
    """Extract notes as list of (pitch, onset_tick, duration_tick, velocity)."""
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


def augment_notes(notes, rng, tempo_factor=None, vel_sigma=8, timing_sigma=10):
    """Apply augmentation to a list of notes.

    Args:
        notes: list of [pitch, onset, duration, velocity]
        rng: numpy random generator
        tempo_factor: if None, randomly sample from [0.85, 1.15]
        vel_sigma: std dev for velocity noise
        timing_sigma: std dev for onset jitter (in ticks)
    """
    if not notes:
        return notes

    aug = []
    if tempo_factor is None:
        tempo_factor = rng.uniform(0.85, 1.15)

    for pitch, onset, dur, vel in notes:
        # Tempo scaling
        new_onset = int(onset * tempo_factor)
        new_dur = max(1, int(dur * tempo_factor))

        # Micro-timing jitter on onset
        jitter = int(rng.normal(0, timing_sigma))
        new_onset = max(0, new_onset + jitter)

        # Velocity humanization
        vel_noise = int(rng.normal(0, vel_sigma))
        new_vel = max(1, min(127, vel + vel_noise))

        aug.append([pitch, new_onset, new_dur, new_vel])

    # Re-sort by onset
    aug.sort(key=lambda n: (n[1], n[0]))
    return aug


def notes_to_midi(notes, output_path, tpb=480, tempo=500000):
    """Write notes to MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    track.append(mido.MetaMessage("track_name", name="Guzheng"))

    events = []
    for pitch, onset, dur, vel in notes:
        events.append((onset, 1, pitch, vel))
        events.append((onset + dur, 0, pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))

    prev = 0
    for abs_t, is_on, pitch, vel in events:
        delta = max(0, abs_t - prev)
        kind = "note_on" if is_on else "note_off"
        track.append(mido.Message(kind, note=pitch, velocity=vel, time=delta))
        prev = abs_t

    track.append(mido.MetaMessage("end_of_track"))
    mid.save(output_path)


def main():
    print("=" * 60)
    print("STEP 3: DATA AUGMENTATION")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    if not os.path.isdir(TRAIN_DIR):
        print(f"Training directory not found: {TRAIN_DIR}")
        print("Run 02_split_data.py first!")
        return

    # Get original training files (exclude augmented files)
    original_files = sorted(
        f for f in os.listdir(TRAIN_DIR)
        if f.endswith(".mid") and not f.startswith("aug_")
    )
    print(f"Original training files: {len(original_files)}")

    augmented_count = 0
    for fname in original_files:
        src_path = os.path.join(TRAIN_DIR, fname)
        notes, tpb, tempo = extract_notes(src_path)

        if len(notes) < 10:
            print(f"  Skip {fname}: too few notes ({len(notes)})")
            continue

        for aug_idx in range(N_AUGMENTATIONS):
            aug_notes = augment_notes(notes, rng)
            base = os.path.splitext(fname)[0]
            aug_fname = f"aug_{aug_idx}_{base}.mid"
            aug_path = os.path.join(TRAIN_DIR, aug_fname)
            notes_to_midi(aug_notes, aug_path, tpb, tempo)
            augmented_count += 1

    total = len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".mid")])
    print(f"\nAugmented files created: {augmented_count}")
    print(f"Total training files: {total} ({len(original_files)} original + {augmented_count} augmented)")


if __name__ == "__main__":
    main()

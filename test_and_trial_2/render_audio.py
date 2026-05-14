#!/usr/bin/env python3
"""
Render MIDI files to WAV audio using FluidSynth.
Run: python3 scripts/render_audio.py [--input_dirs dir1 dir2 ...] [--out_dir outputs/audio]
"""

import os, argparse, wave, struct
import numpy as np
import pretty_midi

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"


def render_midi_to_wav(midi_path, wav_path, sf2_path=None, fs=44100):
    """Render MIDI to WAV using FluidSynth via pretty_midi."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        if sf2_path:
            audio = pm.fluidsynth(fs=fs, sf2_path=sf2_path)
        else:
            audio = pm.fluidsynth(fs=fs)

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_16bit.tobytes())

        return True
    except Exception as e:
        print(f"  ERROR rendering {midi_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", default=None,
                        help="Directories containing MIDI files to render")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "outputs/audio"))
    parser.add_argument("--sf2", default=None, help="Path to SoundFont (.sf2)")
    parser.add_argument("--max_per_dir", type=int, default=5,
                        help="Max files to render per directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Default directories to render
    if not args.input_dirs:
        candidates = [
            os.path.join(ROOT, "outputs/midirwkv_state_constrained"),
            os.path.join(ROOT, "outputs/midirwkv_state_constrained_unconstrained"),
            os.path.join(ROOT, "outputs/midirwkv_lora_constrained"),
            os.path.join(ROOT, "archive/outputs/midirwkv_finetuned"),
            os.path.join(ROOT, "archive/outputs/moonbeam_finetuned"),
            os.path.join(ROOT, "MIDI"),  # training examples for reference
        ]
        args.input_dirs = [d for d in candidates if os.path.isdir(d)]

    total_rendered = 0
    for input_dir in args.input_dirs:
        dir_name = os.path.basename(input_dir)
        midi_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir) if f.endswith(".mid")
        ])[:args.max_per_dir]

        if not midi_files:
            continue

        sub_dir = os.path.join(args.out_dir, dir_name)
        os.makedirs(sub_dir, exist_ok=True)

        print(f"\nRendering {len(midi_files)} files from {dir_name}/")
        for midi_path in midi_files:
            base = os.path.splitext(os.path.basename(midi_path))[0]
            wav_path = os.path.join(sub_dir, f"{base}.wav")
            ok = render_midi_to_wav(midi_path, wav_path, sf2_path=args.sf2)
            if ok:
                print(f"  {base}.wav")
                total_rendered += 1

    print(f"\nTotal: {total_rendered} audio files rendered to {args.out_dir}")


if __name__ == "__main__":
    main()

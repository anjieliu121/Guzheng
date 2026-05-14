#!/usr/bin/env python3
"""
Transpose one MIDI file (or every MIDI in a directory) into five versions
(D, G, F, C, A major pentatonic keys).

- Identifies source key from MetaMessage key_signature when present; otherwise
  infers which of {D,G,F,C,A} best matches note pitch-class histogram against
  each scale's pentatonic pitch classes from guzheng_scales.json entries.
- Applies chromatic transpose + optional uniform octave shift k*12 so all notes
  stay in [0,127] without clipping (chooses k that best centers the result in the
  target guzheng compass when possible).
- Writes: MIDI_transposed/{basename}_{scale}.mid only when the transposed
  note range lies inside that scale’s guzheng compass (entries + pressed_strings);
  otherwise that target is skipped (no file; stale file at that path is removed).
- Reports note range per target and note_on counts for written files only.

Usage:
  python scripts/transpose_midi_to_pentatonic_scales.py MIDI/foo.mid
  python scripts/transpose_midi_to_pentatonic_scales.py MIDI --recursive
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Iterable

import mido

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
DEFAULT_SCALES_JSON = os.path.join(REPO, "metadata/guzheng_scales.json")
OUT_DIR_NAME = "MIDI_transposed"
TARGET_SCALES = ("D", "G", "F", "C", "A")

# Major / minor tonic name -> pitch class (0=C)
_NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def spn_to_midi(s: str) -> int:
    s = s.replace("♭", "b").replace("♯", "#")
    m = re.match(r"^([A-Ga-g])([#b]?)(\d+)$", s)
    if not m:
        raise ValueError(f"Bad SPN: {s!r}")
    name = m.group(1).upper() + m.group(2)
    octave = int(m.group(3))
    pc = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }[name]
    return (octave + 1) * 12 + pc


def midi_to_spn(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{(n // 12) - 1}"


def load_scale_block(scales_path: str, scale_name: str) -> dict:
    with open(scales_path, encoding="utf-8") as f:
        data = json.load(f)
    for s in data["scales"]:
        if s["scale"] == scale_name:
            return s
    raise KeyError(scale_name)


def pentatonic_pitch_classes(entries: list[dict]) -> set[int]:
    return {spn_to_midi(e["spn"]) % 12 for e in entries}


def guzheng_compass_midi(scale_block: dict) -> tuple[int, int]:
    """Min/max MIDI across entries + pressed_strings."""
    vals: list[int] = []
    for e in scale_block["entries"] + scale_block.get("pressed_strings", []):
        vals.append(spn_to_midi(e["spn"]))
    return min(vals), max(vals)


def collect_pitch_classes_and_range(path: str) -> tuple[Counter, int, int, int]:
    """Return (pc Counter, min_note, max_note, note_on_count)."""
    mid = mido.MidiFile(path)
    pcs: Counter = Counter()
    min_n = 127
    max_n = 0
    count = 0
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "note_on" and msg.velocity > 0:
                n = msg.note
                pcs[n % 12] += 1
                min_n = min(min_n, n)
                max_n = max(max_n, n)
                count += 1
    if count == 0:
        return pcs, 60, 60, 0
    return pcs, min_n, max_n, count


def parse_key_to_tonic_pc(key_str: str | None) -> int | None:
    """
    Parse mido key_signature key field (e.g. 'D', 'Dm', 'Bb', 'F#').
    Returns pitch class of tonic (minor: first letter is tonic).
    """
    if not key_str:
        return None
    s = key_str.strip()
    m = re.match(r"^([A-Ga-g])([#b]?)(m)?$", s)
    if not m:
        return None
    name = m.group(1).upper() + m.group(2)
    if name not in _NOTE_TO_PC:
        return None
    return _NOTE_TO_PC[name]


def infer_scale_from_histogram(
    pcs: Counter, scales_path: str, candidates: Iterable[str]
) -> str:
    best = next(iter(candidates))
    best_score = -1.0
    for name in candidates:
        sc = load_scale_block(scales_path, name)
        pent = pentatonic_pitch_classes(sc["entries"])
        total = sum(pcs.values()) or 1
        hit = sum(pcs.get(pc, 0) for pc in pent)
        score = hit / total
        if score > best_score:
            best_score = score
            best = name
    return best


def signed_semitone_shift(src_pc: int, tgt_pc: int) -> int:
    d = (tgt_pc - src_pc) % 12
    if d > 6:
        d -= 12
    return d


def choose_octave_shift(
    min_n: int, max_n: int, delta: int, gmin: int, gmax: int
) -> int | None:
    """
    Find integer k such that [min_n,max_n] + delta + 12*k lies in [0,127].
    Prefer k that centers the transposed span near the guzheng compass [gmin,gmax].
    """
    lo = min_n + delta
    hi = max_n + delta
    valid: list[int] = []
    for k in range(-11, 12):
        nlo = lo + 12 * k
        nhi = hi + 12 * k
        if nlo >= 0 and nhi <= 127:
            valid.append(k)
    if not valid:
        return None
    mid_g = (gmin + gmax) // 2
    center = (lo + hi) // 2
    return min(valid, key=lambda k: abs(center + 12 * k - mid_g))


def transpose_midifile(
    src_path: str,
    out_path: str,
    delta: int,
    k_oct: int,
    target_key_name: str,
) -> int:
    """Write transposed MIDI; returns note_on count."""
    mid = mido.MidiFile(src_path)
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    count = 0
    for tr in mid.tracks:
        new_tr = mido.MidiTrack()
        for msg in tr:
            if msg.is_meta:
                if msg.type == "key_signature":
                    new_tr.append(
                        mido.MetaMessage(
                            "key_signature", key=target_key_name, time=msg.time
                        )
                    )
                else:
                    new_tr.append(msg.copy())
                continue
            if hasattr(msg, "note") and msg.type in (
                "note_on",
                "note_off",
                "polytouch",
            ):
                nn = msg.note + delta + 12 * k_oct
                if nn < 0 or nn > 127:
                    raise ValueError(f"Internal: note out of range after shift: {nn}")
                if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                    count += 1
                new_tr.append(msg.copy(note=int(nn), time=msg.time))
            else:
                new_tr.append(msg.copy(time=msg.time))
        out.tracks.append(new_tr)
    out.save(out_path)
    return count


def collect_input_midis(path: str, recursive: bool) -> list[str]:
    src = os.path.abspath(path)
    if os.path.isfile(src):
        if src.lower().endswith((".mid", ".midi")):
            return [src]
        return []
    if not os.path.isdir(src):
        return []

    out: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(src):
            for fn in files:
                if fn.lower().endswith((".mid", ".midi")):
                    out.append(os.path.join(root, fn))
    else:
        for fn in sorted(os.listdir(src)):
            p = os.path.join(src, fn)
            if os.path.isfile(p) and fn.lower().endswith((".mid", ".midi")):
                out.append(p)
    return sorted(out)


def process_one_midi(src: str, args: argparse.Namespace, compass: dict[str, tuple[int, int]]) -> int:
    pcs, min_n, max_n, src_note_count = collect_pitch_classes_and_range(src)
    basename = os.path.splitext(os.path.basename(src))[0]

    # Key signature
    mid = mido.MidiFile(src)
    key_str = None
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "key_signature":
                key_str = msg.key
                break
        if key_str is not None:
            break

    src_pc = parse_key_to_tonic_pc(key_str)
    if src_pc is None and key_str:
        inferred = infer_scale_from_histogram(pcs, args.scales_json, TARGET_SCALES)
        src_pc = parse_key_to_tonic_pc(inferred)
        detection = (
            f"key_signature {key_str!r} (unparsed) → inferred {inferred} "
            f"(tonic pc={src_pc})"
        )
    elif src_pc is None:
        inferred = infer_scale_from_histogram(pcs, args.scales_json, TARGET_SCALES)
        src_pc = parse_key_to_tonic_pc(inferred)
        if src_pc is None:
            src_pc = 2  # D
        detection = f"no key_signature; inferred {inferred} (tonic pc={src_pc})"
    else:
        detection = f"key_signature meta: {key_str!r} → tonic pitch class {src_pc}"

    print(f"Source: {src}")
    print(f"  Identified scale / tonic: {detection}")
    print(f"  Original note range: {midi_to_spn(min_n)}–{midi_to_spn(max_n)} (MIDI {min_n}–{max_n})")
    print(f"  Original note_on count: {src_note_count}")
    print()

    written: list[tuple[str, str, int]] = []

    for tgt in TARGET_SCALES:
        tgt_pc = parse_key_to_tonic_pc(tgt)
        assert tgt_pc is not None
        delta = signed_semitone_shift(src_pc, tgt_pc)
        lo = min_n + delta
        hi = max_n + delta
        gmin, gmax = compass[tgt]

        k = choose_octave_shift(min_n, max_n, delta, gmin, gmax)
        if k is None:
            print(f"ERROR: {tgt}: cannot fit transposed notes in 0–127 without clipping.", file=sys.stderr)
            return 1

        nmin = min_n + delta + 12 * k
        nmax = max_n + delta + 12 * k
        in_range = nmin >= gmin and nmax <= gmax

        out_path = os.path.join(args.out_dir, f"{basename}_{tgt}.mid")
        print(f"{tgt}: transpose Δ={delta:+d} semitones, octave shift k={k:+d} (×12)")
        print(f"     Range: {midi_to_spn(nmin)}–{midi_to_spn(nmax)} (MIDI {nmin}–{nmax})")
        print(f"     guzheng compass for {tgt}: MIDI {gmin}–{gmax}")

        if not in_range:
            if os.path.isfile(out_path):
                os.remove(out_path)
            print(f"     SKIPPED — outside compass (no file written)")
            print()
            continue

        ncount = transpose_midifile(src, out_path, delta, k, tgt)
        written.append((tgt, out_path, ncount))
        print(f"     Wrote: {out_path}")
        print(f"     note_on count: {ncount}")
        print()

    print("Summary note_on counts (written files only):")
    if not written:
        print("  (none)")
    for tgt, out_path, ncount in written:
        print(f"  {os.path.basename(out_path)}: {ncount}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_path", help="Input .mid/.midi file or directory")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="When input_path is a directory, recurse into subdirectories",
    )
    ap.add_argument(
        "--scales-json",
        default=DEFAULT_SCALES_JSON,
        help="Path to guzheng_scales.json",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(REPO, OUT_DIR_NAME),
        help=f"Output directory (default: {OUT_DIR_NAME}/ under repo)",
    )
    args = ap.parse_args()

    sources = collect_input_midis(args.input_path, recursive=args.recursive)
    if not sources:
        print(f"No MIDI files found from input: {args.input_path}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    compass = {
        name: guzheng_compass_midi(load_scale_block(args.scales_json, name))
        for name in TARGET_SCALES
    }

    rc = 0
    for i, src in enumerate(sources):
        if i > 0:
            print("\n" + "=" * 72)
        rc_one = process_one_midi(src, args, compass)
        if rc_one != 0:
            rc = rc_one
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

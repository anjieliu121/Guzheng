#!/usr/bin/env python3
"""
Scan a MIDI file for notes whose pitch class is outside the scale's pentatonic
set (derived from metadata/guzheng_scales.json: unique pitch classes among `entries` only).

Reports non-pentatonic note-ons (all tracks). With --apply, appends any such
pitches not already listed in entries + pressed_strings for that scale as new
dictionaries in `pressed_strings` (jianpu key: m<MIDI>, e.g. m65 for F4).

Usage:
  python scripts/scan_midi_non_pentatonic.py MIDI/foo.mid --scale D
  python scripts/scan_midi_non_pentatonic.py MIDI/foo.mid --scale D --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import mido

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
DEFAULT_SCALES = os.path.join(REPO, "metadata", "guzheng_scales.json")

# Pitch-class sets for note names (chromatic)
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
    return f"{PC_NAMES[n % 12]}{(n // 12) - 1}"


def staff_letter_from_pc(pc: int) -> str:
    return PC_NAMES[pc % 12]


def pentatonic_pitch_classes_from_entries(entries: list[dict]) -> set[int]:
    """Unique pitch classes (0–11) from open-string entries only."""
    pcs: set[int] = set()
    for e in entries:
        m = spn_to_midi(e["spn"])
        pcs.add(m % 12)
    return pcs


def known_midi_pitches(scale_block: dict) -> set[int]:
    s = set()
    for e in scale_block["entries"] + scale_block.get("pressed_strings", []):
        s.add(spn_to_midi(e["spn"]))
    return s


def known_jianpu_keys(scale_block: dict) -> set[str]:
    keys: set[str] = set()
    for e in scale_block["entries"] + scale_block.get("pressed_strings", []):
        keys.add(e["jianpu"])
    return keys


def nearest_string_number(entries: list[dict], midi_note: int) -> int:
    """Pick stringNumber from the entry whose pitch is closest to midi_note."""
    best_sn = entries[0]["stringNumber"]
    best_d = abs(spn_to_midi(entries[0]["spn"]) - midi_note)
    for e in entries[1:]:
        d = abs(spn_to_midi(e["spn"]) - midi_note)
        if d < best_d:
            best_d = d
            best_sn = e["stringNumber"]
    return best_sn


def collect_note_on_pitches(path: str) -> dict[int, int]:
    """Return map pitch -> count of note_ons (all tracks)."""
    mid = mido.MidiFile(path)
    counts: dict[int, int] = {}
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "note_on" and msg.velocity > 0:
                counts[msg.note] = counts.get(msg.note, 0) + 1
    return counts


def make_auto_entry(midi_note: int, string_number: int) -> dict:
    spn = midi_to_spn(midi_note)
    pc = midi_note % 12
    return {
        "stringNumber": string_number,
        "jianpu": f"m{midi_note}",
        "staff": staff_letter_from_pc(pc),
        "spn": spn,
        "midi": midi_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("midi", help="Path to .mid file")
    ap.add_argument(
        "--scale",
        required=True,
        help="Scale key in metadata/guzheng_scales.json (e.g. D, G, C, A, Bb, F)",
    )
    ap.add_argument(
        "--scales-json",
        default=DEFAULT_SCALES,
        help="Path to metadata/guzheng_scales.json",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Append missing non-pentatonic pitches to pressed_strings",
    )
    args = ap.parse_args()

    midi_path = os.path.abspath(args.midi)
    if not os.path.isfile(midi_path):
        print(f"Not found: {midi_path}", file=sys.stderr)
        return 1

    with open(args.scales_json, encoding="utf-8") as f:
        data = json.load(f)

    scale_name = args.scale
    sc = next((s for s in data["scales"] if s["scale"] == scale_name), None)
    if sc is None:
        print(
            f"Unknown scale {scale_name!r}. Options: "
            f"{[s['scale'] for s in data['scales']]}",
            file=sys.stderr,
        )
        return 1

    pent_pc = pentatonic_pitch_classes_from_entries(sc["entries"])
    known = known_midi_pitches(sc)
    jianpu_keys = known_jianpu_keys(sc)

    pitch_counts = collect_note_on_pitches(midi_path)
    sorted_pitches = sorted(pitch_counts.keys())

    non_pent: list[int] = []
    for p in sorted_pitches:
        if p % 12 not in pent_pc:
            non_pent.append(p)

    print(f"Scale {scale_name}: pentatonic pitch classes (mod 12) = {sorted(pent_pc)}")
    print(f"MIDI file: {midi_path}")
    print()

    if not non_pent:
        print("No non-pentatonic note-ons (by pitch class).")
        return 0

    print("Non-pentatonic pitches (unique):")
    for p in non_pent:
        spn = midi_to_spn(p)
        c = pitch_counts[p]
        in_json = p in known
        print(f"  {p:3}  {spn:4}  count={c:5}  {'already in JSON' if in_json else 'NOT in JSON'}")
    print()

    to_add = [p for p in non_pent if p not in known]
    if not to_add:
        print("All non-pentatonic pitches are already listed in entries or pressed_strings.")
        return 0

    print(f"Pitches to add to pressed_strings ({len(to_add)}): {[midi_to_spn(p) for p in to_add]}")

    if not args.apply:
        print("\n(dry-run; pass --apply to write to metadata/guzheng_scales.json)")
        return 0

    entries = sc["entries"]
    new_rows = []
    for p in sorted(to_add):
        jp = f"m{p}"
        if jp in jianpu_keys:
            print(f"Skip jianpu collision {jp!r}", file=sys.stderr)
            continue
        sn = nearest_string_number(entries, p)
        row = make_auto_entry(p, sn)
        new_rows.append(row)
        jianpu_keys.add(jp)

    pressed = list(sc.get("pressed_strings", []))
    pressed.extend(new_rows)
    pressed.sort(key=lambda e: spn_to_midi(e["spn"]))
    sc["pressed_strings"] = pressed

    with open(args.scales_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(new_rows)} entr(y/ies) to {args.scales_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

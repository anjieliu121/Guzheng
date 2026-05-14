#!/usr/bin/env python3
"""
Rewrite MIDI note numbers so chromatic variants match pentatonic targets
(see notes/data_processing.md: 4#,→5,, 5#→6, 3b→3, 3b'→3', 7b,,→7,,, 7b,→7,).

Uses the same scale detection and mapping as check_midi_note_quality.py
(metadata/guzheng_scales.json + jianpu shift map + pitch-class fallback).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mido

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_midi_note_quality as cmq  # noqa: E402


def iter_mid_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.mid"))
    return sorted(root.glob("*.mid"))


def main() -> int:
    p = argparse.ArgumentParser(description="Correct chromatic guzheng MIDI to pentatonic targets.")
    p.add_argument(
        "paths",
        nargs="*",
        default=[str(cmq.REPO_ROOT / "MIDI")],
        help="Files or directories (default: repo MIDI/)",
    )
    p.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories")
    args = p.parse_args()

    models = cmq.load_scale_models()
    if not models:
        print("No scale models loaded; check metadata/guzheng_scales.json", file=sys.stderr)
        return 1

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw).resolve()
        if path.is_file() and path.suffix.lower() == ".mid":
            files.append(path)
        elif path.is_dir():
            files.extend(iter_mid_files(path, args.recursive))
        else:
            print(f"Skip (not .mid file or dir): {path}", file=sys.stderr)

    files = sorted(set(files))
    if not files:
        print("No .mid files found.", file=sys.stderr)
        return 1

    total_corrections = 0
    for mid_path in files:
        mid = mido.MidiFile(str(mid_path))
        scale = cmq.detect_scale(mid, models)
        model = models.get(scale)
        if not model:
            print(f"{mid_path.name}: unknown scale {scale!r}, skip")
            continue
        n = cmq.autocorrect_non_scale_chromatic_notes(mid, model)
        if n:
            mid.save(str(mid_path))
            total_corrections += n
        print(f"{mid_path.name}: scale={scale}, chromatic_notes_corrected={n}")

    print(f"Done. Files processed: {len(files)}, total note_on corrections: {total_corrections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

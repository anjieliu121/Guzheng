#!/usr/bin/env python3
"""Analyze a MIDI file and upsert its metadata into a JSON index."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mido


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def note_name(n: int) -> str:
    """Return the scientific pitch name for a MIDI note number (e.g. 60 → 'C4')."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{(n // 12) - 1}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    return [v for v in values if not (v in seen or seen.add(v))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class MidiSummary:
    track_count: int
    track_names: list[str]
    tempo_bpm: list[float]
    time_signature: list[str]
    key_signature: list[str]
    duration_seconds: float
    note_lowest: str | None
    note_highest: str | None
    note_count: int
    bar_count: int | None
    has_pitch_bend: bool
    has_velocity: bool  # True when more than one distinct velocity is used
    instrument_program: list[int]


def analyze(path: Path) -> MidiSummary:
    mid = mido.MidiFile(str(path))

    track_names: list[str] = [
        (getattr(tr, "name", "") or "").strip() or f"Track {i}"
        for i, tr in enumerate(mid.tracks)
    ]

    tempos: list[float] = []
    time_sigs: list[str] = []
    key_sigs: list[str] = []
    notes: list[int] = []
    velocities: set[int] = set()
    has_pitch_bend = False
    instrument_program: list[int] = []
    seen_program: set[int] = set()

    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta:
                if msg.type == "pitchwheel":
                    has_pitch_bend = True
                elif msg.type == "note_on" and msg.velocity > 0:
                    notes.append(msg.note)
                    velocities.add(msg.velocity)
                elif msg.type == "program_change":
                    if msg.program not in seen_program:
                        seen_program.add(msg.program)
                        instrument_program.append(msg.program)
            else:
                if msg.type == "set_tempo":
                    tempos.append(round(mido.tempo2bpm(msg.tempo), 3))
                elif msg.type == "time_signature":
                    time_sigs.append(f"{msg.numerator}/{msg.denominator}")
                elif msg.type == "key_signature":
                    key_sigs.append(msg.key)

    bar_count: int | None = None
    ts = next(
        (msg for tr in mid.tracks for msg in tr
         if msg.is_meta and msg.type == "time_signature"),
        None,
    )
    if ts is not None and mid.ticks_per_beat:
        beats_per_bar = ts.numerator * (4.0 / ts.denominator)
        total_ticks = max(
            (sum(msg.time for msg in tr) for tr in mid.tracks),
            default=0,
        )
        total_beats = total_ticks / mid.ticks_per_beat
        bar_count = math.floor(total_beats / beats_per_bar) if beats_per_bar else None

    return MidiSummary(
        track_count=len(mid.tracks),
        track_names=track_names,
        tempo_bpm=[float(x) for x in _dedupe([str(t) for t in tempos])],
        time_signature=_dedupe(time_sigs),
        key_signature=_dedupe(key_sigs),
        duration_seconds=round(mid.length, 3),
        note_lowest=note_name(min(notes)) if notes else None,
        note_highest=note_name(max(notes)) if notes else None,
        note_count=len(notes),
        bar_count=bar_count,
        has_pitch_bend=has_pitch_bend,
        has_velocity=len(velocities) > 1,
        instrument_program=instrument_program if instrument_program else [0],
    )


def enforce_instrument_program(
    path: Path, target_program: int = 0
) -> list[int]:
    """
    Ensure every note channel has program_change(target_program).
    If missing or different, update MIDI in-place and return the resulting
    deduped program list.
    """
    mid = mido.MidiFile(str(path))
    changed = False

    for tr in mid.tracks:
        note_channels: set[int] = set()
        pc_channels: set[int] = set()
        for i, msg in enumerate(tr):
            if msg.type in ("note_on", "note_off"):
                note_channels.add(msg.channel)
            if msg.type == "program_change":
                pc_channels.add(msg.channel)
                if msg.program != target_program:
                    tr[i] = msg.copy(program=target_program)
                    changed = True

        missing = sorted(ch for ch in note_channels if ch not in pc_channels)
        if missing:
            prepend = [
                mido.Message(
                    "program_change", channel=ch, program=target_program, time=0
                )
                for ch in missing
            ]
            tr[:] = prepend + list(tr)
            changed = True

    if changed:
        mid.save(str(path))

    # Re-read resulting programs for robust reporting.
    final = mido.MidiFile(str(path))
    out: list[int] = []
    seen: set[int] = set()
    for tr in final.tracks:
        for msg in tr:
            if msg.type == "program_change" and msg.program not in seen:
                seen.add(msg.program)
                out.append(msg.program)
    return out if out else [target_program]


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(f"Index must be {{\"items\": [...]}}:  {path}")
    return data


def save_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upsert(index: dict[str, Any], item: dict[str, Any], *, overwrite: bool) -> bool:
    """Insert item; optionally overwrite if id exists.

    Returns True if the index was modified, False otherwise.
    """
    items: list[dict[str, Any]] = index.setdefault("items", [])
    for i, existing in enumerate(items):
        if isinstance(existing, dict) and existing.get("id") == item["id"]:
            if not overwrite:
                return False
            items[i] = item
            return True
    items.append(item)
    return True


def build_item(midi_path: Path, summary: MidiSummary, args: argparse.Namespace) -> dict[str, Any]:
    try:
        rel_path = midi_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_path = midi_path.as_posix()

    return {
        "id": midi_path.stem,
        "path": rel_path,
        "title": {
            "en": args.title_en,
            "zh": args.title_zh,
            "alt": args.title_alt,
        },
        "mode": args.mode,
        "has_velocity": summary.has_velocity,
        "has_pitch_bend": summary.has_pitch_bend,
        "instrument_program": summary.instrument_program,
        **{
            k: v
            for k, v in asdict(summary).items()
            if k not in {"has_velocity", "has_pitch_bend", "instrument_program"}
        },
        "source_sheet_url": args.source_sheet_url,
        "source": args.source,
        "notes": args.notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("midi_file", help="Path to a .mid file")
    p.add_argument("--index", default="metadata/MIDI.json",
                   help="Path to the JSON index (default: metadata/MIDI.json)")
    p.add_argument("--title-en", default="")
    p.add_argument("--title-zh", default="")
    p.add_argument("--title-alt", action="append", default=[],
                   metavar="ALT", help="Repeatable alternative title")
    p.add_argument("--mode", default="pentatonic")
    p.add_argument("--source-sheet-url", default="")
    p.add_argument("--source", default="daw_programmatic",
                   help='How it was created (default: "daw_programmatic")')
    p.add_argument("--notes", default="")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing item with the same id (default: false)",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)

    index_path = Path(args.index)
    # Requirement: if program is missing or not target, force it to 0 and record it.
    enforce_instrument_program(midi_path, target_program=0)
    summary = analyze(midi_path)
    item = build_item(midi_path, summary, args)

    index = load_index(index_path)
    modified = upsert(index, item, overwrite=bool(args.overwrite))
    if not modified:
        print(
            f"ID '{item['id']}' already exists in {index_path}. "
            "Use --overwrite to replace it."
        )
        return 1
    save_index(index_path, index)

    print(f"Upserted '{item['id']}' → {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
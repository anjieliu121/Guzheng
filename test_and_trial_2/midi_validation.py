#!/usr/bin/env python3
"""
Audit MIDI_transposed/ and write:
- metadata/midi_validation_report.md
- metadata/midi_issues.csv

This does NOT modify any MIDI files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mido


REPO = Path(__file__).resolve().parents[1]
MIDI_DIR_DEFAULT = REPO / "MIDI_transposed"
SCALES_JSON_DEFAULT = REPO / "metadata" / "guzheng_scales.json"
REPORT_MD_DEFAULT = REPO / "metadata" / "midi_validation_report.md"
ISSUES_CSV_DEFAULT = REPO / "metadata" / "midi_issues.csv"

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_spn(n: int) -> str:
    return f"{PC_NAMES[n % 12]}{(n // 12) - 1}"


def iter_midi_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.mid") if p.is_file()])


def piece_base(name: str) -> str:
    stem = Path(name).stem
    for suf in ("_A", "_C", "_D", "_F", "_G"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def spn_to_midi(s: str) -> int:
    s = s.replace("♭", "b").replace("♯", "#")
    m = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)$", s)
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


def load_scale_pitch_classes(path: Path) -> Dict[str, set[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, set[int]] = {}
    for s in data.get("scales", []):
        pcs: set[int] = set()
        for e in (s.get("entries", []) or []):
            midi = int(e.get("midi", -1))
            if 0 <= midi <= 127:
                pcs.add(midi % 12)
        if pcs:
            out[str(s.get("scale"))] = pcs
    return out


@dataclass
class Issue:
    filename: str
    issue_type: str
    severity: str
    description: str
    suggested_action: str


def ongrid_ratio(onsets: List[int], tpb: int, grid_div: int = 4, tol: int = 2) -> float:
    if not onsets or not tpb:
        return float("nan")
    grid = max(1, int(round(tpb / grid_div)))
    good = 0
    for t in onsets:
        r = t % grid
        d = min(r, grid - r)
        if d <= tol:
            good += 1
    return good / len(onsets)


def collect_note_ons(mid: mido.MidiFile) -> List[Tuple[int, int, int, int]]:
    """(abs_tick, pitch, velocity, channel) across all tracks."""
    out: List[Tuple[int, int, int, int]] = []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                out.append((t, int(msg.note), int(msg.velocity), int(msg.channel)))
    out.sort(key=lambda x: (x[0], x[1], x[3]))
    return out


def find_track_note_bearing(mid: mido.MidiFile) -> List[int]:
    idxs = []
    for i, tr in enumerate(mid.tracks):
        if any((m.type in ("note_on", "note_off")) for m in tr if not m.is_meta):
            idxs.append(i)
    return idxs


def program_changes(mid: mido.MidiFile) -> List[int]:
    progs: List[int] = []
    seen = set()
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "program_change":
                if msg.program not in seen:
                    seen.add(msg.program)
                    progs.append(int(msg.program))
    return sorted(progs)


def midi_header_ok(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"MThd"
    except Exception:
        return False


def note_sequence_fingerprint(mid: mido.MidiFile, tick_res: int = 10) -> str:
    # A stable hash for duplicates: quantize onsets/durations and store (dt,p,d,v)
    events = collect_note_ons(mid)
    if not events:
        return "EMPTY"
    # Pair durations by crude heuristic: assume note_off exists; if not, omit duration.
    # We only need a coarse duplicate indicator.
    pending: Dict[Tuple[int, int], Tuple[int, int]] = {}
    notes: List[Tuple[int, int, int, int]] = []  # (onset, pitch, dur, vel)
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in pending:
                    onset, vel = pending.pop(k)
                    dur = max(1, t - onset)
                    notes.append((onset, int(msg.note), dur, int(vel)))
    notes.sort(key=lambda x: (x[0], x[1]))
    prev = notes[0][0]
    seq: List[Tuple[int, int, int, int]] = []
    for onset, pitch, dur, vel in notes:
        dt = max(0, onset - prev)
        seq.append((int(round(dt / tick_res)), pitch, int(round(dur / tick_res)), vel // 4))
        prev = onset
    h = hashlib.sha1()
    h.update(repr(seq).encode("utf-8"))
    return h.hexdigest()


def estimate_training_sequences(total_notes: int, context_len: int = 2048, stride: int = 1024) -> int:
    # Approx 4 tokens/note + 2 tokens (BOS+KEY) overhead
    total_tokens = total_notes * 4
    if total_tokens <= context_len:
        return 1 if total_tokens > 0 else 0
    return max(1, int(math.ceil((total_tokens - context_len) / max(1, stride))) + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default=str(MIDI_DIR_DEFAULT))
    ap.add_argument("--scales_json", default=str(SCALES_JSON_DEFAULT))
    ap.add_argument("--report_md", default=str(REPORT_MD_DEFAULT))
    ap.add_argument("--issues_csv", default=str(ISSUES_CSV_DEFAULT))
    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    report_md = Path(args.report_md)
    issues_csv = Path(args.issues_csv)
    key_pc_sets = load_scale_pitch_classes(Path(args.scales_json))

    files = iter_midi_files(midi_dir)
    issues: List[Issue] = []

    # dataset aggregates
    durations: List[float] = []
    note_counts: List[int] = []
    tempos: List[float] = []
    pitch_mins: List[int] = []
    pitch_maxs: List[int] = []
    pcs_total = [0] * 12
    key_guess_counts = Counter()

    fingerprints: Dict[str, List[str]] = defaultdict(list)
    bases: Dict[str, List[str]] = defaultdict(list)

    per_file_rows = []

    for p in files:
        name = p.name
        bases[piece_base(name)].append(name)

        if not midi_header_ok(p):
            issues.append(
                Issue(name, "corrupt_file", "error", "Missing MThd header", "Re-export / fix file")
            )
            continue

        try:
            mid = mido.MidiFile(str(p))
        except Exception as e:
            issues.append(Issue(name, "corrupt_file", "error", f"Unreadable MIDI: {e}", "Re-export / fix file"))
            continue

        fmt = int(getattr(mid, "type", -1))
        tpb = int(mid.ticks_per_beat)
        duration = float(mid.length)

        note_ons = collect_note_ons(mid)
        notes = [n for _t, n, _v, _ch in note_ons]
        vels = [v for _t, _n, v, _ch in note_ons]
        onsets = [t for t, _n, _v, _ch in note_ons]

        pc_counts = [0] * 12
        for n in notes:
            pc_counts[n % 12] += 1
            pcs_total[n % 12] += 1

        pitch_min = min(notes) if notes else None
        pitch_max = max(notes) if notes else None

        # key guess by overlap
        key_guess = "unknown"
        if notes and key_pc_sets:
            present = {i for i, c in enumerate(pc_counts) if c > 0}
            best = None
            for k, pcs in key_pc_sets.items():
                score = len(present & pcs) / max(1, len(present | pcs))
                best = (score, k) if best is None or score > best[0] else best
            if best is not None:
                key_guess = best[1]
        key_guess_counts[key_guess] += 1

        # Track structure
        note_tracks = find_track_note_bearing(mid)
        if len(note_tracks) > 1:
            issues.append(
                Issue(
                    name,
                    "multi_track",
                    "warning",
                    f"{len(note_tracks)} note-bearing tracks: {note_tracks}",
                    "Review track merge / export settings",
                )
            )

        # Instruments
        progs = program_changes(mid)
        if progs:
            ok = set(progs).issubset({0, 107})
            if not ok:
                issues.append(
                    Issue(
                        name,
                        "wrong_instrument",
                        "warning",
                        f"Program changes present: {progs}",
                        "Set instrument to program 0 or 107 (if desired)",
                    )
                )

        # Pitch range
        if pitch_min is not None and pitch_max is not None:
            if pitch_min < 38 or pitch_max > 86:
                issues.append(
                    Issue(
                        name,
                        "out_of_range_pitch",
                        "warning",
                        f"Pitch range {pitch_min}({midi_to_spn(pitch_min)})..{pitch_max}({midi_to_spn(pitch_max)}) outside 38..86",
                        "Check octave / transcription; confirm if intentional",
                    )
                )

        # Non-pentatonic tones (info) + suspicious singletons far from any key pcs
        present_pcs = {i for i, c in enumerate(pc_counts) if c > 0}
        pent_pcs = key_pc_sets.get(key_guess, set())
        non_pent = sorted([pc for pc in present_pcs if pc not in pent_pcs]) if pent_pcs else []
        if non_pent:
            issues.append(
                Issue(
                    name,
                    "suspicious_pitch",
                    "info",
                    f"Non-pentatonic pitch classes: {[PC_NAMES[x] for x in non_pent]}",
                    "Document only (chromatic tones may be intentional)",
                )
            )

        # Timing (grid)
        grid_pct = float("nan")
        if onsets:
            grid_pct = ongrid_ratio(onsets, tpb, grid_div=4, tol=2)
            if not math.isnan(grid_pct) and grid_pct < 0.7:
                issues.append(
                    Issue(
                        name,
                        "off_grid_timing",
                        "warning",
                        f"On-grid onset ratio (16th,±2 ticks) is {grid_pct*100:.1f}%",
                        "Review quantization / export timing",
                    )
                )

        # Tempo presence
        has_tempo = any(msg.is_meta and msg.type == "set_tempo" for tr in mid.tracks for msg in tr)
        if not has_tempo:
            issues.append(
                Issue(
                    name,
                    "no_tempo",
                    "info",
                    "No set_tempo event; default tempo assumed by parser",
                    "Add explicit tempo meta event if needed",
                )
            )

        # Velocity
        if vels:
            if min(vels) < 10:
                issues.append(
                    Issue(
                        name,
                        "near_silent_note",
                        "warning",
                        f"Velocity range {min(vels)}..{max(vels)} includes <10",
                        "Review low-velocity artifacts",
                    )
                )
            if len(set(vels)) == 1:
                issues.append(
                    Issue(
                        name,
                        "flat_velocity",
                        "info",
                        f"All notes share velocity {vels[0]}",
                        "OK but may sound flat; consider velocity augmentation",
                    )
                )

        # Structural
        if len(notes) < 30:
            issues.append(
                Issue(name, "too_short", "warning", f"Only {len(notes)} notes", "Consider removing or augmenting")
            )
        if len(notes) > 2000:
            issues.append(
                Issue(name, "too_long", "warning", f"{len(notes)} notes (>2000)", "Check for concatenation")
            )

        # dataset aggregates
        durations.append(duration)
        note_counts.append(len(notes))
        if pitch_min is not None:
            pitch_mins.append(pitch_min)
        if pitch_max is not None:
            pitch_maxs.append(pitch_max)
        for t in (float(round(mido.tempo2bpm(msg.tempo), 3)) for tr in mid.tracks for msg in tr if msg.is_meta and msg.type == "set_tempo"):
            tempos.append(t)

        fp = note_sequence_fingerprint(mid)
        fingerprints[fp].append(name)

        per_file_rows.append(
            {
                "filename": name,
                "format": fmt,
                "tracks": len(mid.tracks),
                "ticks_per_beat": tpb,
                "duration_sec": round(duration, 3),
                "note_tracks": note_tracks,
                "note_count": len(notes),
                "pitch_min": pitch_min,
                "pitch_max": pitch_max,
                "tempo_bpm": sorted(set(float(round(mido.tempo2bpm(msg.tempo), 3)) for tr in mid.tracks for msg in tr if msg.is_meta and msg.type == "set_tempo")),
                "program_changes": progs,
                "key_guess": key_guess,
                "on_grid_ratio": None if math.isnan(grid_pct) else round(grid_pct, 4),
                "velocity_min": min(vels) if vels else None,
                "velocity_max": max(vels) if vels else None,
                "velocity_unique": len(set(vels)) if vels else 0,
            }
        )

    # duplicates
    for fp, names in fingerprints.items():
        if fp != "EMPTY" and len(names) > 1:
            for n in names:
                issues.append(
                    Issue(n, "duplicate", "warning", f"Duplicate note sequence hash {fp} shared with {names}", "Deduplicate if unintended")
                )

    # near-duplicates: same base piece with multiple keys
    for base, names in bases.items():
        if len(names) > 1:
            for n in names:
                issues.append(
                    Issue(
                        n,
                        "near_duplicate",
                        "info",
                        f"Same base piece '{base}' appears in multiple keys: {sorted(names)}",
                        "OK if intentional transpositions; count unique pieces separately",
                    )
                )

    # summary
    total_files = len(files)
    total_notes = int(sum(note_counts))
    total_minutes = float(sum(durations) / 60.0) if durations else 0.0
    seq_est = estimate_training_sequences(total_notes, context_len=2048, stride=1024)

    # tokens-to-params ratio (rough)
    total_tokens = total_notes * 4
    lora_trainable = 3_090_000  # ~1% of 309M as a ballpark
    ratio = total_tokens / max(1, lora_trainable)

    md: List[str] = []
    md.append("## MIDI validation report\n\n")
    md.append(f"- **MIDI dir**: `{midi_dir}`\n")
    md.append(f"- **Total files discovered**: **{total_files}**\n")
    md.append(f"- **Total notes**: **{total_notes}**\n")
    md.append(f"- **Total duration**: **{total_minutes:.2f}** minutes\n")
    md.append(f"- **Estimated training sequences** (ctx=2048, stride=1024, ~4 tokens/note): **{seq_est}**\n")
    md.append(f"- **Tokens / trainable-params (rough)**: **{ratio:.4f}** (assuming ~{lora_trainable:,} LoRA params)\n")
    md.append("\n### Dataset-wide pitch class distribution\n\n")
    total_pc = sum(pcs_total)
    md.append("| PC | Count | % |\n|---|---:|---:|\n")
    for i, c in enumerate(pcs_total):
        pct = (c / total_pc * 100.0) if total_pc else 0.0
        md.append(f"| {PC_NAMES[i]} | {c} | {pct:.2f}% |\n")

    md.append("\n### Key guesses (by pitch-class overlap)\n\n")
    md.append("| key | files |\n|---|---:|\n")
    for k, v in key_guess_counts.most_common():
        md.append(f"| {k} | {v} |\n")

    md.append("\n## Per-file summary\n\n")
    md.append("| file | fmt | tracks | tpb | dur(s) | notes | pitch(min-max) | tempos | programs | key |\n")
    md.append("|---|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for r in per_file_rows:
        mn = r["pitch_min"]
        mx = r["pitch_max"]
        pr = "-" if mn is None else f"{mn}..{mx}"
        md.append(
            f"| `{r['filename']}` | {r['format']} | {r['tracks']} | {r['ticks_per_beat']} | {r['duration_sec']} | {r['note_count']} | {pr} | {r['tempo_bpm']} | {r['program_changes']} | {r['key_guess']} |\n"
        )

    md.append("\n## Sufficiency verdict\n\n")
    if total_files >= 31 and total_notes >= 20_000:
        verdict = "sufficient"
    elif total_files >= 20 and total_notes >= 10_000:
        verdict = "marginal"
    else:
        verdict = "insufficient"
    md.append(f"- **Verdict**: **{verdict.upper()}**\n")
    md.append("- Notes: LoRA can work with small datasets, but more data (unique pieces) is better.\n")
    md.append("- Chromatic tones (4#, 7b, etc.) are treated as **info** when detected.\n")

    report_md.write_text("".join(md), encoding="utf-8")

    # issues csv
    issues_csv.parent.mkdir(parents=True, exist_ok=True)
    with issues_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "issue_type", "severity", "description", "suggested_action"])
        for it in issues:
            w.writerow([it.filename, it.issue_type, it.severity, it.description, it.suggested_action])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Check MIDI note duration bounds and overlapping same-pitch notes.

Reports for each MIDI file:
- notes with duration < 10ms or > 10240ms
- overlapping notes on the same channel and pitch

For each issue, report:
- note index in the file (1-based in the emitted note list)
- bar number
- beat number (float, 1-based within bar)
- pitch (MIDI + SPN)
- duration (seconds/ms)
- overlap entries also include start_on tick

Usage examples:
  python scripts/check_midi_note_quality.py MIDI/chun_miao.mid
  python scripts/check_midi_note_quality.py MIDI/*.mid
  python scripts/check_midi_note_quality.py MIDI --recursive
  python scripts/check_midi_note_quality.py MIDI/chun_miao.mid --apply

By default the script does not modify MIDI files (dry run). Pass --apply to write
corrections back to disk.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mido

MIN_DUR_SEC = 0.010
MAX_DUR_SEC = 10.240
GRID_TICK = 60
OFFGRID_TOL = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
SCALES_JSON = REPO_ROOT / "metadata" / "guzheng_scales.json"
DEFAULT_REPORT_MD = REPO_ROOT / "notes" / "data_processing.md"


@dataclass
class NoteEvent:
    idx: int
    ch: int
    pitch: int
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    velocity: int

    @property
    def dur_sec(self) -> float:
        return self.end_sec - self.start_sec


def midi_to_spn(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{(n // 12) - 1}"


def dist_to_grid(tick: int, grid_tick: int = GRID_TICK) -> int:
    r = tick % grid_tick
    return min(r, grid_tick - r)


def nearest_grid_tick(tick: int, grid_tick: int = GRID_TICK) -> int:
    r = tick % grid_tick
    if r <= grid_tick - r:
        return tick - r
    return tick + (grid_tick - r)


def _jianpu_to_target(jianpu: str) -> str | None:
    """
    Convert chromatic pressed jianpu to the pentatonic target (octave marks kept):
    4#→5, 5#→6, 3b→3, 7b→7 with same suffix (e.g. 7b,,→7,,, 7b,→7,).
    """
    mapping = {"4#": "5", "5#": "6", "3b": "3", "7b": "7"}
    for src, dst in mapping.items():
        if jianpu.startswith(src):
            return dst + jianpu[len(src) :]
    return None


def load_scale_models(scales_json: Path = SCALES_JSON) -> dict[str, dict]:
    """
    Returns model per scale:
    - valid_midi: all entries + pressed_strings MIDI notes (never chromatic-remapped)
    - entry_pitch_classes: pitch classes of entries (core pentatonic)
    - shift_up_map: source midi -> corrected midi for (4#,5#,3b,7b) families
    """
    data = json.loads(scales_json.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for s in data.get("scales", []):
        scale = str(s.get("scale"))
        entries = list(s.get("entries", []) or [])
        pressed = list(s.get("pressed_strings", []) or [])
        all_rows = entries + pressed
        valid_midi = {
            int(e["midi"]) for e in all_rows if isinstance(e, dict) and isinstance(e.get("midi"), int)
        }
        entry_pcs = {
            int(e["midi"]) % 12
            for e in entries
            if isinstance(e, dict) and isinstance(e.get("midi"), int)
        }

        by_jianpu: dict[str, int] = {}
        for e in all_rows:
            if isinstance(e, dict) and isinstance(e.get("jianpu"), str) and isinstance(e.get("midi"), int):
                by_jianpu[e["jianpu"]] = int(e["midi"])

        shift_up_map: dict[int, int] = {}
        for jp, src_midi in by_jianpu.items():
            tgt_jp = _jianpu_to_target(jp)
            if not tgt_jp:
                continue
            tgt_midi = by_jianpu.get(tgt_jp, src_midi + 1)
            # Guard: only accept if target is still a valid note in this scale model.
            if tgt_midi in valid_midi:
                shift_up_map[src_midi] = tgt_midi

        # Derive per-scale fallback source pitch classes from target degrees.
        # The four chromatic families (4#→5, 5#→6, 3b→3, 7b→7) each have a
        # source note one semitone below the target degree.
        chromatic_target_degrees = {"3", "5", "6", "7"}
        degree_pcs: dict[str, int] = {}
        for jp, midi_val in by_jianpu.items():
            base = jp.rstrip(",'" )
            if base in chromatic_target_degrees:
                degree_pcs.setdefault(base, midi_val % 12)
        fallback_source_pcs: set[int] = {
            (degree_pcs[d] - 1) % 12
            for d in chromatic_target_degrees
            if d in degree_pcs
        }

        out[scale] = {
            "valid_midi": valid_midi,
            "entry_pitch_classes": entry_pcs,
            "shift_up_map": shift_up_map,
            "fallback_source_pcs": fallback_source_pcs,
        }
    return out


def read_scale_from_midi(mid: mido.MidiFile, scale_models: dict[str, dict]) -> str:
    """Read key_signature meta event from MIDI and map to a known scale model."""
    for tr in mid.tracks:
        for msg in tr:
            if msg.is_meta and msg.type == "key_signature":
                key = msg.key.replace(" major", "").replace(" minor", "")
                if key in scale_models:
                    return key
    return "D" if "D" in scale_models else next(iter(scale_models.keys()))


# channel, start_tick, src_pitch, dst_pitch (one row per corrected note_on)
ScaleChromaticCorrection = tuple[int, int, int, int]


def apply_pitch_map(
    mid: mido.MidiFile,
    pitch_map: dict[int, int],
    *,
    never_remap_from: set[int] | None = None,
) -> tuple[int, list[ScaleChromaticCorrection]]:
    """
    Rewrite note_on / paired note_off pitches. Uses LIFO per (channel, source pitch)
    so overlapping same-pitch notes stay consistent.
    """
    corrected = 0
    details: list[ScaleChromaticCorrection] = []
    for tr in mid.tracks:
        abs_tick = 0
        active_replacements: dict[tuple[int, int], list[int]] = {}
        for i, msg in enumerate(tr):
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                src = int(msg.note)
                dst = pitch_map.get(src)
                if never_remap_from is not None and src in never_remap_from:
                    dst = None
                if dst is not None and dst != src:
                    ch = int(msg.channel)
                    tr[i] = msg.copy(note=dst)
                    active_replacements.setdefault((ch, src), []).append(dst)
                    corrected += 1
                    details.append((ch, abs_tick, src, dst))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                ch = int(msg.channel)
                src = int(msg.note)
                key = (ch, src)
                if active_replacements.get(key):
                    dst = active_replacements[key].pop()
                    tr[i] = msg.copy(note=dst)
    return corrected, details


def autocorrect_non_scale_chromatic_notes(
    mid: mido.MidiFile, scale_model: dict
) -> tuple[int, list[ScaleChromaticCorrection]]:
    """
    For detected scale, shift supported chromatic variants (4#,5#,3b,7b families)
    to mapped pentatonic targets (e.g. 7b,→7, per jianpu map or pitch-class fallback).
    """
    valid_midi: set[int] = scale_model["valid_midi"]
    shift_up_map: dict[int, int] = scale_model["shift_up_map"]
    fallback_source_pcs: set[int] = scale_model.get("fallback_source_pcs", set())
    corrected = 0
    details: list[ScaleChromaticCorrection] = []

    for tr in mid.tracks:
        abs_tick = 0
        active_replacements: dict[tuple[int, int], list[int]] = {}
        for i, msg in enumerate(tr):
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                src = int(msg.note)
                dst = None
                # Open strings and pressed_strings from metadata are allowed as-is;
                # do not treat them as chromatic mistakes (e.g. A-scale G#5 "7" is valid).
                if src not in valid_midi:
                    dst = shift_up_map.get(src)
                    if dst is None:
                        # Fallback: normalize known chromatic classes by +1 semitone
                        # when the resulting note is valid in the detected scale.
                        cand = src + 1
                        if (src % 12) in fallback_source_pcs and cand in valid_midi:
                            dst = cand
                if dst is not None and dst != src:
                    ch = int(msg.channel)
                    tr[i] = msg.copy(note=dst)
                    active_replacements.setdefault((ch, src), []).append(dst)
                    corrected += 1
                    details.append((ch, abs_tick, src, dst))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                ch = int(msg.channel)
                src = int(msg.note)
                key = (ch, src)
                if active_replacements.get(key):
                    dst = active_replacements[key].pop()
                    tr[i] = msg.copy(note=dst)

    return corrected, details


def _retime_track_with_abs_updates(
    track: mido.MidiTrack, updates: dict[int, int]
) -> mido.MidiTrack:
    """Apply absolute tick updates to selected events, then rebuild deltas."""
    out: list[tuple[int, mido.Message]] = []
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        new_abs = updates.get(id(msg), abs_tick)
        out.append((new_abs, msg.copy()))

    # Stable sort keeps original order for events that land on same tick.
    out.sort(key=lambda x: x[0])

    new_track = mido.MidiTrack()
    prev = 0
    for abs_new, msg in out:
        dt = abs_new - prev
        if dt < 0:
            dt = 0
        msg.time = dt
        new_track.append(msg)
        prev = abs_new
    return new_track


def autocorrect_slightly_offgrid_notes(
    mid: mido.MidiFile, grid_tick: int = GRID_TICK, tol: int = OFFGRID_TOL
) -> int:
    """Snap note start/end to nearest grid if both endpoints are within tolerance."""
    corrected = 0

    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        active: dict[tuple[int, int], list[tuple[int, mido.Message]]] = {}
        updates: dict[int, int] = {}

        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active.setdefault(key, []).append((abs_tick, msg))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if not active.get(key):
                    continue
                start_tick, start_msg = active[key].pop()
                end_tick = abs_tick

                start_close = dist_to_grid(start_tick, grid_tick) <= tol
                end_close = dist_to_grid(end_tick, grid_tick) <= tol
                off_grid = (start_tick % grid_tick != 0) or (end_tick % grid_tick != 0)
                if not (off_grid and start_close and end_close):
                    continue

                new_start = nearest_grid_tick(start_tick, grid_tick)
                new_end = nearest_grid_tick(end_tick, grid_tick)
                if new_end <= new_start:
                    new_end = new_start + 1

                if new_start != start_tick:
                    updates[id(start_msg)] = new_start
                if new_end != end_tick:
                    updates[id(msg)] = new_end

                if new_start != start_tick or new_end != end_tick:
                    corrected += 1

        if updates:
            mid.tracks[ti] = _retime_track_with_abs_updates(track, updates)

    return corrected


def build_bars(mid: mido.MidiFile) -> list[tuple[int, int, int, int, int]]:
    """Return list of (bar_no, start_tick, end_tick, numerator, denominator)."""
    tpb = mid.ticks_per_beat

    ts: list[tuple[int, int, int]] = []
    max_tick = 0
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.is_meta and msg.type == "time_signature":
                ts.append((t, msg.numerator, msg.denominator))
        max_tick = max(max_tick, t)

    if not ts:
        ts = [(0, 4, 4)]

    # Keep the last time signature if multiple appear at same tick.
    by_tick: dict[int, tuple[int, int]] = {}
    for t, n, d in ts:
        by_tick[t] = (n, d)
    ts = sorted((t, n, d) for t, (n, d) in by_tick.items())

    bars: list[tuple[int, int, int, int, int]] = []
    bar_no = 1
    for i, (tick, n, d) in enumerate(ts):
        seg_end = ts[i + 1][0] if i + 1 < len(ts) else max_tick + 1
        bar_len = int((n * tpb * 4) // d)
        cur = tick
        while cur < seg_end:
            nxt = min(cur + bar_len, seg_end)
            bars.append((bar_no, cur, nxt, n, d))
            bar_no += 1
            cur = nxt
    return bars


def bar_beat_of_tick(tick: int, bars: list[tuple[int, int, int, int, int]], tpb: int) -> tuple[int, float]:
    for b, s, e, n, d in bars:
        if s <= tick < e:
            beat_len = tpb * 4 / d
            beat = (tick - s) / beat_len + 1.0
            return b, beat
    if not bars:
        return 1, 1.0
    b, s, _, _, d = bars[-1]
    beat_len = tpb * 4 / d
    return b, (tick - s) / beat_len + 1.0


def collect_notes(mid: mido.MidiFile) -> list[NoteEvent]:
    """Collect note intervals from all tracks with tempo-aware seconds."""
    tpb = mid.ticks_per_beat

    events: list[tuple[int, int, mido.Message]] = []
    for ti, tr in enumerate(mid.tracks):
        t = 0
        for msg in tr:
            t += msg.time
            events.append((t, ti, msg))
    events.sort(key=lambda x: (x[0], x[1]))

    tempo = 500000  # default 120 BPM
    prev_tick = 0
    sec = 0.0

    active: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    notes: list[NoteEvent] = []

    for tick, _ti, msg in events:
        dt = tick - prev_tick
        if dt:
            sec += mido.tick2second(dt, tpb, tempo)
            prev_tick = tick

        if msg.is_meta and msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            k = (msg.channel, msg.note)
            active.setdefault(k, []).append((sec, tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            k = (msg.channel, msg.note)
            if active.get(k):
                s_sec, s_tick, vel = active[k].pop()
                notes.append(
                    NoteEvent(
                        idx=0,
                        ch=msg.channel,
                        pitch=msg.note,
                        start_tick=s_tick,
                        end_tick=tick,
                        start_sec=s_sec,
                        end_sec=sec,
                        velocity=vel,
                    )
                )

    notes.sort(key=lambda n: (n.start_tick, n.pitch, n.ch, n.end_tick))
    for i, n in enumerate(notes, start=1):
        n.idx = i
    return notes


def find_duration_issues(notes: Iterable[NoteEvent]) -> list[NoteEvent]:
    out: list[NoteEvent] = []
    for n in notes:
        d = n.dur_sec
        if d < MIN_DUR_SEC or d > MAX_DUR_SEC:
            out.append(n)
    return out


def find_overlaps(notes: list[NoteEvent]) -> list[tuple[NoteEvent, NoteEvent]]:
    """Find overlaps for same (channel, pitch): prev.start < cur.start < prev.end."""
    by_key: dict[tuple[int, int], list[NoteEvent]] = {}
    for n in notes:
        by_key.setdefault((n.ch, n.pitch), []).append(n)

    overlaps: list[tuple[NoteEvent, NoteEvent]] = []
    for seq in by_key.values():
        seq.sort(key=lambda x: (x.start_tick, x.end_tick))
        prev: NoteEvent | None = None
        for cur in seq:
            if prev is not None and cur.start_tick < prev.end_tick:
                overlaps.append((prev, cur))
                if cur.end_tick > prev.end_tick:
                    prev = cur
            else:
                prev = cur
    return overlaps


def iter_midi_paths(inputs: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_file() and p.suffix.lower() in {".mid", ".midi"}:
            out.append(p)
        elif p.is_dir():
            globber = p.rglob if recursive else p.glob
            out.extend(x for x in globber("*.mid") if x.is_file())
            out.extend(x for x in globber("*.midi") if x.is_file())
    # unique, stable order
    uniq = []
    seen = set()
    for p in sorted(out):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _upsert_file_section(markdown_text: str, file_heading: str, block: str) -> str:
    """
    Keep one H1 per file (`# filename`). If exists, append block inside that section;
    otherwise create a new section at the end.
    """
    heading = f"# {file_heading}"
    lines = markdown_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break

    if start is None:
        if markdown_text and not markdown_text.endswith("\n"):
            markdown_text += "\n"
        if markdown_text.strip():
            markdown_text += "\n"
        markdown_text += f"{heading}\n\n{block.rstrip()}\n"
        return markdown_text

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("# "):
            end = j
            break

    section = "\n".join(lines[start:end]).rstrip()
    section += "\n\n" + block.rstrip() + "\n"
    new_lines = lines[:start] + section.splitlines() + lines[end:]
    return "\n".join(new_lines) + ("\n" if markdown_text.endswith("\n") or not markdown_text else "")


def _build_run_block(
    scale: str,
    total_notes: int,
    scale_corrected: int,
    scale_correction_details: list[ScaleChromaticCorrection],
    corrected: int,
    dur_issues: list[NoteEvent],
    overlaps: list[tuple[NoteEvent, NoteEvent]],
    bars: list[tuple[int, int, int, int, int]],
    tpb: int,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out: list[str] = []
    out.append(f"## Run {ts}")
    out.append("")
    out.append("### Scale (from MIDI key signature)")
    out.append(f"- `{scale}`")
    out.append("")
    out.append("### Corrections")
    out.append(f"- Scale-corrected chromatic notes: **{scale_corrected}**")
    if scale_correction_details:
        out.append("")
        out.append("| ch | bar | beat | start tick | from | to |")
        out.append("|---:|---:|---:|---:|---|---|")
        for ch, start_tick, src, dst in scale_correction_details:
            bar, beat = bar_beat_of_tick(start_tick, bars, tpb)
            out.append(
                f"| {ch} | {bar} | {beat:.3f} | {start_tick} | "
                f"`{src} ({midi_to_spn(src)})` | `{dst} ({midi_to_spn(dst)})` |"
            )
    out.append("")
    out.append(
        f"- Off-grid corrected notes (±{OFFGRID_TOL} ticks on {GRID_TICK}-tick grid): **{corrected}**"
    )
    out.append("")
    out.append("### Basic counts")
    out.append(f"- Total notes: **{total_notes}**")
    out.append("")

    out.append("### Duration out-of-range")
    if not dur_issues:
        out.append("- None")
    else:
        out.append(
            "| note# | bar | beat | pitch | duration (s/ms) |"
        )
        out.append("|---:|---:|---:|---|---|")
        for n in dur_issues:
            bar, beat = bar_beat_of_tick(n.start_tick, bars, tpb)
            out.append(
                f"| {n.idx} | {bar} | {beat:.3f} | `{n.pitch} ({midi_to_spn(n.pitch)})` | `{n.dur_sec:.6f}s / {n.dur_sec*1000:.3f}ms` |"
            )
    out.append("")

    out.append("### Overlapping notes (same channel+pitch)")
    if not overlaps:
        out.append("- None")
    else:
        out.append(
            "| cur note# | bar | beat | pitch | start_on | duration (s/ms) | overlaps note# |"
        )
        out.append("|---:|---:|---:|---|---:|---|---:|")
        for prev, cur in overlaps:
            bar, beat = bar_beat_of_tick(cur.start_tick, bars, tpb)
            out.append(
                f"| {cur.idx} | {bar} | {beat:.3f} | `{cur.pitch} ({midi_to_spn(cur.pitch)})` | {cur.start_tick} | `{cur.dur_sec:.6f}s / {cur.dur_sec*1000:.3f}ms` | {prev.idx} |"
            )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="MIDI files or directories")
    ap.add_argument("--recursive", action="store_true", help="Recurse into directories")
    ap.add_argument(
        "--no-autocorrect-scale",
        action="store_true",
        help="Disable scale-based chromatic note correction (4#/5#/3b/7b -> pentatonic)",
    )
    ap.add_argument(
        "--no-autocorrect-offgrid",
        action="store_true",
        help="Disable auto-correction for notes within ±2 ticks of 60-tick grid",
    )
    ap.add_argument(
        "--report-md",
        default=str(DEFAULT_REPORT_MD),
        help="Append markdown output to this file (default: notes/data_processing.md)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write autocorrections to the MIDI files (default: dry run, no file changes)",
    )
    args = ap.parse_args()

    paths = iter_midi_paths(args.inputs, recursive=args.recursive)
    if not paths:
        print("No MIDI files found.")
        return 1

    scale_models = load_scale_models(SCALES_JSON)
    any_issue = False
    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    for path in paths:
        mid = mido.MidiFile(str(path))
        detected_scale = read_scale_from_midi(mid, scale_models)
        work = mid if args.apply else copy.deepcopy(mid)

        scale_corrected = 0
        scale_correction_details: list[ScaleChromaticCorrection] = []
        if not args.no_autocorrect_scale and detected_scale in scale_models:
            scale_corrected, scale_correction_details = autocorrect_non_scale_chromatic_notes(
                work, scale_models[detected_scale]
            )
        corrected = 0
        if not args.no_autocorrect_offgrid:
            corrected = autocorrect_slightly_offgrid_notes(
                work, grid_tick=GRID_TICK, tol=OFFGRID_TOL
            )
        if args.apply and (scale_corrected or corrected):
            work.save(str(path))

        bars = build_bars(work)
        notes = collect_notes(work)
        dur_issues = find_duration_issues(notes)
        overlaps = find_overlaps(notes)

        print(f"\n=== {path} ===")
        if args.apply:
            print("Mode: apply (MIDI files may be saved when corrections run)")
        else:
            print("Mode: dry run (use --apply to write MIDI changes; MIDI not modified)")
        print(f"Scale (from MIDI key signature): {detected_scale}")
        print(f"Total notes: {len(notes)}")
        print(f"Scale-corrected chromatic notes: {scale_corrected}")
        if scale_correction_details:
            print("  Scale chromatic pitch changes (per note_on):")
            for ch, start_tick, src, dst in scale_correction_details:
                bar, beat = bar_beat_of_tick(start_tick, bars, work.ticks_per_beat)
                print(
                    f"    ch={ch} bar={bar} beat={beat:.3f} tick={start_tick} "
                    f"{src}({midi_to_spn(src)}) -> {dst}({midi_to_spn(dst)})"
                )
        print(f"Auto-corrected off-grid notes (±{OFFGRID_TOL} ticks on {GRID_TICK}-tick grid): {corrected}")

        if dur_issues:
            any_issue = True
            print(f"Duration out-of-range notes (<10ms or >10240ms): {len(dur_issues)}")
            for n in dur_issues:
                bar, beat = bar_beat_of_tick(n.start_tick, bars, mid.ticks_per_beat)
                print(
                    f"  note#{n.idx} bar={bar} beat={beat:.3f} "
                    f"pitch={n.pitch}({midi_to_spn(n.pitch)}) "
                    f"duration={n.dur_sec:.6f}s ({n.dur_sec*1000:.3f}ms)"
                )
        else:
            print("Duration out-of-range notes: 0")

        if overlaps:
            any_issue = True
            print(f"Overlapping notes (same channel+pitch): {len(overlaps)}")
            for prev, cur in overlaps:
                bar, beat = bar_beat_of_tick(cur.start_tick, bars, mid.ticks_per_beat)
                print(
                    f"  note#{cur.idx} bar={bar} beat={beat:.3f} "
                    f"pitch={cur.pitch}({midi_to_spn(cur.pitch)}) "
                    f"start_on={cur.start_tick} duration={cur.dur_sec:.6f}s ({cur.dur_sec*1000:.3f}ms) "
                    f"overlaps note#{prev.idx} [{prev.start_tick},{prev.end_tick})"
                )
        else:
            print("Overlapping notes: 0")

        run_block = _build_run_block(
            scale=detected_scale,
            total_notes=len(notes),
            scale_corrected=scale_corrected,
            scale_correction_details=scale_correction_details,
            corrected=corrected,
            dur_issues=dur_issues,
            overlaps=overlaps,
            bars=bars,
            tpb=work.ticks_per_beat,
        )
        report_text = _upsert_file_section(report_text, path.name, run_block)

    report_path.write_text(report_text, encoding="utf-8")

    return 2 if any_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())

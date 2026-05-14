#!/usr/bin/env python3
"""
Generate dataset-level visualizations for MIDI_transposed/ and write outputs under metadata/.

Outputs:
- metadata/analysis_output/*.png
- metadata/dataset_analysis.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mido
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
MIDI_DIR_DEFAULT = REPO / "MIDI_transposed"
SCALES_JSON_DEFAULT = REPO / "metadata" / "guzheng_scales.json"
OUT_DIR_DEFAULT = REPO / "metadata" / "analysis_output"
REPORT_MD_DEFAULT = REPO / "metadata" / "dataset_analysis.md"

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_spn(n: int) -> str:
    return f"{PC_NAMES[n % 12]}{(n // 12) - 1}"


def iter_midi_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.mid") if p.is_file()])


def load_scales_pitch_class_sets(path: Path) -> Dict[str, set[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, set[int]] = {}
    for s in data.get("scales", []):
        pcs: set[int] = set()
        for e in s.get("entries", []) or []:
            midi = int(e.get("midi", -1))
            if 0 <= midi <= 127:
                pcs.add(midi % 12)
        if pcs:
            out[str(s.get("scale"))] = pcs
    return out


@dataclass
class MidiStats:
    filename: str
    duration_sec: float
    note_count: int
    pitch_min: Optional[int]
    pitch_max: Optional[int]
    tempos_bpm: List[float]
    pitch_class_counts: List[int]  # len=12
    velocities: List[int]
    onset_ticks: List[int]
    key_guess: str


def analyze_midi(path: Path, key_pc_sets: Dict[str, set[int]]) -> MidiStats:
    mid = mido.MidiFile(str(path))
    duration = float(mid.length)

    tempos_bpm: List[float] = []
    notes: List[int] = []
    velocities: List[int] = []
    onset_ticks: List[int] = []

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.is_meta and msg.type == "set_tempo":
                tempos_bpm.append(float(round(mido.tempo2bpm(msg.tempo), 3)))
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(int(msg.note))
                velocities.append(int(msg.velocity))
                onset_ticks.append(int(abs_t))

    pc_counts = [0] * 12
    for n in notes:
        pc_counts[n % 12] += 1

    pitch_min = min(notes) if notes else None
    pitch_max = max(notes) if notes else None

    key_guess = "unknown"
    if notes and key_pc_sets:
        present = {i for i, c in enumerate(pc_counts) if c > 0}
        best = None
        for k, pcs in key_pc_sets.items():
            score = len(present & pcs) / max(1, len(present | pcs))
            best = (score, k) if best is None or score > best[0] else best
        if best is not None:
            key_guess = best[1]

    return MidiStats(
        filename=path.name,
        duration_sec=duration,
        note_count=len(notes),
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        tempos_bpm=sorted(set(tempos_bpm)),
        pitch_class_counts=pc_counts,
        velocities=velocities,
        onset_ticks=onset_ticks,
        key_guess=key_guess,
    )


def ongrid_ratio(onsets: List[int], tpb: int, grid_div: int = 4, tol: int = 2) -> float:
    """grid_div=4 means 16th notes (tpb/4)."""
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


def save_hist(values: List[float], title: str, xlabel: str, out_path: Path, bins: int = 20):
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    if values:
        mean = float(np.mean(values))
        med = float(np.median(values))
        plt.axvline(mean, color="#F58518", linestyle="--", label=f"mean={mean:.2f}")
        plt.axvline(med, color="#54A24B", linestyle="--", label=f"median={med:.2f}")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default=str(MIDI_DIR_DEFAULT))
    ap.add_argument("--scales_json", default=str(SCALES_JSON_DEFAULT))
    ap.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--report_md", default=str(REPORT_MD_DEFAULT))
    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.out_dir)
    report_md = Path(args.report_md)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_pc_sets = load_scales_pitch_class_sets(Path(args.scales_json))
    mids = iter_midi_files(midi_dir)

    stats: List[MidiStats] = []
    for p in mids:
        stats.append(analyze_midi(p, key_pc_sets))

    # 1 note_count_distribution
    note_counts = [s.note_count for s in stats]
    save_hist(
        [float(x) for x in note_counts],
        "Note count distribution per file",
        "notes per file",
        out_dir / "note_count_distribution.png",
        bins=20,
    )

    # 2 duration_distribution
    durations = [s.duration_sec for s in stats]
    save_hist(
        durations,
        "Duration distribution per file",
        "seconds",
        out_dir / "duration_distribution.png",
        bins=20,
    )

    # 3 pitch_class_heatmap
    mat = np.array([s.pitch_class_counts for s in stats], dtype=float)
    if mat.size:
        mat_norm = mat / np.maximum(1.0, mat.sum(axis=1, keepdims=True))
        plt.figure(figsize=(12, max(4, 0.25 * len(stats))))
        plt.imshow(mat_norm, aspect="auto", cmap="viridis")
        plt.colorbar(label="pitch class frequency (per file)")
        plt.xticks(range(12), PC_NAMES, rotation=0)
        plt.yticks(range(len(stats)), [s.filename for s in stats], fontsize=7)
        plt.title("Pitch class heatmap (rows=files, cols=pitch classes)")
        plt.tight_layout()
        plt.savefig(out_dir / "pitch_class_heatmap.png")
        plt.close()

    # 4 pitch_range_per_file
    plt.figure(figsize=(12, max(4, 0.25 * len(stats))))
    y = np.arange(len(stats))
    mins = [s.pitch_min if s.pitch_min is not None else np.nan for s in stats]
    maxs = [s.pitch_max if s.pitch_max is not None else np.nan for s in stats]
    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        if math.isnan(mn) or math.isnan(mx):
            continue
        plt.plot([mn, mx], [i, i], color="#4C78A8", linewidth=2)
    plt.axvspan(38, 86, color="#54A24B", alpha=0.15, label="nominal guzheng range D2–D6")
    plt.yticks(y, [s.filename for s in stats], fontsize=7)
    plt.xlabel("MIDI note number")
    plt.title("Pitch range per file")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "pitch_range_per_file.png")
    plt.close()

    # 5 tempo_distribution
    tempos = [t for s in stats for t in s.tempos_bpm] or []
    save_hist(
        tempos,
        "Tempo distribution (set_tempo events)",
        "BPM",
        out_dir / "tempo_distribution.png",
        bins=20,
    )

    # 6 velocity_distribution (box plot per file)
    vel_lists = [s.velocities for s in stats if s.velocities]
    labels = [s.filename for s in stats if s.velocities]
    if vel_lists:
        plt.figure(figsize=(12, max(4, 0.25 * len(labels))))
        plt.boxplot(vel_lists, vert=False, labels=labels, showfliers=False)
        plt.xlabel("velocity")
        plt.title("Velocity distribution per file")
        plt.tight_layout()
        plt.savefig(out_dir / "velocity_distribution.png")
        plt.close()

    # 7 interval_distribution
    intervals: List[int] = []
    for s in stats:
        # crude: approximate melodic order by pitch of note-ons only isn't correct without pairing,
        # but it's good enough for dataset-level sanity.
        pcs = []
        # skip if no notes
        # reuse pitch_class_counts doesn't preserve order, so interval distribution omitted here
        _ = pcs
    # we'll instead compute from each file quickly via a second pass for ordered note-ons
    for p in mids:
        mid = mido.MidiFile(str(p))
        ordered: List[int] = []
        for tr in mid.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    ordered.append((t, int(msg.note)))
        ordered.sort(key=lambda x: (x[0], x[1]))
        pitches = [n for _t, n in ordered]
        for a, b in zip(pitches, pitches[1:]):
            intervals.append(b - a)
    save_hist(
        [float(x) for x in intervals],
        "Melodic interval distribution (note_on pitch diffs)",
        "semitones (next - prev)",
        out_dir / "interval_distribution.png",
        bins=41,
    )

    # 8 onset_grid_alignment
    ongrid: List[float] = []
    for p, s in zip(mids, stats):
        try:
            mid = mido.MidiFile(str(p))
            r = ongrid_ratio(s.onset_ticks, mid.ticks_per_beat, grid_div=4, tol=2)
        except Exception:
            r = float("nan")
        ongrid.append(r)
    plt.figure(figsize=(12, max(4, 0.25 * len(stats))))
    plt.barh(np.arange(len(stats)), [0 if math.isnan(x) else x * 100 for x in ongrid], color="#4C78A8")
    plt.xlim(0, 100)
    plt.yticks(np.arange(len(stats)), [s.filename for s in stats], fontsize=7)
    plt.xlabel("% on-grid onsets (16th-note grid, ±2 ticks)")
    plt.title("Onset grid alignment per file")
    plt.tight_layout()
    plt.savefig(out_dir / "onset_grid_alignment.png")
    plt.close()

    # 9 dataset_sufficiency infographic (simple bar)
    total_files = len(stats)
    total_notes = int(sum(s.note_count for s in stats))
    total_minutes = float(sum(s.duration_sec for s in stats) / 60.0)
    est_tokens = total_notes * 4
    plt.figure(figsize=(10, 4))
    labels2 = ["files", "notes", "minutes", "est_tokens"]
    vals2 = [total_files, total_notes, total_minutes, est_tokens]
    plt.bar(labels2, vals2, color=["#4C78A8"] * 4)
    plt.title("Dataset sufficiency summary")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset_sufficiency.png")
    plt.close()

    # 10 key_distribution
    key_counts = Counter(s.key_guess for s in stats)
    plt.figure(figsize=(8, 4))
    ks = list(key_counts.keys())
    vs = [key_counts[k] for k in ks]
    plt.bar(ks, vs, color="#4C78A8")
    plt.title("Detected key distribution (pitch-class overlap)")
    plt.ylabel("files")
    plt.tight_layout()
    plt.savefig(out_dir / "key_distribution.png")
    plt.close()

    # Write dataset_analysis.md
    rel_out = os.path.relpath(out_dir, report_md.parent)
    md = []
    md.append("## Dataset overview\n")
    md.append(f"- **MIDI dir**: `{midi_dir}`\n")
    md.append(f"- **Total files**: **{total_files}**\n")
    md.append(f"- **Total notes**: **{total_notes}**\n")
    md.append(f"- **Total duration**: **{total_minutes:.2f}** minutes\n")
    md.append(f"- **Estimated tokens** (4 tokens/note): **{est_tokens}**\n")
    md.append("\n## Visuals\n")

    def img(name: str, desc: str) -> str:
        return f"![{desc}]({rel_out}/{name})\n"

    md.append("\n### Note counts\n")
    md.append(img("note_count_distribution.png", "Histogram of note counts per file"))
    md.append("\n### Durations\n")
    md.append(img("duration_distribution.png", "Histogram of durations per file"))
    md.append("\n### Pitch classes\n")
    md.append(img("pitch_class_heatmap.png", "Pitch class heatmap"))
    md.append("\n### Pitch ranges\n")
    md.append(img("pitch_range_per_file.png", "Pitch min/max per file"))
    md.append("\n### Tempos\n")
    md.append(img("tempo_distribution.png", "Tempo distribution"))
    md.append("\n### Velocities\n")
    md.append(img("velocity_distribution.png", "Velocity distribution per file"))
    md.append("\n### Intervals\n")
    md.append(img("interval_distribution.png", "Interval distribution"))
    md.append("\n### Onset grid alignment\n")
    md.append(img("onset_grid_alignment.png", "Onset grid alignment per file"))
    md.append("\n### Sufficiency summary\n")
    md.append(img("dataset_sufficiency.png", "Dataset sufficiency summary"))
    md.append("\n### Key distribution\n")
    md.append(img("key_distribution.png", "Key distribution"))

    md.append("\n## Data readiness\n")
    md.append("- This report is an **audit only** and does not modify any MIDI files.\n")
    md.append("- Use `metadata/midi_validation_report.md` + `metadata/midi_issues.csv` for per-file issues.\n")

    report_md.write_text("".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


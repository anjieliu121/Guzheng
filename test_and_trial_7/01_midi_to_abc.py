"""MIDI -> single-voice ABC -> NotaGen interleaved + key-augmented ABC.

Direct pretty_midi -> ABC converter (music21's MusicXML writer crashes
on a third of our files due to a tuplet-handling bug). We:
  1. Pick ONE canonical MIDI per piece (prefer _C, fall back to first
     available key). NotaGen will do its own 15-key transposition; we
     don't want to feed our pre-augmented copies.
  2. Quantize notes to a 16th-note grid.
  3. Group simultaneous notes into chord brackets, emit ABC bars.
  4. Run NotaGen's preprocess pipeline (lifted from 2_data_preprocess.py)
     to produce interleaved + key-augmented ABC + train/eval JSONL.
"""
import json
import os
import random
import re
import shutil
from pathlib import Path

import pretty_midi
from tqdm import tqdm

random.seed(0)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MIDI_DIR = REPO / "MIDI_transposed"
DATA = ROOT / "data"
ABC_STD = DATA / "abc_standard"
ABC_INTER = DATA / "abc_interleaved"
ABC_AUG = DATA / "abc_augmented"

EVAL_SPLIT = 0.10
KEY_PRIORITY = ["C", "G", "D", "F", "A"]

# 32nd-note grid (8 per quarter note, 32 per 4/4 bar)
GRID_DIVISIONS = 8         # divisions per quarter note
BEATS_PER_BAR = 4          # 4/4
GRID_PER_BAR = GRID_DIVISIONS * BEATS_PER_BAR  # 32

# Tempo normalization. For each piece, compute median inter-onset interval
# (seconds), then pick a Q:1/4=BPM tag so the median onset spacing lands on
# TARGET_MEDIAN_GRIDS grid units. This way every piece — Tech99 sparse and
# repertoire dense alike — uses the SAME rhythmic vocabulary in the ABC,
# while Q carries the real musical tempo. Also rescales the quantization
# grid so re-rendering the ABC sounds identical to the original MIDI.
TARGET_MEDIAN_GRIDS = 4    # median IOI ≈ a sixteenth note (4 × 1/32)
BPM_CLAMP = (40, 240)      # sane tempo range

# ----- NotaGen preprocess (lifted from data/2_data_preprocess.py) -----
from abctoolkit.utils import (
    Barlines,
    Quote_re,
    extract_barline_and_bartext_dict,
    extract_global_and_local_metadata,
    extract_metadata_and_parts,
    remove_bar_no_annotations,
    remove_information_field,
)
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.transpose import Key2index, transpose_an_abc_text


def abc_preprocess_pipeline(abc_path: str, interleaved_folder: str, augmented_folder: str):
    with open(abc_path, "r", encoding="utf-8") as f:
        abc_lines = f.readlines()
    abc_lines = [l for l in abc_lines if l.strip() != ""]
    abc_lines = unidecode_abc_lines(abc_lines)
    abc_lines = remove_information_field(
        abc_lines=abc_lines,
        info_fields=["X:", "T:", "C:", "W:", "w:", "Z:", "%%MIDI"],
    )
    abc_lines = remove_bar_no_annotations(abc_lines)

    _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
    if not bar_no_equal_flag:
        raise RuntimeError("unequal bar number")

    abc_name = os.path.splitext(os.path.basename(abc_path))[0]

    metadata_lines, _ = extract_metadata_and_parts(abc_lines)
    global_md, _ = extract_global_and_local_metadata(metadata_lines)
    if global_md["K"][0] == "none":
        global_md["K"][0] = "C"
    ori_key = global_md["K"][0]

    interleaved_abc = rotate_abc(abc_lines)
    with open(os.path.join(interleaved_folder, abc_name + ".abc"), "w") as w:
        w.writelines(interleaved_abc)

    for k in Key2index.keys():
        transposed_text = transpose_an_abc_text(abc_lines, k)
        transposed_lines = [l + "\n" for l in filter(None, transposed_text.split("\n"))]
        metadata_lines, prefix_dict, left_bl, bar_text, right_bl = \
            extract_barline_and_bartext_dict(transposed_lines)
        reduced = list(metadata_lines)
        for i in range(len(bar_text["V:1"])):
            line = ""
            for symbol in prefix_dict.keys():
                valid = any(
                    c.isalpha() and c not in ["Z", "z", "X", "x"]
                    for c in bar_text[symbol][i]
                )
                if not valid:
                    continue
                if i == 0:
                    patch = ("[" + symbol + "]" + prefix_dict[symbol]
                             + left_bl[symbol][0] + bar_text[symbol][0]
                             + right_bl[symbol][0])
                else:
                    patch = "[" + symbol + "]" + bar_text[symbol][i] + right_bl[symbol][i]
                line += patch
            line += "\n"
            reduced.append(line)
        out = os.path.join(augmented_folder, k, abc_name + "_" + k + ".abc")
        with open(out, "w", encoding="utf-8") as w:
            w.writelines(reduced)

    return abc_name, ori_key


# ----- ABC pitch encoding -----
PITCH_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
SEMITONE_OFFSETS = {0: ("C", ""), 1: ("C", "^"), 2: ("D", ""), 3: ("D", "^"),
                    4: ("E", ""), 5: ("F", ""), 6: ("F", "^"), 7: ("G", ""),
                    8: ("G", "^"), 9: ("A", ""), 10: ("A", "^"), 11: ("B", "")}


def midi_to_abc_pitch(midi_pitch: int) -> str:
    """ABC notation: C4=60 -> 'C'; C5=72 -> 'c'; below C4 add commas; above C5 add apostrophes."""
    octave = midi_pitch // 12 - 1   # MIDI octave (C4=60 -> octave 4)
    semi = midi_pitch % 12
    letter, accidental = SEMITONE_OFFSETS[semi]
    if octave == 4:
        return accidental + letter
    if octave == 5:
        return accidental + letter.lower()
    if octave < 4:
        return accidental + letter + "," * (4 - octave)
    return accidental + letter.lower() + "'" * (octave - 5)


def encode_duration(d_units: int) -> str:
    """Duration in 16th-note units -> ABC duration suffix."""
    if d_units == 1:
        return ""
    return str(d_units)


# ----- core converter -----
def midi_to_abc(midi_path: Path) -> str:
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # --- Tempo normalization ---
    # Derive a per-piece Q tag by computing the median inter-onset interval
    # (in seconds) across all note onsets, then solving:
    #     median_IOI_sec = TARGET_MEDIAN_GRIDS * sec_per_grid
    #     sec_per_grid   = (60 / bpm) / GRID_DIVISIONS
    # => bpm = 60 * TARGET_MEDIAN_GRIDS / (median_IOI_sec * GRID_DIVISIONS)
    onset_times = sorted({n.start
                          for instr in pm.instruments if not instr.is_drum
                          for n in instr.notes})
    iois = [b - a for a, b in zip(onset_times, onset_times[1:]) if b > a]
    if len(iois) >= 4:
        iois_sorted = sorted(iois)
        median_ioi = iois_sorted[len(iois_sorted) // 2]
        bpm = 60.0 * TARGET_MEDIAN_GRIDS / (median_ioi * GRID_DIVISIONS)
    else:
        # fall back to whatever tempo the MIDI declares
        tempo_times, tempos = pm.get_tempo_changes()
        bpm = float(tempos[0]) if len(tempos) else 120.0
    bpm = max(BPM_CLAMP[0], min(BPM_CLAMP[1], bpm))
    bpm_int = int(round(bpm))
    # Using the derived BPM, 1 grid unit corresponds to:
    sec_per_beat = 60.0 / bpm
    sec_per_grid = sec_per_beat / GRID_DIVISIONS

    # Collect all notes (across all instruments) with quantized start/end on the grid.
    raw = []
    for instr in pm.instruments:
        if instr.is_drum:
            continue
        for n in instr.notes:
            start = int(round(n.start / sec_per_grid))
            end = int(round(n.end / sec_per_grid))
            if end <= start:
                end = start + 1
            raw.append((start, end, n.pitch))

    if not raw:
        raise RuntimeError("no notes")

    # Slice the timeline at every note start/end so within each slice the set of
    # sounding pitches is constant. This preserves long sustained notes alongside
    # shorter moving voices (per-pitch ties between slices).
    change_points = sorted({s for s, _, _ in raw} | {e for _, e, _ in raw})
    if change_points[0] > 0:
        change_points = [0] + change_points

    # Pre-index notes by pitch for fast tie lookup
    notes_by_pitch = {}
    for s, e, p in raw:
        notes_by_pitch.setdefault(p, []).append((s, e))

    def is_tied(pitch, slice_start, slice_end):
        """A pitch is tied across a slice boundary iff some source note covers
        [slice_start, slice_end) AND extends past slice_end."""
        for s, e in notes_by_pitch.get(pitch, ()):
            if s <= slice_start and e > slice_end:
                return True
        return False

    # Build slices: (start, dur, sorted_pitches_or_None_for_rest)
    slices = []
    for i in range(len(change_points) - 1):
        t0 = change_points[i]
        t1 = change_points[i + 1]
        active = sorted({p for s, e, p in raw if s <= t0 < e})
        slices.append((t0, t1 - t0, active if active else None))

    # Split slices that cross bar lines
    split = []
    for s, d, pitches in slices:
        cur = s
        rem = d
        while rem > 0:
            bar_pos = cur % GRID_PER_BAR
            room = GRID_PER_BAR - bar_pos
            take = min(room, rem)
            split.append((cur, take, pitches))
            cur += take
            rem -= take

    # Pad to whole bars
    total = (split[-1][0] + split[-1][1]) if split else 0
    if total % GRID_PER_BAR != 0:
        pad = GRID_PER_BAR - (total % GRID_PER_BAR)
        split.append((total, pad, None))

    # Merge consecutive identical-pitch-set slices that are NOT separated by a
    # bar line (the slicer creates a boundary at every note start; if two
    # adjacent slices have the same pitches, they came from a re-articulation
    # OR from a note ending exactly where another with the same pitch begins —
    # but the latter is rare; for the former we want to keep them separate.
    # Actually we should NOT merge: separate slices preserve re-articulation.
    # We only merge rests.)
    merged = []
    for s, d, pitches in split:
        if (merged and pitches is None and merged[-1][2] is None
                and merged[-1][0] + merged[-1][1] == s
                and (merged[-1][0] + merged[-1][1]) % GRID_PER_BAR != 0):
            ps, pd, _ = merged[-1]
            merged[-1] = (ps, pd + d, None)
        else:
            merged.append((s, d, pitches))
    split = merged

    # Emit ABC: walk slices, insert "|" at every bar boundary
    out = []
    bar_buf = []
    for s, d, pitches in split:
        slice_end = s + d
        if pitches is None:
            tok = "z" + encode_duration(d)
        elif len(pitches) == 1:
            p = pitches[0]
            tie = "-" if is_tied(p, s, slice_end) else ""
            tok = midi_to_abc_pitch(p) + encode_duration(d) + tie
        else:
            inner = ""
            for p in pitches:
                inner += midi_to_abc_pitch(p)
                if is_tied(p, s, slice_end):
                    inner += "-"
            tok = "[" + inner + "]" + encode_duration(d)
        bar_buf.append(tok)
        if slice_end % GRID_PER_BAR == 0:
            out.append(" ".join(bar_buf) + " |")
            bar_buf = []

    body = "\n".join(out)

    header = (
        "X:1\n"
        "T:Guzheng\n"
        "L:1/32\n"
        "M:4/4\n"
        f"Q:1/4={bpm_int}\n"
        "K:C\n"
        "V:1 treble nm=\"Guzheng\"\n"
        "V:1\n"
    )
    return header + body + "\n"


# ----- main -----
def pick_canonical_midis():
    by_piece = {}
    for p in sorted(MIDI_DIR.glob("*.mid")):
        m = re.match(r"^(.*)_([ACDFG])\.mid$", p.name)
        if not m:
            continue
        piece, k = m.group(1), m.group(2)
        by_piece.setdefault(piece, {})[k] = p

    chosen = []
    for piece, kmap in by_piece.items():
        for k in KEY_PRIORITY:
            if k in kmap:
                chosen.append((piece, k, kmap[k]))
                break
    return chosen


def main():
    DATA.mkdir(exist_ok=True)
    for d in [ABC_STD, ABC_INTER, ABC_AUG]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for k in Key2index.keys():
        (ABC_AUG / k).mkdir(exist_ok=True)

    canonical = pick_canonical_midis()
    print(f"Found {len(canonical)} unique pieces.")

    succeeded, failed = [], []
    for piece, kletter, midi_path in tqdm(canonical):
        try:
            abc_text = midi_to_abc(midi_path)
            abc_path = ABC_STD / (piece + ".abc")
            abc_path.write_text(abc_text)
            name, ori_key = abc_preprocess_pipeline(
                str(abc_path), str(ABC_INTER), str(ABC_AUG))
            succeeded.append({"path": os.path.join(str(ABC_AUG), name), "key": ori_key})
        except Exception as e:
            failed.append((piece, str(e)[:160]))

    print(f"\nSucceeded: {len(succeeded)}   Failed: {len(failed)}")
    for p, err in failed[:10]:
        print(f"  {p}: {err}")

    random.shuffle(succeeded)
    n_eval = max(1, int(EVAL_SPLIT * len(succeeded)))
    eval_data = succeeded[:n_eval]
    train_data = succeeded[n_eval:]

    with open(DATA / "abc_augmented.jsonl", "w") as w:
        for d in succeeded:
            w.write(json.dumps(d) + "\n")
    with open(DATA / "abc_augmented_train.jsonl", "w") as w:
        for d in train_data:
            w.write(json.dumps(d) + "\n")
    with open(DATA / "abc_augmented_eval.jsonl", "w") as w:
        for d in eval_data:
            w.write(json.dumps(d) + "\n")

    print(f"Train: {len(train_data)}   Eval: {len(eval_data)}")
    print(f"Wrote indices under {DATA}")


if __name__ == "__main__":
    main()

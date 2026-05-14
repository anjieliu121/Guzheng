"""Load guzheng tuning scales and map to allowed MIDI pitches per key."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, FrozenSet, Set

from config import TokenizerConfig, repo_root

_SPN_RE = re.compile(r"^([A-Ga-g])([#b]?)(\d+)$")


def scientific_to_midi(spn: str) -> int:
    s = spn.strip().replace("\u266d", "b").replace("\u266f", "#")
    m = _SPN_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized pitch notation: {spn!r}")
    note, acc, oct_s = m.group(1).upper(), m.group(2), int(m.group(3))
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note]
    if acc == "#":
        pc = (pc + 1) % 12
    elif acc == "b":
        pc = (pc - 1) % 12
    return 12 * (oct_s + 1) + pc


@lru_cache(maxsize=1)
def _scale_to_midi_pitches() -> Dict[str, FrozenSet[int]]:
    path = os.path.join(repo_root(), "metadata", "guzheng_scales.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Set[int]] = {}
    for block in data["scales"]:
        letter = block["scale"]
        if letter == "Bb":
            continue
        mids: Set[int] = set()
        for part in ("entries", "pressed_strings"):
            for entry in block.get(part, []) or []:
                # Support both key names
                midi_val = entry.get("midi")
                if midi_val is not None and 0 <= midi_val <= 127:
                    mids.add(midi_val)
                else:
                    spn = entry.get("spn") or entry.get("scientificPitchNotation")
                    if spn:
                        p = scientific_to_midi(spn)
                        if 0 <= p <= 127:
                            mids.add(p)
        if mids:
            out[letter] = frozenset(mids)
    return dict(out)


def midi_pitches_for_scale(scale: str) -> FrozenSet[int]:
    scale = scale.upper()
    table = _scale_to_midi_pitches()
    if scale not in table:
        scale = "D"
    return table[scale]


def pitch_token_ids_for_scale(scale: str, cfg: TokenizerConfig) -> Set[int]:
    return {cfg.pitch_offset + p for p in midi_pitches_for_scale(scale)}


# Pentatonic pitch classes per scale (for purity checking)
PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},
    "G": {7, 9, 11, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "A": {9, 11, 1, 4, 6},
    "F": {5, 7, 9, 0, 2},
}
PRESSED_PCS = {
    "D": {7, 1}, "G": {0, 6}, "C": {5, 11}, "A": {2, 8}, "F": {10, 4},
}
GUZHENG_PITCH_RANGE = (37, 86)  # D2 to D6

"""MIDI tokenizer for guzheng sequences."""

import os
import mido
from dataclasses import dataclass
from typing import List, Tuple

from config import TokenizerConfig


def scale_from_midi_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_A", "_C", "_D", "_F", "_G"):
        if name.endswith(suf):
            return suf[-1]
    return "D"


@dataclass
class Note:
    onset: int
    pitch: int
    duration: int
    velocity: int


class MidiTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.cfg = config
        self.tick_res = config.tick_resolution

    def midi_to_notes(self, midi_path: str) -> Tuple[List[Note], int, int]:
        mid = mido.MidiFile(midi_path)
        tpb = mid.ticks_per_beat
        tempo = 500000

        notes = []
        for track in mid.tracks:
            abs_time = 0
            pending: dict = {}
            for msg in track:
                abs_time += msg.time
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                if msg.type == "note_on" and msg.velocity > 0:
                    pending[(msg.note, msg.channel)] = (abs_time, msg.velocity)
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    key = (msg.note, msg.channel)
                    if key in pending:
                        onset, vel = pending.pop(key)
                        dur = abs_time - onset
                        if dur > 0:
                            notes.append(Note(onset, msg.note, dur, vel))

        notes.sort(key=lambda n: (n.onset, n.pitch))
        return notes, tpb, tempo

    def encode_notes(self, notes: List[Note], scale: str = "D") -> List[int]:
        cfg = self.cfg
        tokens = [cfg.bos_token, cfg.key_token_id(scale)]
        prev_onset = 0

        for note in notes:
            delta = max(0, note.onset - prev_onset)
            ts = min(round(delta / self.tick_res), cfg.max_time_shift)
            tokens.append(cfg.time_shift_offset + ts)
            tokens.append(cfg.pitch_offset + note.pitch)
            dur = max(1, min(round(note.duration / self.tick_res), cfg.max_duration))
            tokens.append(cfg.duration_offset + dur - 1)
            vel_bin = min(
                note.velocity * cfg.num_velocity_bins // 128,
                cfg.num_velocity_bins - 1,
            )
            tokens.append(cfg.velocity_offset + vel_bin)
            prev_onset = note.onset

        tokens.append(cfg.eos_token)
        return tokens

    def encode_midi(self, midi_path: str) -> List[int]:
        notes, _, _ = self.midi_to_notes(midi_path)
        return self.encode_notes(notes, scale_from_midi_filename(midi_path))

    def decode_tokens(self, tokens: List[int]) -> List[Note]:
        cfg = self.cfg
        notes: List[Note] = []
        onset = 0
        i = 0

        while i < len(tokens):
            tok = tokens[i]
            if tok in (cfg.pad_token, cfg.bos_token, cfg.eos_token):
                i += 1
                continue
            if cfg.is_key_token_id(tok):
                i += 1
                continue
            if not (cfg.time_shift_offset <= tok < cfg.pitch_offset):
                i += 1
                continue

            onset += (tok - cfg.time_shift_offset) * self.tick_res
            i += 1

            if i >= len(tokens):
                break
            tok = tokens[i]
            if not (cfg.pitch_offset <= tok < cfg.duration_offset):
                continue
            pitch = tok - cfg.pitch_offset
            i += 1

            dur = 240
            if i < len(tokens):
                tok = tokens[i]
                if cfg.duration_offset <= tok < cfg.velocity_offset:
                    dur = (tok - cfg.duration_offset + 1) * self.tick_res
                    i += 1

            vel = 80
            if i < len(tokens):
                tok = tokens[i]
                if cfg.velocity_offset <= tok < cfg.velocity_offset + cfg.num_velocity_bins:
                    vel = (tok - cfg.velocity_offset) * 128 // cfg.num_velocity_bins
                    vel += 128 // (2 * cfg.num_velocity_bins)
                    vel = min(127, max(1, vel))
                    i += 1

            notes.append(Note(onset, pitch, dur, vel))

        return notes

    def tokens_to_midi(
        self,
        tokens: List[int],
        output_path: str,
        ticks_per_beat: int = 480,
        tempo: int = 750000,
    ):
        notes = self.decode_tokens(tokens)
        if not notes:
            print("Warning: no notes decoded from tokens")
            return

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo))
        track.append(mido.MetaMessage("track_name", name="Guzheng"))

        events = []
        for n in notes:
            events.append((n.onset, 1, n.pitch, n.velocity))
            events.append((n.onset + n.duration, 0, n.pitch, 0))
        events.sort(key=lambda e: (e[0], e[1]))

        prev = 0
        for abs_t, is_on, pitch, vel in events:
            delta = max(0, abs_t - prev)
            kind = "note_on" if is_on else "note_off"
            track.append(mido.Message(kind, note=pitch, velocity=vel, time=delta))
            prev = abs_t

        track.append(mido.MetaMessage("end_of_track"))
        mid.save(output_path)

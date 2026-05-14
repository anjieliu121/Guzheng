#!/usr/bin/env python3
"""
Post-process generated MIDI to make it sound like guzheng on Ample Sound
Ample China Zheng (ACZ4). Adds articulation keyswitches and a hold-pedal CC,
following the keyswitch map in the ACZ4 Main Panel Manual.

Keyswitch map (Ample labels: C3 = MIDI 60, so C0 = MIDI 24):
  Head Group  (single-press, articulation persists):
    C0 (24)  Sustain          (default)
    C#0 (25) Natural Harmonic
    D0 (26)  Tremolo
    D#0 (27) Glissando
    E0 (28)  Glissando Up
    F0 (29)  Glissando Down
  Body Group  (held while note plays, returns to Sustain when released):
    F#0 (30) Bend and Release
    G0 (31)  Bend Up
    G#0 (32) Bend Down
    A0 (33)  Single Vibrato
    A#0 (34) Vibrato
  Lick:
    B0 (35)

Playable guzheng range: C1-D5 (MIDI 36-86) — keyswitches 24-35 never conflict.

Rules applied (combined ACZ4 spec + musical heuristics):

  1. Hold-pedal CC64=127 at the very start so notes sustain (per manual 2.10).
  2. A Sustain (C0) keyswitch is laid down at t=0 as the baseline articulation.
  3. Long held notes (>= LONG_NOTE_SEC) get a Tremolo head-keyswitch with
     velocity scaled to length (longer => lower vel => longer fade-in, per
     manual 2.2.8: "larger velocity will cause shorter fade in time").
     Tremolo is sticky, so a Sustain keyswitch is laid down right after the
     long note ends to return to the default articulation.
  4. Single isolated high notes (MIDI >= HIGH_HARMONIC) get a Natural Harmonic
     head-keyswitch -- the topmost guzheng notes are commonly played as
     harmonics in real performance.
  5. Ascending runs of >= 4 stepwise notes: keep only the first note with a
     Glissando Up keyswitch; remove the rest (ACZ4 plays the sweep automatically).
     Descending runs likewise use Glissando Down. The trigger note's duration
     is extended to span the full run.
  6. Small ascending intervals (2-3 semitones inside the same scale) on
     medium-length notes get a Bend Up body-keyswitch (held for the note
     duration). This is the classic guzheng "yin/rou" left-hand bend.
  7. Small descending intervals (2-3 semitones) get Bend Down similarly.
  8. Medium-length notes followed by a long rest get a Vibrato body-keyswitch
     (held during the note) to add expressive vibrato to phrase ends.
  9. Note-off velocity is forced to 64 (< 126) so the hold-pedal sustain
     actually works (per manual 2.10 footnote: vel >= 126 disables sustain).
 10. Note-on velocities are gently re-mapped into a musically useful 55..110
     range so dynamics are not flat.

The transformations are conservative: notes are preserved with adjusted
velocities and keyswitch events inserted around them. The exception is
glissando runs, where only the trigger note is kept — the sampler plays
the sweep, so the intermediate run notes are removed to avoid doubling.

Usage:
  python3 scripts/postprocess_guzheng_keyswitches.py \
      --input_dir test_and_trial_6/generated/constrained \
      --output_dir test_and_trial_6/generated/constrained_ks
"""

import argparse
import os
import sys
from collections import namedtuple

import mido

# ---------- Keyswitch MIDI numbers (Ample C3=60 convention) ----------
KS_SUSTAIN          = 24  # C0
KS_NAT_HARMONIC     = 25  # C#0
KS_TREMOLO          = 26  # D0
KS_GLISSANDO        = 27  # D#0
KS_GLISSANDO_UP     = 28  # E0
KS_GLISSANDO_DOWN   = 29  # F0
KS_BEND_AND_RELEASE = 30  # F#0
KS_BEND_UP          = 31  # G0
KS_BEND_DOWN        = 32  # G#0
KS_SINGLE_VIBRATO   = 33  # A0
KS_VIBRATO          = 34  # A#0
KS_LICK             = 35  # B0

HEAD_GROUP = {KS_SUSTAIN, KS_NAT_HARMONIC, KS_TREMOLO,
              KS_GLISSANDO, KS_GLISSANDO_UP, KS_GLISSANDO_DOWN}
BODY_GROUP = {KS_BEND_AND_RELEASE, KS_BEND_UP, KS_BEND_DOWN,
              KS_SINGLE_VIBRATO, KS_VIBRATO}

GUZHENG_MIN = 38  # D1 (21-string guzheng lowest)
GUZHENG_MAX = 86  # D6

# D-pentatonic string tuning: pitch classes D(2) E(4) F#(6) A(9) B(11)
# These are the open-string pitches across all octaves on a standard guzheng.
D_PENTA_PCS = {2, 4, 6, 9, 11}

def nearest_string_below(pitch):
    """Return the nearest D-pentatonic open-string pitch at or below `pitch`."""
    pc = pitch % 12
    if pc in D_PENTA_PCS:
        return pitch  # already an open string
    # Walk downward until we hit a pentatonic pitch class
    for offset in range(1, 12):
        cand = pitch - offset
        if cand % 12 in D_PENTA_PCS:
            return max(GUZHENG_MIN, cand)
    return pitch  # fallback

def nearest_string_above(pitch):
    """Return the nearest D-pentatonic open-string pitch at or above `pitch`."""
    pc = pitch % 12
    if pc in D_PENTA_PCS:
        return pitch
    for offset in range(1, 12):
        cand = pitch + offset
        if cand % 12 in D_PENTA_PCS:
            return min(GUZHENG_MAX, cand)
    return pitch

# ---------- Heuristic thresholds ----------
LONG_NOTE_SEC      = 1.40   # >= this triggers Tremolo
VIBRATO_MIN_SEC    = 0.55
VIBRATO_MAX_SEC    = 1.39
BEND_NOTE_MIN_SEC  = 0.25   # bends only on notes long enough to hear
RUN_MIN_LEN        = 4      # ascending/descending run length for glissando
RUN_MAX_GAP_SEC    = 0.30   # consecutive notes in a run must be tight
HIGH_HARMONIC      = 81     # high notes that may be harmonics
PHRASE_END_GAP_SEC = 0.60   # silence after a note that marks a phrase end

Note = namedtuple("Note", "start end pitch velocity index")


# ---------- MIDI <-> note list conversion ----------
def midi_to_notes(mid):
    """Flatten a MidiFile to an absolute-time list of Notes (single track assumed)."""
    notes = []
    abs_ticks = 0
    open_notes = {}  # pitch -> (start_tick, vel, idx)
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat
    # Walk merged track to keep tempo events in order
    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            open_notes[msg.note] = (abs_ticks, msg.velocity, len(notes))
            notes.append(None)  # placeholder, filled at note_off
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in open_notes:
                start, vel, idx = open_notes.pop(msg.note)
                notes[idx] = (start, abs_ticks, msg.note, vel)
    notes = [n for n in notes if n is not None]
    notes.sort(key=lambda n: (n[0], n[2]))

    # Convert ticks -> seconds using a tempo map walk (we approximate with the
    # last seen tempo; generated MIDI almost always has a single set_tempo).
    # For correctness with multiple tempo changes we redo a tick->sec scan.
    tempo_map = []
    abs_ticks = 0
    cur_tempo = 500000
    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.type == "set_tempo":
            tempo_map.append((abs_ticks, msg.tempo))
            cur_tempo = msg.tempo
    if not tempo_map:
        tempo_map = [(0, 500000)]

    def tick_to_sec(target):
        sec = 0.0
        prev_tick = 0
        prev_tempo = tempo_map[0][1]
        for t_tick, t_tempo in tempo_map:
            if t_tick >= target:
                break
            sec += mido.tick2second(t_tick - prev_tick, ticks_per_beat, prev_tempo)
            prev_tick = t_tick
            prev_tempo = t_tempo
        sec += mido.tick2second(target - prev_tick, ticks_per_beat, prev_tempo)
        return sec

    out = []
    for i, (start_t, end_t, pitch, vel) in enumerate(notes):
        out.append(Note(start=tick_to_sec(start_t),
                        end=tick_to_sec(end_t),
                        pitch=pitch, velocity=vel, index=i))
    return out, ticks_per_beat, tempo_map


# ---------- Heuristics ----------
def humanize_velocity(v):
    """Squash to a musically useful range with mild contrast."""
    # original 1..127 -> 55..110
    return int(round(55 + (max(1, min(127, v)) - 1) * (110 - 55) / 126))


def detect_runs(notes):
    """Return list of (start_idx, end_idx, direction) for ascending/descending runs."""
    runs = []
    n = len(notes)
    i = 0
    while i < n - 1:
        # try ascending
        j = i
        while (j + 1 < n
               and notes[j + 1].pitch > notes[j].pitch
               and notes[j + 1].pitch - notes[j].pitch <= 4
               and notes[j + 1].start - notes[j].end <= RUN_MAX_GAP_SEC):
            j += 1
        if j - i + 1 >= RUN_MIN_LEN:
            runs.append((i, j, +1))
            i = j + 1
            continue
        # try descending
        j = i
        while (j + 1 < n
               and notes[j + 1].pitch < notes[j].pitch
               and notes[j].pitch - notes[j + 1].pitch <= 4
               and notes[j + 1].start - notes[j].end <= RUN_MAX_GAP_SEC):
            j += 1
        if j - i + 1 >= RUN_MIN_LEN:
            runs.append((i, j, -1))
            i = j + 1
            continue
        i += 1
    return runs


def plan_articulations(notes):
    """
    Decide which keyswitch events to attach to each note.
    Returns:
      head_marks  : dict idx -> head-group keyswitch (single press before note)
      body_marks  : dict idx -> body-group keyswitch (held during note)
      pitch_overrides : dict idx -> new MIDI pitch (for bends: the starting string)
      restore_sus : set of idx whose note ends a sticky head articulation
      skip_notes  : set of idx to remove entirely (consumed by glissando)
    """
    head_marks = {}
    body_marks = {}
    pitch_overrides = {}
    restore_sus = set()
    skip_notes = set()

    ret = lambda: (head_marks, body_marks, pitch_overrides, restore_sus, skip_notes)

    if not notes:
        return ret()

    # 1) Glissando Up / Down on stepwise runs
    #    ACZ4 plays the glissando sweep automatically from the trigger note.
    #    We keep only the FIRST note of the run and remove the rest —
    #    the sampler generates the ascending/descending sweep itself.
    #    The kept note's duration is extended to cover the full run span.
    for start_i, end_i, direction in detect_runs(notes):
        ks = KS_GLISSANDO_UP if direction > 0 else KS_GLISSANDO_DOWN
        head_marks[start_i] = ks
        # Mark all run notes after the first for removal
        for j in range(start_i + 1, end_i + 1):
            skip_notes.add(j)
        # Restore Sustain after the glissando trigger note
        restore_sus.add(start_i)

    # 2) Tremolo on long held notes
    for i, n in enumerate(notes):
        if i in skip_notes:
            continue
        dur = n.end - n.start
        if dur >= LONG_NOTE_SEC and i not in head_marks:
            head_marks[i] = KS_TREMOLO
            restore_sus.add(i)

    # 3) Natural Harmonic on isolated very high notes
    for i, n in enumerate(notes):
        if i in skip_notes:
            continue
        if n.pitch < HIGH_HARMONIC or i in head_marks:
            continue
        prev_gap = n.start - notes[i - 1].end if i > 0 else 999
        next_gap = notes[i + 1].start - n.end if i + 1 < len(notes) else 999
        if prev_gap > 0.20 and next_gap > 0.20:
            head_marks[i] = KS_NAT_HARMONIC
            restore_sus.add(i)

    # 4) Bend Up / Bend Down on small stepwise intervals
    #    IMPORTANT: For bends the MIDI note must be the STARTING pitch of the
    #    gesture (the physical string being plucked), not the target pitch.
    #
    #    Bend Up:  pluck the lower string, left hand presses to raise pitch.
    #              MIDI note = previous note's pitch (the string we bend from).
    #              ACZ4 plays that pitch and bends upward.
    #
    #    Bend Down: pre-press string, pluck at higher pitch, release to fall.
    #              MIDI note = previous note's pitch (the starting higher pitch).
    #              ACZ4 plays that pitch and bends downward.
    #
    #    In both cases: note[i].pitch → note[i-1].pitch (the gesture's start).
    #    (only on medium-length notes; cap how often to keep it musical)
    bend_count = 0
    for i in range(1, len(notes)):
        if i in skip_notes or i in head_marks or i in body_marks:
            continue
        n = notes[i]
        prev = notes[i - 1]
        dur = n.end - n.start
        if dur < BEND_NOTE_MIN_SEC:
            continue
        interval = n.pitch - prev.pitch
        # ACZ4 automatically determines bend amount (major 2nd or minor 3rd)
        # based on the Key setting (Section 2.3 of ACZ4 manual).
        # With Key=D: D→E(2), E→F#(2), F#→A(3), A→B(2), B→D(3) all work.
        if 2 <= interval <= 3:
            body_marks[i] = KS_BEND_UP
            # Start from the lower string (prev note) and bend up to target
            pitch_overrides[i] = nearest_string_below(prev.pitch)
            bend_count += 1
        elif -3 <= interval <= -2:
            body_marks[i] = KS_BEND_DOWN
            # Start from the higher pitch (prev note) and bend down
            pitch_overrides[i] = nearest_string_above(prev.pitch)
            bend_count += 1
        # rate-limit so the piece does not turn into all bends
        if bend_count >= max(2, len(notes) // 12):
            break

    # 5) Vibrato on phrase-end notes
    for i, n in enumerate(notes):
        if i in skip_notes or i in head_marks or i in body_marks:
            continue
        dur = n.end - n.start
        if not (VIBRATO_MIN_SEC <= dur <= VIBRATO_MAX_SEC):
            continue
        next_gap = notes[i + 1].start - n.end if i + 1 < len(notes) else 999
        if next_gap >= PHRASE_END_GAP_SEC:
            body_marks[i] = KS_VIBRATO

    return ret()


# ---------- Building the new MIDI ----------
def build_event_list(notes, head_marks, body_marks, pitch_overrides, restore_sus,
                     skip_notes, ticks_per_beat, tempo_map):
    """
    Produce a list of (abs_tick, order, mido.Message) ready to be sorted and
    deltified back into a MidiTrack.
    """
    events = []
    order = 0

    def sec_to_tick(sec):
        # Inverse of tick_to_sec walk.
        ticks = 0
        prev_tick = 0
        prev_tempo = tempo_map[0][1]
        for t_tick, t_tempo in tempo_map:
            seg_sec = mido.tick2second(t_tick - prev_tick, ticks_per_beat, prev_tempo)
            if seg_sec >= sec:
                ticks = prev_tick + int(round(mido.second2tick(sec, ticks_per_beat, prev_tempo)))
                return ticks
            sec -= seg_sec
            prev_tick = t_tick
            prev_tempo = t_tempo
        return prev_tick + int(round(mido.second2tick(sec, ticks_per_beat, prev_tempo)))

    def add(tick, msg):
        nonlocal order
        events.append((max(0, tick), order, msg))
        order += 1

    # Hold pedal on at the very start
    add(0, mido.Message("control_change", control=64, value=127, time=0))
    # Default Sustain articulation
    add(0, mido.Message("note_on",  note=KS_SUSTAIN, velocity=64, time=0))
    add(1, mido.Message("note_off", note=KS_SUSTAIN, velocity=0, time=0))

    KS_LEAD_TICKS  = max(2, ticks_per_beat // 32)   # tiny lead before a note
    KS_HOLD_PAD    = max(2, ticks_per_beat // 32)   # body ks released after note

    # Pre-compute time shifts: when glissando run notes are removed, shift all
    # subsequent notes earlier to close the gap. The trigger note keeps its
    # original short duration — ACZ4 plays the sweep automatically.
    time_shift = 0.0   # accumulated seconds to shift earlier
    shift_at = {}      # note index -> cumulative shift in seconds at that point
    for i, n in enumerate(notes):
        if i in skip_notes:
            # This note is removed; accumulate its duration into the shift
            gap_before = n.start - notes[i - 1].end if i > 0 else 0
            note_span = n.end - n.start + max(0, gap_before)
            # Only shift by the IOI contribution of removed notes
            if i + 1 < len(notes) and i + 1 not in skip_notes:
                # Last removed note in run: shift = run_last.end - trigger.end
                # Find trigger (walk back to first non-skipped)
                pass
            time_shift += (n.end - n.start)
            # If there's a small gap between consecutive run notes, include it
            if i + 1 < len(notes) and (i + 1) in skip_notes:
                time_shift += max(0, notes[i + 1].start - n.end)
        shift_at[i] = time_shift

    # For glissando triggers: also add the gap between trigger note end and
    # the first removed note's start
    for i in range(len(notes)):
        if (i in head_marks and head_marks[i] in (KS_GLISSANDO_UP, KS_GLISSANDO_DOWN)
                and i + 1 < len(notes) and i + 1 in skip_notes):
            gap = notes[i + 1].start - notes[i].end
            if gap > 0:
                # Add this gap to all shifts from i+1 onward
                for j in range(i + 1, len(notes)):
                    shift_at[j] = shift_at.get(j, 0) + gap

    for i, n in enumerate(notes):
        # Skip notes consumed by glissando runs (sampler plays the sweep)
        if i in skip_notes:
            continue

        cur_shift = shift_at.get(i, 0.0)
        start_tick = sec_to_tick(n.start - cur_shift)
        end_tick = sec_to_tick(n.end - cur_shift)
        if end_tick <= start_tick:
            end_tick = start_tick + 1

        # Head-group keyswitch (single short press just before the note)
        if i in head_marks:
            ks = head_marks[i]
            ks_vel = 96
            # Tremolo: lower velocity for longer notes (longer fade-in)
            if ks == KS_TREMOLO:
                dur = n.end - n.start
                ks_vel = 60 if dur > 2.0 else (80 if dur > 1.7 else 110)
            ks_tick = max(0, start_tick - KS_LEAD_TICKS)
            add(ks_tick,     mido.Message("note_on",  note=ks, velocity=ks_vel, time=0))
            add(ks_tick + 1, mido.Message("note_off", note=ks, velocity=0,      time=0))

        # Body-group keyswitch (held while the note plays)
        if i in body_marks:
            ks = body_marks[i]
            ks_vel = 96
            ks_on  = max(0, start_tick - KS_LEAD_TICKS)
            ks_off = end_tick + KS_HOLD_PAD
            add(ks_on,  mido.Message("note_on",  note=ks, velocity=ks_vel, time=0))
            add(ks_off, mido.Message("note_off", note=ks, velocity=0,      time=0))

        # The melodic note itself (clamped to guzheng range).
        # For bends, use the overridden starting pitch instead of the target.
        # For non-bend notes that are off the pentatonic scale, snap to the
        # nearest open string (the guzheng has no frets — only string bends
        # produce non-pentatonic pitches, handled above).
        raw_pitch = pitch_overrides.get(i, n.pitch)
        if i not in pitch_overrides and raw_pitch % 12 not in D_PENTA_PCS:
            raw_pitch = nearest_string_below(raw_pitch)
        pitch = max(GUZHENG_MIN, min(GUZHENG_MAX, raw_pitch))
        vel   = humanize_velocity(n.velocity)
        add(start_tick, mido.Message("note_on",  note=pitch, velocity=vel, time=0))
        add(end_tick,   mido.Message("note_off", note=pitch, velocity=64,  time=0))

        # Restore Sustain after a sticky head articulation finishes
        if i in restore_sus:
            r_tick = end_tick + KS_HOLD_PAD + 1
            add(r_tick,     mido.Message("note_on",  note=KS_SUSTAIN, velocity=64, time=0))
            add(r_tick + 1, mido.Message("note_off", note=KS_SUSTAIN, velocity=0,  time=0))

    return events


def events_to_track(events, tempo_map):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Guzheng KS", time=0))
    # Re-emit the original tempo map at tick 0 (single tempo is the common case).
    base_tempo = tempo_map[0][1] if tempo_map else 500000
    track.append(mido.MetaMessage("set_tempo", tempo=base_tempo, time=0))

    events.sort(key=lambda e: (e[0], e[1]))
    last_tick = 0
    for abs_tick, _order, msg in events:
        delta = abs_tick - last_tick
        if delta < 0:
            delta = 0
        track.append(msg.copy(time=delta))
        last_tick = abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def process_file(in_path, out_path):
    mid = mido.MidiFile(in_path)
    notes, tpb, tempo_map = midi_to_notes(mid)
    if not notes:
        print(f"  SKIP {os.path.basename(in_path)}: no notes")
        return None
    head, body, pitch_ov, restore, skip = plan_articulations(notes)
    events = build_event_list(notes, head, body, pitch_ov, restore, skip, tpb, tempo_map)
    track = events_to_track(events, tempo_map)

    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    new_mid.tracks.append(track)
    new_mid.save(out_path)

    return {
        "notes":      len(notes),
        "gliss_skip": len(skip),
        "head_ks":    len(head),
        "body_ks":    len(body),
        "tremolo":    sum(1 for v in head.values() if v == KS_TREMOLO),
        "gliss_up":   sum(1 for v in head.values() if v == KS_GLISSANDO_UP),
        "gliss_dn":   sum(1 for v in head.values() if v == KS_GLISSANDO_DOWN),
        "harmonic":   sum(1 for v in head.values() if v == KS_NAT_HARMONIC),
        "bend_up":    sum(1 for v in body.values() if v == KS_BEND_UP),
        "bend_dn":    sum(1 for v in body.values() if v == KS_BEND_DOWN),
        "vibrato":    sum(1 for v in body.values() if v == KS_VIBRATO),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".mid"))
    if not files:
        print(f"No .mid files in {args.input_dir}")
        sys.exit(0)

    print(f"Processing {len(files)} files: {args.input_dir}  ->  {args.output_dir}")
    totals = {}
    for fn in files:
        out = os.path.join(args.output_dir, fn.replace(".mid", "_ks.mid"))
        stats = process_file(os.path.join(args.input_dir, fn), out)
        if stats is None:
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"  {fn}: notes={stats['notes']:4d}  head={stats['head_ks']:3d} "
              f"(trem={stats['tremolo']}, glU={stats['gliss_up']}, glD={stats['gliss_dn']}, "
              f"harm={stats['harmonic']})  body={stats['body_ks']:3d} "
              f"(bU={stats['bend_up']}, bD={stats['bend_dn']}, vib={stats['vibrato']})")

    print("\nTotals:", totals)


if __name__ == "__main__":
    main()

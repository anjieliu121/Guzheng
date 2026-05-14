# MIDI Tokenization Schemes

- **MidiTok:** https://github.com/Natooz/MidiTok | [Docs](https://miditok.readthedocs.io/) | [Paper](https://arxiv.org/abs/2310.17202)
- **Symusic (MIDI I/O):** https://github.com/Yikai-Liao/symusic

## Overview

MidiTok is the standard library for tokenizing MIDI files for music generation models. It supports multiple tokenization schemes, BPE compression, and integrates with HuggingFace tokenizers.

## Tokenization Schemes

### REMI (Revamped MIDI)

The foundational scheme. Single-track focused.

**Token sequence:** `Bar → Position → Pitch → Velocity → Duration`

- **Time:** Bar tokens for measures, Position tokens for within-bar location (resolution set by `beat_res`)
- **Notes:** Pitch, Velocity, Duration token triplets
- **Multi-track:** Not natively supported (see REMI+)

### REMI+ (Extended REMI)

Extended REMI for multi-track, multi-signature music (introduced in FIGARO).

**Token sequence:** `Track_Start → Bar → Position → Program → Pitch → Velocity → Duration → ... → Track_End`

- Adds `Program` tokens before Pitch for instrument identification
- Adds `TimeSignature` tokens
- Tracks placed sequentially in the token stream
- **Used by MIDI-RWKV** with BPE compression (663 base → 16,000 post-BPE)

### MIDI-Like

Closest to raw MIDI messages.

**Token sequence:** `TimeShift → NoteOn(Pitch) → NoteOff` or `TimeShift → Program → Pitch → NoteOff`

- **Time:** TimeShift tokens (relative deltas between events)
- Used by Music Transformer and MT3
- **Limitation:** Poor with extended silence (uses max TimeShift repeatedly)
- May alter durations of overlapping notes

### TSD (Time Shift Duration)

Like MIDI-Like but with explicit Duration tokens instead of NoteOff.

**Token sequence:** `TimeShift → Program → Pitch → Velocity → Duration`

- Duration tokens shown to outperform NoteOff for generation
- Same silence limitation as MIDI-Like

### Structured

Fixed token order for consistency.

**Token sequence:** `Program → Pitch → Velocity → Duration → TimeShift` (always this order)

- Simultaneous notes get TimeShift=0
- No additional tokens can be inserted between core elements
- Used in Piano Inpainting Application

### CPWord (Compound Word)

Multiple token types pooled into single embeddings.

**Compound token:** `[Family, Bar/Position, Pitch, Velocity, Duration, Program, ...]`

- Reduces sequence length via embedding pooling
- **Not recommended for generation with small models** — requires sampling from multiple distributions
- Requires multiple loss functions during training

### Octuple

Similar to CPWord with even more aggressive pooling.

**Compound token:** `[Pitch, Position, Bar, Velocity, Duration, Program, Tempo, TimeSignature]`

- Tracks with same Program merged
- **Very short sequences** but cannot represent time signatures accurately
- Used by MusicBERT
- Not recommended for generation with small models

### MuMIDI

Designed specifically for multi-track generation.

**Compound token:** `[Pitch/Position/Bar/Program, BarPosEnc, PositionPosEnc, Tempo, Velocity, Duration]`

- Track tokens precede note tokens
- Built-in + learned positional encoding
- Not recommended for generation with small models

### MMM (Multi-Track Music Machine)

Designed for music inpainting/infilling.

- Tracks tokenized independently then concatenated
- `Bar_Fill` tokens mark sections for infilling
- Duration tokens instead of NoteOff (better for causal generation)
- Requires `density_bins_max` configuration
- Only first track's tempos decoded during generation

### PerTok (Performance Tokenizer)

Captures performance nuances.

**Token sequence:** `Bar → TimeShift → Pitch → Velocity → MicroTiming → Duration`

- **MicroTiming tokens:** Quantized base TimeShift + remainder for micro-timing
- Captures complex rhythms (16ths, 32nds, triplets)
- Minimizes vocabulary and sequence length
- Configurable via `use_microtiming`, `max_microtiming_shift`, `num_microtiming_bins`

## Comparison Table

| Scheme | Time | Seq Length | Multi-track | Best For |
|--------|------|-----------|-------------|----------|
| REMI | Bar/Pos | Moderate | Via REMI+ | General single-track |
| REMI+ | Bar/Pos | Moderate | Native | Multi-track generation |
| MIDI-Like | TimeShift | Longer | Optional | MIDI-aligned tasks |
| TSD | TimeShift | Moderate | Optional | Generation (better than MIDI-Like) |
| Structured | TimeShift | Moderate | Optional | Inpainting |
| CPWord | Bar/Pos | Short | Optional | Large model understanding |
| Octuple | Bar/Pos | Very short | Native | BERT-style understanding |
| MuMIDI | Bar/Pos | Short | Native | Multi-track |
| MMM | Bar | Moderate | Independent | Infilling/inpainting |
| PerTok | Bar/TimeShift | Short | Optional | Expressive performance |

## Moonbeam's Approach (Not MidiTok-Based)

Moonbeam uses a custom **compound token** scheme with FME:

**6-tuple:** `(onset, duration, octave, pitch_class, instrument, velocity)`

- 5 of 6 attributes use continuous sinusoidal FME (not discrete tokens)
- Only instrument uses standard embedding lookup
- Absolute onsets (not deltas)
- See [moonbeam.md](moonbeam.md) for full details

## BPE for MIDI

BPE (Byte Pair Encoding) compresses token sequences by merging frequent token pairs:

- Supported algorithms: BPE, Unigram, WordPiece (via HuggingFace tokenizers)
- Distinguishes "basic" tokens (original vocabulary) from "learned" tokens (merged pairs)
- MIDI-RWKV: 663 base → 16,000 post-BPE vocabulary
- Significantly reduces sequence length, enabling longer context windows

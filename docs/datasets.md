# Datasets

## GigaMIDI

- **HuggingFace:** https://huggingface.co/datasets/Metacreation/GigaMIDI
- **GitHub:** https://github.com/Metacreation-Lab/GigaMIDI-Dataset
- **Paper:** https://transactions.ismir.net/articles/10.5334/tismir.203
- **License:** CC BY-NC 4.0

### Scale
| Metric | Value |
|--------|-------|
| Unique MIDI files | 2,136,218 |
| Total tracks | 6,891,738 |
| Total beats | 153,947,183 |
| Instruments | 128 GM melodic + 47 percussion (175 total) |
| Non-expressive loops | 9.2M |
| Expressive loops | 2.3M |

Split: 80% train / 10% validation / 10% test.

### Collection
- Aggregated from Zenodo, GitHub, web scraping
- Deduplicated via MD5 checksums
- Standardized to General MIDI spec (drum track remapping, channel correction)
- File names anonymized via MD5 hashing
- Collected under Canadian Fair Dealing provisions

### Metadata
Each file includes:
- Core: `md5`, `num_tracks`, `TPQN`, `total_notes`, `tempo`
- Performance: `avg_note_duration`, `avg_velocity`, `min/max_velocity`, `NOMML`
- Loop detection: `loop_track_idx`, `loop_instrument_type`, `loop_start/end`, `loop_duration_beats`, `loop_note_density`
- Genre: `music_styles_curated`, `music_style_scraped`, audio-text matches from Discogs/Last.fm/Tagtraum
- Identity: `title`, `artist`, Spotify/MusicBrainz IDs

### Expressiveness Heuristics
Novel metrics for classifying expressive vs non-expressive performances:

- **NOMML (Note Onset Median Metric Level):** Median deviation of note onsets from quantized metric grid
- **DNVR (Distinctive Note Velocity Ratio):** Dynamic variation in velocities
- **DNODR (Distinctive Note Onset Deviation Ratio):** Microtiming variations in onset timing

~71.4% non-expressive, ~28.6% expressive tracks.

### Usage
```python
from datasets import load_dataset
from symusic import Score

dataset = load_dataset("Metacreation/GigaMIDI", split="train")
# Or streaming:
dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)

# Filter by quality
def is_valid(score, min_bars=8, min_notes=50):
    score = Score.from_midi(score) if isinstance(score, bytes) else score
    return len(get_bars_ticks(score)) >= min_bars and score.note_num() > min_notes

dataset = dataset.filter(lambda ex: is_valid(ex["music"]))
```

### Bias
Western digital music production bias — piano dominance (ubiquitous MIDI controller), drum prevalence (drum pads/machines), underrepresentation of instruments not commonly played via MIDI.

---

## XMIDI

- **Paper:** https://arxiv.org/html/2501.08809v1

### Scale
- **108,023 MIDI files** (~5,278 hours, avg 176s per file)
- ~10× larger than previous largest emotion-labeled dataset

### Labels
- **11 emotions:** exciting, warm, happy, romantic, funny, sad, angry, lazy, quiet, fear, magnificent
- **6 genres:** rock, pop, country, jazz, classical, folk

### Instruments (17 types)
Piano, xylophone, organ, guitar, bass, violin, harp, string, trumpet, tuba, sax, flute, lead, pad, **pipa**, **guzheng**, drum.

**Notable:** One of the few large-scale MIDI datasets with explicit Chinese instrument categories (guzheng and pipa).

### Collection
Crawled from Internet Archive, GitHub, Reddit. Automatic cleaning, deduplication via chroma features + cosine similarity, manual verification by annotators (≥3 experts, 95% accuracy threshold).

---

## POP909

- 909 Chinese pop songs
- Three tracks per song: melody, bridge, piano accompaniment
- Used as benchmark for infilling evaluation in MIDI-RWKV
- Relevant for Chinese music context

---

## Aria-MIDI

- Piano MIDI dataset
- Used in MIREX 2025 experiments alongside MIDI-RWKV

---

## CCMUSIC Database (Guzheng)

- **Link:** https://ccmusic-database.github.io/en/database/csmtd.html
- **GuzhengTech99:** https://lidcc.github.io/GuzhengTech99/
- ~300 audio recordings, ~20 hours
- Guzheng playing technique detection dataset
- Reference: Li et al., ISMIR 2022

---

## Chinese Instrument Datasets (General)

- **AI Audio Datasets List:** https://github.com/AMAAI-Lab/ai-audio-datasets-list
- Covers Erhu, Pipa, Guzheng, Yangqin, Dizi, and other Chinese instruments

---

## Moonbeam Pretraining Data

81.6K hours (~18B tokens) across 20+ datasets:

| Dataset | Type | Hours | Tokens |
|---------|------|-------|--------|
| AriaMIDI | Piano | 57,380 | 8.40B |
| MetaMIDI | Multitrack | 18,189 | 7.26B |
| SymphonyNet | Classical | 3,135 | 1.61B |
| Maestro | Piano | 197 | 42.24M |
| + 15 more | Various | Various | Various |

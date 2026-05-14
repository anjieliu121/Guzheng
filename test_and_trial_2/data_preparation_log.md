# Data Preparation Log

## Date: 2026-03-25

## 1. Inventory

- **Original MIDI files:** 18 (in `MIDI/`)
- **Transposed MIDI files:** 72 (in `MIDI_transposed/`)
- **Total corpus:** 90 files
- 18 pieces × 1-5 pentatonic transpositions each

## 2. Data Inspection Summary

All 90 files passed inspection:
- **Pentatonic adherence:** 100% (including pressed-string notes 4 and 7)
- **Pitch range:** MIDI 37-86 (C#2-D6) — fully within 21-string guzheng compass
- **Track count:** All single-track
- **Instrument program:** All set to 0 (piano GM, will be rendered with guzheng soundfont)
- **Velocity:** Constant 64 across all files (no dynamic variation encoded)
- **Note density:** Mean 3.53 notes/sec (range: 0.92-8.75)
- **Ornaments:** 555 tremolo regions, 326 glissando regions detected
- **Texture:** Mean max simultaneous notes 3.0; 46/90 files are predominantly monophonic (≤2 simultaneous)

## 3. Cleaning Actions

No cleaning required:
- All notes are within guzheng range
- All notes are pentatonic (including pressed strings)
- All files are single-track
- No corrupt/empty files
- Prior validation pipeline (`scripts/check_midi_note_quality.py`) already ensured:
  - Note durations between 10ms-10240ms
  - No overlapping notes
  - Non-pentatonic notes corrected to nearest pentatonic pitch

## 4. Augmentation

Transposition augmentation already applied:
- Each of 18 pieces transposed to all valid pentatonic keys (D, G, F, C, A)
- Some pieces have fewer transpositions due to compass constraints
- Result: 72 transposed files (4x average augmentation)

No additional augmentation applied (tempo variation, etc.) — the 90-file corpus is sufficient for initial training. Can revisit if overfitting is observed.

## 5. Validation Split

### MIDI-RWKV
Already split in `archive/midi-rwkv/RWKV-PEFT/data/test/` (6 files):
- chun_miao_A.mid, chun_miao_D.mid
- dan_dian_tou_luan_cha_hua_G.mid
- shang_lou_D.mid
- ya_shan_ai_F.mid
- zhan_tai_feng_D.mid

**Note:** Training data is missing 18 files from recent additions (ba_yue_gui_hua_bian_di_kai, bai_jia_chun_han_gao, da_yan). These need to be added and the preprocessed cache rebuilt.

### Moonbeam
Will use same validation files when setting up Moonbeam preprocessing.

## 6. Data Quality Notes

- **Constant velocity** is a limitation — real guzheng has rich dynamics. Models won't learn dynamic variation from this data. Consider this when evaluating output.
- **Large leap rates** (mean 13%) are genuine guzheng idiom — octave-spanning glissandi and arpeggio patterns. This is NOT an error.
- **Polyphony up to 8 simultaneous notes** in some pieces (cai_yun_zhui_yue, chun_dao_la_sa, gao_shan_liu_shui) — genuine guzheng technique (strumming/arpeggiated chords).

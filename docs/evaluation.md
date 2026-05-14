# Evaluation Metrics

## MusPy Objective Metrics

- **Paper:** https://cseweb.ucsd.edu/~jmcauley/pdfs/ismir20.pdf
- **Code:** https://github.com/salu133445/muspy

MusPy provides objective metrics for evaluating music generation by comparing statistical differences between training data and generated samples.

### Pitch Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Pitch Range | `pitch_range(music)` | Max pitch − min pitch (excludes drums) |
| Pitches Used | `n_pitches_used(music)` | Count of distinct pitch values |
| Pitch Classes Used | `n_pitch_classes_used(music)` | Count of distinct pitch classes (0-11, ignoring octave) |
| Polyphony | `polyphony(music)` | Avg simultaneous pitches when notes active (NaN if no notes) |
| Polyphony Rate | `polyphony_rate(music, threshold=2)` | Proportion of timesteps with ≥threshold simultaneous pitches |
| Pitch in Scale Rate | `pitch_in_scale_rate(music, root, mode)` | Fraction of notes matching given scale (root 0-11, "major"/"minor") |
| Scale Consistency | `scale_consistency(music)` | Max pitch-in-scale rate across all 24 major/minor scales |
| Pitch Entropy | `pitch_entropy(music)` | Shannon entropy of pitch histogram (128 bins). Higher = more uniform |
| Pitch Class Entropy | `pitch_class_entropy(music)` | Shannon entropy of pitch class histogram (12 bins). Higher = more balanced |

### Rhythm Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Empty Beat Rate | `empty_beat_rate(music)` | Fraction of beats with no notes |
| Drum in Pattern Rate | `drum_in_pattern_rate(music, meter)` | Fraction of drum notes matching pattern ("duple"/"triple") |
| Drum Pattern Consistency | `drum_pattern_consistency(music)` | Max drum-in-pattern rate across duple/triple |
| Groove Consistency | `groove_consistency(music, measure_resolution)` | 1 − mean Hamming distance between consecutive measure onset vectors |

### Other Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Empty Measure Rate | `empty_measure_rate(music, measure_resolution)` | Fraction of measures with no notes |

---

## MIDI-RWKV Evaluation Metrics

### Content Preservation (CP)
Average cosine similarity between moving averages of pitch chroma vectors. Measures style retention. **Higher is better.**

### Groove Similarity (GS)
Average ratio of onset positions that match between corresponding bars. Measures rhythm preservation. **Higher is better.**

### Pitch Class Histogram Entropy Difference (PCHE)
Difference between entropy of pitch frequency vectors. Measures tonality preservation. **Lower is better.**

### F1 Score
Harmonic mean of precision and recall — how well infilled notes match originals. **Higher is better.** Authors note limited value for creative tasks (exact reproduction isn't the goal).

### StyleRank Distribution Metrics
Quantifies stylistic similarity to training data by testing whether generated content matches corpus distribution.

### Attribute Adherence
Extracts musical attributes from generated content and compares against control tokens (note density, duration, polyphony targets).

### Subjective Evaluation
28 participants ranked anonymized clips (original, base model, LoRA, state-tuned) on overall preference. Statistical analysis: Wilcoxon signed rank tests with Holm-Bonferroni correction.

---

## Moonbeam Evaluation

### Generation Metrics
- **Pitch/Velocity Accuracy:** Notes within correct range (±5 bin tolerance)
- **End-time Accuracy:** Mean/std difference between accompaniment and generated end times

### Subjective Evaluation
20 music experts (55% with 11+ years training, 90% with 4+ years) rated pairs on 5-point Likert scale:
1. Chord condition fit
2. Metadata condition fit
3. Coherence
4. Overall enjoyment

### Classification Metrics
Accuracy and F1 (macro) on PiJAMA30, Pianist8, Emopia, GPM30 datasets.

---

## Guzheng-Specific Evaluation (Chen et al. 2022)

### Quantitative
Note accuracy: Delete final N notes and regenerate:
- N=10: 80% accuracy
- N=20: 75% accuracy
- N=25: 68% accuracy

### Qualitative
10 experienced guzheng players scored on 10-point scales for:
- Melody authenticity
- Technique authenticity

---

## Practical Notes for This Project

When evaluating guzheng generation, consider:
1. **Scale consistency** should reflect pentatonic scale (C-D-E-G-A) — standard major/minor may not apply
2. **Pitch range** should match guzheng range (D2–D6, ~3 octaves)
3. **Polyphony** is typically low for guzheng (mostly monophonic/sparse)
4. **Playing techniques** (tremolo, glissando, bends) are not captured by standard MIDI metrics
5. Custom metrics may be needed for evaluating stylistic authenticity

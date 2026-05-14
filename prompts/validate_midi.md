# MIDI Data Validation Prompt

## Context

I am building a guzheng (Chinese zither) music generation system. The pipeline is:

1. **Data Preparation** ← YOU ARE HERE
2. Tokenization (converting to Moonbeam's token format)
3. LoRA fine-tuning on Moonbeam 309M
4. Generation with pitch constraints
5. Evaluation

The `MIDI_transposed/` folder contains N guzheng MIDI files transcribed from sheet music, already transposed into multiple pentatonic keys. These files will be tokenized and fed into Moonbeam for LoRA fine-tuning. Before tokenization, I need to validate the data, assess whether the dataset is sufficient for fine-tuning, and produce a visual analysis report.

## Guzheng Music Domain Knowledge

- Guzheng is a 21-string Chinese zither tuned to the pentatonic scale.
- Standard tuning uses D-pentatonic: D, E, F#, A, B across octaves.
- Valid guzheng range: D2 (MIDI 38) to D6 (MIDI 86).
- Common alternate tuning keys: G, C, F, A (all pentatonic).
- Guzheng music is predominantly single-track, monophonic or with occasional simultaneous notes (chords, arpeggios).
- Characteristic techniques include glissando (rapid ascending/descending runs), tremolo (rapid note repetition), and arpeggios. In MIDI, these appear as rapid note sequences, not as special events.

## Task

Examine every MIDI file in the `MIDI_transposed/` folder. For each file, analyze and report:

### 1. Basic File Integrity
- Confirm it is a valid MIDI file (proper MThd header).
- Report format type (0 or 1), number of tracks, ticks per beat.
- Report total duration in seconds.
- Flag any corrupted or unreadable files.

### 2. Track Structure
- How many tracks contain note data vs. metadata-only tracks?
- Flag any files with more than one note-bearing track (unexpected for guzheng).
- List any program change events — guzheng should be program 0 (Piano) or program 107 (Koto, sometimes used for guzheng). Flag other instruments.

### 3. Pitch Analysis
- Report the full pitch range (min and max MIDI note) per file.
- Flag any notes outside guzheng range D2–D6 (MIDI 38–86).
- Report pitch class distribution per file (count of each note name: C, C#, D, etc.).
- Identify the likely key/scale of each file based on the dominant pitch classes.
- List non-pentatonic tones and their frequency — these are expected but should be documented.
- Flag any pitch that appears only once in a file and is far from the pentatonic scale — these might be transcription errors rather than intentional chromatic tones.

### 4. Timing and Rhythm
- Report tempo (BPM) from tempo events. Flag files with no tempo event.
- Check if note onsets align to a musical grid (quantized to 16th notes or finer). Report the percentage of on-grid vs. off-grid onsets.
- Flag extremely short notes (duration < 20 ticks at 480 tpb) — could be artifacts.
- Flag extremely long gaps between consecutive notes (> 4 beats of silence) — could indicate missing sections.
- Report time signature if present.

### 5. Velocity Analysis
- Report velocity range per file.
- Flag near-silent notes (velocity < 10) — likely artifacts.
- Flag files where all notes have identical velocity (might indicate flat/unexpressive transcription).

### 6. Structural Quality
- Report total note count per file.
- Flag very short files (< 30 notes) — may not contain enough musical content for training.
- Flag very long files (> 2000 notes) — check if these are complete pieces or concatenations.
- Check for duplicate files (identical note sequences).
- Check for near-duplicate files (same piece in different keys — these should be created by transposition in the pipeline, not present in the raw data).

### 7. Consistency Across Dataset
- Report summary statistics across all files: median/mean/min/max for note count, duration, pitch range, tempo.
- Identify outlier files that differ significantly from the rest.
- Report the overall pitch class distribution across the entire dataset.

### 8. Data Sufficiency Assessment

Evaluate whether the dataset is large enough for LoRA fine-tuning on Moonbeam 309M. Report:

- Total number of files, total number of notes across all files, total duration in minutes.
- After chunking into training sequences (context length 2048 tokens, with sliding window overlap), estimate how many training sequences the dataset will produce. Assume roughly 4 tokens per note (time shift, pitch, duration, velocity).
- Estimated tokens-to-parameters ratio. Moonbeam has 309M parameters; LoRA fine-tuning updates ~1-3% of those. Report the ratio of total training tokens to trainable parameters.
- Compare the dataset size to known baselines: Chen et al. 2022 used 31 guzheng MIDI files with LSTM+RL and achieved listenable results. LoRA fine-tuning is more data-efficient than full training, so fewer examples can work, but more is better.
- Provide a clear verdict: is the dataset sufficient, marginal, or insufficient for fine-tuning? If marginal, suggest specific data augmentation strategies (tempo shifting, velocity perturbation, segment shuffling) with estimated impact on effective dataset size.
- If the dataset has transposed copies (same piece in multiple keys), note how many unique pieces vs. transpositions exist, and whether the transpositions add meaningful diversity or are redundant for a model that may already be key-agnostic.

## Output Format

Produce three outputs:

### Output directory

Put **all generated outputs under `metadata/`**.

### A. `metadata/midi_validation_report.md`
A detailed report covering all the above (sections 1-8), organized per-file with a summary section at the end. Use tables where appropriate.

### B. `metadata/midi_issues.csv`
A CSV file with columns: `filename, issue_type, severity, description, suggested_action`

Where:
- `issue_type` is one of: `out_of_range_pitch`, `corrupt_file`, `multi_track`, `wrong_instrument`, `artifact_note`, `near_silent_note`, `no_tempo`, `too_short`, `too_long`, `duplicate`, `flat_velocity`, `suspicious_pitch`, `off_grid_timing`, `long_silence`
- `severity` is one of: `error` (must fix before training), `warning` (should review), `info` (document but likely fine)
- `suggested_action` describes what to do (e.g., "Remove note at tick 5230", "Transpose down one octave", "Review if intentional")

### C. `scripts/midi_analysis_visuals.py`

Write a standalone Python script that reads all MIDI files in `MIDI_transposed/`, generates the following visualizations as PNG files under `metadata/analysis_output/`, and produces a `metadata/dataset_analysis.md` markdown report that embeds/references them. Use `matplotlib` and `mido`.

**Graphs to generate:**

1. **`note_count_distribution.png`** — Histogram of note counts per file. Mark the mean and median. Add a horizontal line or annotation showing the Chen et al. 2022 baseline (31 files).

2. **`duration_distribution.png`** — Histogram of piece durations in seconds. Mark mean and median.

3. **`pitch_class_heatmap.png`** — Heatmap where rows are files (or grouped by detected key) and columns are the 12 pitch classes (C, C#, D, ..., B). Cell color = frequency of that pitch class in the file. This shows at a glance which files are pentatonic and which have chromatic tones.

4. **`pitch_range_per_file.png`** — Horizontal bar chart showing the min-max pitch range for each file. Overlay the valid guzheng range (MIDI 38-86) as a shaded region. Files extending outside the shaded region have out-of-range notes.

5. **`tempo_distribution.png`** — Histogram of tempos across files. Flag outliers.

6. **`velocity_distribution.png`** — Box plot of velocity distributions per file, or a combined violin plot. Highlight files with flat (constant) velocity.

7. **`interval_distribution.png`** — Histogram of melodic intervals (pitch difference between consecutive notes) across the entire dataset. Guzheng music should be predominantly stepwise (intervals of 1-5 semitones) with some octave leaps. Large intervals (>12 semitones) may indicate issues.

8. **`onset_grid_alignment.png`** — Bar chart showing the percentage of on-grid note onsets per file (quantized to 16th notes). Files with low on-grid percentage may have timing issues.

9. **`dataset_sufficiency.png`** — A summary infographic showing: total files, total notes, total duration, estimated training sequences, and a comparison bar against the Chen et al. baseline.

10. **`key_distribution.png`** — Pie chart or bar chart of detected keys across all files.

**The `metadata/dataset_analysis.md` report should include:**

- An overview section summarizing the dataset.
- Each graph embedded with `![description](filename.png)` syntax.
- Brief interpretation of each graph (2-3 sentences explaining what it shows and whether it raises concerns).
- A final "Data Readiness" section with a go/no-go recommendation for proceeding to tokenization, listing any blocking issues and suggested fixes.

## Rules

- Do NOT modify any MIDI files. This is an audit only.
- Use Python with the `mido` and `matplotlib` libraries for MIDI parsing and visualization. Install with `pip install mido matplotlib`.
- If a file has issues that would cause problems during tokenization, clearly state what needs to be fixed and why.
- Process all files in `MIDI_transposed/` including subdirectories.
- Save all PNG graphs and the `dataset_analysis.md` report in `metadata/analysis_output/`.
- The `midi_analysis_visuals.py` script should be self-contained and re-runnable — no hardcoded file lists, it should discover files dynamically from `MIDI_transposed/`.
- Use consistent styling across all graphs (same color palette, font sizes, figure dimensions).
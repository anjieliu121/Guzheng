# Trial 2: Expanded Data + Anti-Repetition for Guzheng Generation

## Goal
Train a small decoder-only transformer on an expanded guzheng MIDI dataset to produce
authentic, non-repetitive pentatonic music. Building on trial 1, this experiment adds
the pittstate_chinese classical repertoire and introduces repetition penalties during
generation.

## Changes from Trial 1
1. **Added data source**: `raw_data_web_scraped/pittstate_chinese/` (20 classical Chinese pieces)
2. **Repetition penalty**: Token-level penalty (default 1.2) during generation
3. **N-gram blocking**: Prevents repeated note-group n-grams (default block size 3)
4. **Self-repetition metrics**: Evaluation now measures internal repetitiveness
5. **Overfitting check**: Now detects excessive self-repetition alongside memorization

## Data Strategy

### Sources
1. **Curated transposed** (`MIDI_transposed/`): 72 files — 18 hand-validated pieces
   transposed to up to 5 pentatonic keys (A, C, D, F, G). 100% pentatonic purity.
2. **Scraped guzheng** (`raw_data_web_scraped/guzheng_tech99/`): 99 MIDI files.
   Quality varies; filtered for pentatonic purity and valid guzheng range.
3. **Pittstate Chinese** (`raw_data_web_scraped/pittstate_chinese/`): 20 classical
   Chinese repertoire pieces (Butterfly Lovers, Gao Shan Liu Shui, etc.).
   Filtered with same criteria as scraped data.

### Cleaning Pipeline (`01_prepare_data.py`)
- Parse every MIDI, extract notes, detect key, compute pentatonic purity
- Reject files with purity < 80% (scraped/pittstate) or < 95% (curated)
- Reject files with fewer than 20 notes (scraped/pittstate) or 10 (curated)
- Reject files with < 50% of notes in guzheng range (MIDI 37-86)
- Copy cleaned files to `data/curated/`, `data/scraped/`, `data/pittstate/`

### Train / Val / Test Split (`02_split_data.py`)
- **Curated**: Split at piece level (all transpositions together). 14 train, 2 val, 2 test.
- **Scraped**: Use existing train/val/test naming (79/10/10).
- **Pittstate**: Random piece-level split. ~16 train, 2 val, 2 test.
- Write split manifest to `data/splits/split.json`

### Data Augmentation (`03_augment_data.py`)
- Tempo jitter: ±15% random tempo scaling
- Velocity humanization: Gaussian noise (σ=8, clipped 1-127)
- Micro-timing: random onset shifts (σ=10 ticks)
- 2 augmented copies per training file → ~3× training data

## Model

### Architecture
Same decoder-only transformer as trial 1:
- `d_model=256`, `n_heads=4`, `n_layers=6`, `d_ff=512`
- Causal attention with learned positional embeddings
- Weight-tied embedding ↔ output head
- Vocab: REMI-style (BOS, KEY, TIME_SHIFT, PITCH, DURATION, VELOCITY, EOS, PAD)

### Regularization
- Dropout: 0.15
- Weight decay: 0.05
- Label smoothing: 0.1
- Early stopping: patience=30 epochs on val loss
- Gradient clipping: 1.0

### Training
- AdamW, lr=3e-4, cosine schedule with warmup (200 steps)
- Batch size: 16, context length: 512, stride: 256
- Max 300 epochs, early stopping

## Generation (`05_generate.py`)
- Load best checkpoint
- Generate with pentatonic pitch masking (constrained decoding)
- **Repetition penalty**: 1.2 (penalize previously generated tokens)
- **N-gram blocking**: block size 3 (prevent repeated 3-note phrases)
- Temperature: 0.9, top_k: 40, top_p: 0.92
- 5 samples per scale (A, C, D, F, G) = 25 constrained
- Optionally generate unconstrained samples for comparison

## Evaluation (`06_evaluate.py`)
- **Distribution metrics**: OA (pitch class, duration, interval, IOI)
- **Purity**: pentatonic adherence per sample
- **Structural**: note density, pitch range, polyphony
- **Self-repetition**: 4/8/12-gram internal repetition rate (NEW)
- **Comparison**: constrained vs unconstrained generation

## Overfitting Check (`07_overfitting_check.py`)
- **N-gram coverage**: what % of generated n-grams exist in training data
- **Longest common substring**: vs training data (plagiarism detection)
- **Self-repetition score**: internal 4/8/12-gram repetition rate (NEW)
- **Longest self-repeat**: longest substring appearing 2+ times within piece (NEW)
- Flags files with possible memorization OR excessive repetition

## Files
```
test_and_trial_2/
├── EXPERIMENT_PLAN.md          ← this file
├── config.py                   ← tokenizer/model/train configs
├── tokenizer.py                ← MIDI tokenizer
├── model.py                    ← transformer model (+ repetition penalty)
├── scales.py                   ← guzheng scale definitions
├── 01_prepare_data.py          ← data cleaning (3 sources)
├── 02_split_data.py            ← train/val/test split
├── 03_augment_data.py          ← data augmentation
├── 04_train.py                 ← training loop
├── 05_generate.py              ← sample generation (+ anti-repetition)
├── 06_evaluate.py              ← evaluation metrics (+ self-repetition)
├── 07_overfitting_check.py     ← plagiarism + repetition detection
├── run_training.sh             ← training launch script
├── data/                       ← processed data
├── checkpoints/                ← model checkpoints
├── generated/                  ← generated MIDI files
├── evaluation/                 ← evaluation reports
├── plots/                      ← loss curves, distributions
└── logs/                       ← training logs
```

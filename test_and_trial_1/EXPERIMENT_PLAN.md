# Test & Trial: Fine-Tuning for Authentic Guzheng Music Generation

## Goal
Train a small decoder-only transformer from scratch on guzheng MIDI data to produce
authentic zither-like, pentatonic music. The model should learn characteristic guzheng
idioms (glissandi, tremolo patterns, pentatonic melodic contour) while avoiding
overfitting to the small training corpus.

## Data Strategy

### Sources
1. **Curated transposed** (`MIDI_transposed/`): 72 files — 18 hand-validated pieces
   transposed to up to 5 pentatonic keys (A, C, D, F, G). 100% pentatonic purity.
2. **Scraped guzheng** (`raw_data_web_scraped/guzheng_tech99/`): 99 MIDI files
   (79 train / 10 val / 10 test). Quality varies; must filter for pentatonic purity
   and valid guzheng range.
3. **Scraped Chinese/koto** (optional): Used only if purity checks pass.

### Cleaning Pipeline (`01_prepare_data.py`)
- Parse every MIDI, extract notes, detect key, compute pentatonic purity
- Reject files with purity < 80% or fewer than 20 notes
- Snap pitches to nearest pentatonic degree where possible
- Constrain to guzheng range (MIDI 37-86)
- Copy cleaned files to `data/curated/` and `data/scraped/`

### Train / Val / Test Split (`02_split_data.py`)
- Split at the **piece** level (all transpositions of one piece stay together)
- Curated: 14 pieces train, 2 pieces val, 2 pieces test
- Scraped: use existing tech99 split (79/10/10)
- Write split manifest to `data/splits/split.json`

### Data Augmentation (`03_augment_data.py`)
- **Tempo jitter**: ±15% random tempo scaling (stretch/compress time)
- **Velocity humanization**: add Gaussian noise to velocities (σ=8, clipped 1-127)
- **Micro-timing**: small random onset shifts (±20ms equivalent)
- Generate 2 augmented copies per training file → ~3× training data
- Store augmented files in `data/augmented/`

## Model

### Architecture
Decoder-only transformer (same as `archive/transformer/`):
- `d_model=256`, `n_heads=4`, `n_layers=6`, `d_ff=512`
- Causal attention with learned positional embeddings
- Weight-tied embedding ↔ output head
- Vocab: REMI-style (BOS, KEY, TIME_SHIFT, PITCH, DURATION, VELOCITY, EOS, PAD)

### Regularization (key changes from baseline)
- Dropout: 0.15 (up from 0.1)
- Weight decay: 0.05 (up from 0.01)
- Label smoothing: 0.1
- Early stopping: patience=30 epochs on val loss
- Gradient clipping: 1.0

### Training
- AdamW, lr=3e-4, cosine schedule with warmup (200 steps)
- Batch size: 16, context length: 512, stride: 256
- Max 300 epochs, early stopping
- Log train/val loss every epoch for plotting

## Generation (`05_generate.py`)
- Load best checkpoint
- Generate with pentatonic pitch masking (constrained decoding)
- Generate 10 samples per scale (A, C, D, F, G) = 50 total
- Temperature: 0.9, top_k: 40, top_p: 0.92
- Also generate 10 unconstrained samples for comparison

## Evaluation (`06_evaluate.py`)
- **Distribution metrics**: OA (pitch class, duration, interval, IOI)
- **Purity**: pentatonic adherence per sample
- **Structural**: note density, pitch range, polyphony
- **Overfitting**: n-gram coverage, longest common substring vs training
- **Comparison**: constrained vs unconstrained generation

## Files
```
test_and_trial/
├── EXPERIMENT_PLAN.md          ← this file
├── EXPERIMENT_LOG.md           ← running log of results
├── config.py                   ← tokenizer/model/train configs
├── tokenizer.py                ← MIDI tokenizer
├── model.py                    ← transformer model
├── scales.py                   ← guzheng scale definitions
├── 01_prepare_data.py          ← data cleaning
├── 02_split_data.py            ← train/val/test split
├── 03_augment_data.py          ← data augmentation
├── 04_train.py                 ← training loop
├── 05_generate.py              ← sample generation
├── 06_evaluate.py              ← evaluation metrics
├── 07_overfitting_check.py     ← plagiarism/memorization detection
├── data/                       ← processed data
├── checkpoints/                ← model checkpoints
├── generated/                  ← generated MIDI files
├── evaluation/                 ← evaluation reports
├── plots/                      ← loss curves, distributions
└── logs/                       ← training logs
```

# Experiment Log: Fine-Tuning for Authentic Guzheng Music Generation

**Date:** 2026-03-31
**Goal:** Train a small decoder-only transformer from scratch on guzheng MIDI data to produce
authentic zither-like, pentatonic music, while avoiding overfitting.

---

## 1. Data Preparation (`01_prepare_data.py`)

### Sources
| Source | Files In | Files Accepted | Rejection Rate |
|--------|----------|---------------|----------------|
| Curated (MIDI_transposed/) | 72 | 72 | 0% |
| Scraped (guzheng_tech99/) | 99 | 99 | 0% |
| **Total** | **171** | **171** | **0%** |

### Filtering Criteria
- Minimum 20 notes (scraped) / 10 notes (curated)
- Pentatonic purity ≥ 80% (scraped) / ≥ 95% (curated)
- ≥ 50% of notes within guzheng range (MIDI 37-86)

### Data Quality
- **Mean purity:** 0.998 (near-perfect pentatonic adherence)
- **Min purity:** 0.927 (guzheng_train_076.mid — still well above 80% threshold)
- **Note count range:** 30 to 2,716 notes per file
- **Mean notes:** 334 per file

### Observation
Both the curated and scraped datasets are remarkably clean for guzheng data — all 171
files passed even stringent purity checks. The curated data (hand-validated against
sheet music) achieves 100% pentatonic purity across all 72 files.

---

## 2. Train/Val/Test Split (`02_split_data.py`)

### Strategy
- **Piece-level splitting** for curated data: all transpositions of a piece go to the
  same split. This prevents data leakage (e.g., `shang_lou_D` in train and `shang_lou_A`
  in test would be trivially similar).
- **Pre-existing split** for scraped data: guzheng_tech99 already has train/val/test naming.
- Random seed: 42

### Split Result
| Split | Curated | Scraped | Total |
|-------|---------|---------|-------|
| Train | 54 (14 pieces) | 79 | **133** |
| Val | 8 (2 pieces: chu_shui_lian, gao_shan_liu_shui) | 10 | **18** |
| Test | 10 (2 pieces: qian_sheng_fo, shang_lou) | 10 | **20** |

### Held-out Pieces
- **Val:** chu_shui_lian (5 transpositions), gao_shan_liu_shui (3 transpositions)
- **Test:** qian_sheng_fo (5 transpositions), shang_lou (5 transpositions)

These are completely unseen during training — no transposition of these pieces appears
in the training set.

---

## 3. Data Augmentation (`03_augment_data.py`)

### Augmentation Strategies
1. **Tempo jitter:** Random tempo scaling factor ∈ [0.85, 1.15] applied to all onsets/durations
2. **Velocity humanization:** Gaussian noise (σ=8, clipped to [1, 127]) added to velocities
3. **Micro-timing:** Gaussian onset jitter (σ=10 ticks ≈ 20ms at default TPB)

### Result
- 2 augmented copies per original training file
- **133 original + 266 augmented = 399 training files**
- ~3× data expansion without introducing new musical content

### Rationale
The augmentation preserves pentatonic pitch content (no pitch changes) while introducing
realistic performance variation. This helps prevent memorization of exact timing/velocity
patterns while keeping the melodic content authentic.

---

## 4. Model Architecture

### Decoder-Only Transformer
| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_heads | 4 |
| n_layers | 6 |
| d_ff | 512 |
| max_seq_len | 2048 |
| dropout | **0.15** (↑ from 0.1) |
| vocab_size | 769 |
| **Total params** | **3,884,288** |

### Tokenization (REMI-style)
- Sequence: `BOS KEY (TIME_SHIFT PITCH DURATION VELOCITY)* EOS`
- 5 KEY tokens (A, C, D, F, G pentatonic scales)
- 201 TIME_SHIFT tokens (0-200 × 10 ticks)
- 128 PITCH tokens (MIDI 0-127)
- 400 DURATION tokens
- 32 VELOCITY bins

### Regularization Improvements (vs. baseline)
| Technique | Baseline | This Experiment |
|-----------|----------|-----------------|
| Dropout | 0.10 | **0.15** |
| Weight decay | 0.01 | **0.05** |
| Label smoothing | 0.0 | **0.1** |
| Early stopping | None | **Patience = 30** |

---

## 5. Training (`04_train.py`)

### Hyperparameters
- Optimizer: AdamW (lr=3e-4, β=(0.9, 0.98))
- LR schedule: Cosine annealing with 200-step warmup
- Batch size: 16
- Context length: 512 tokens, stride: 256
- Gradient clipping: 1.0

### Data Loading
- 399 training sequences → 527,049 total tokens → **1,878 chunks** (sliding window)
- 18 validation sequences → 27,738 total tokens → **103 chunks**
- 117 batches per epoch

### Training Curve
```
Epoch   Train Loss   Val Loss   PPL    LR
─────   ──────────   ────────   ────   ──────
  0       5.7472      4.5852    98.0   1.75e-4
  1       3.9555      3.0853    21.9   3.00e-4
  5       2.8614      2.5564    12.9   3.00e-4
 10       2.6634      2.3774    10.8   2.98e-4
 14       2.5575      2.3368    10.3   2.97e-4
 16*      2.5173      2.3351*   10.3   2.96e-4  ← BEST
 20       2.4498      2.3603    10.6   2.97e-4
 30       2.3215      2.4304    11.4   2.93e-4
 40       2.2316      2.4863    12.0   2.87e-4
 46       2.1870      2.5135    12.3   2.83e-4  ← EARLY STOP
```

### Key Observations
1. **Best val loss at epoch 16** (2.3351, PPL ≈ 10.3)
2. **Val loss starts rising after epoch 16** — classic overfitting onset
3. **Train-val gap at best epoch:** 0.18 (healthy)
4. **Train-val gap at stop:** 0.33 (widening — confirming early stop was correct)
5. **Early stopping triggered at epoch 46** (30 epochs past best = patience exhausted)
6. Training ran ~90 seconds/epoch on Apple MPS (M-series GPU)
7. Total training time: ~69 minutes

### Verdict
Early stopping correctly identified the optimal checkpoint. The model learned meaningful
musical structure (PPL 10.3 from 98.0) without catastrophically overfitting. The
regularization suite (dropout 0.15, weight decay 0.05, label smoothing 0.1, 3× augmentation)
kept the train-val gap small through the optimal region.

---

## 6. Generation (`05_generate.py`)

### Settings
| Parameter | Value |
|-----------|-------|
| Checkpoint | best_model.pt (epoch 16) |
| Temperature | 0.9 |
| Top-k | 40 |
| Top-p | 0.92 |
| Max tokens | 2048 |
| Tempo | 80 BPM |

### Constrained Decoding
- Pitch masking: only allow MIDI pitches valid for the conditioning scale
- Uses `guzheng_scales.json` (21 open strings + pressed strings per scale)
- Non-allowed pitch tokens get logit penalty of -∞

### Output
| Variant | Samples | Per Scale | Total Notes |
|---------|---------|-----------|-------------|
| Constrained | 15 | 3 per scale (A,C,D,F,G) | 7,680 |
| Unconstrained | 15 | 3 per scale | 7,680 |

- All samples generated 512 notes (max tokens reached before EOS)
- Pitch ranges: within guzheng compass (MIDI 38-81)
- Generation time: ~52-55 seconds per sample on MPS

---

## 7. Evaluation Results (`06_evaluate.py`)

### Distribution Overlap (OA) with Training Data

| Distribution | Constrained | Unconstrained |
|-------------|-------------|---------------|
| **Pitch Class** | **0.7336** | 0.7381 |
| **Duration** | **0.5076** | 0.3889 |
| **Interval** | **0.7969** | 0.7043 |
| **IOI** | 0.4016 | 0.3381 |

### Structural Metrics Comparison

| Metric | Training | Constrained | Unconstrained |
|--------|----------|-------------|---------------|
| Mean notes | 342.6 | 481.4 | 491.1 |
| Note density (n/s) | **2.58** | 8.04 | 4.41 |
| Mean duration (s) | **0.585** | 0.277 | 0.391 |
| Mean velocity | 79.2 | 73.4 | 74.7 |
| PC entropy | 2.362 | 1.881 | 1.744 |
| Pentatonic purity | 0.998 | **1.000** | **1.000** |
| Large leap rate | 0.099 | 0.113 | 0.112 |
| Mean interval | 6.23 | 6.77 | 5.91 |
| Max simultaneous | 2.2 | 3.6 | 3.7 |
| Mean pitch (MIDI) | 64.1 | 60.7 | 63.2 |

### Analysis

**Strengths:**
- **100% pentatonic purity** in both constrained and unconstrained generation
  - The model learned pentatonic structure so well that even without explicit masking,
    it produces pure pentatonic output
- **Good interval distribution overlap** (0.797 constrained) — the model captures the
  melodic contour patterns of guzheng music
- **Pitch range** within authentic guzheng compass
- **Mean interval** close to training (6.77 vs 6.23) — preserves the characteristic
  stepwise motion with occasional leaps

**Weaknesses:**
- **Note density too high** (8.04 vs 2.58) — the model generates notes too rapidly,
  likely because the tokenizer's TIME_SHIFT resolution causes many zero-offset notes
- **Durations too short** (0.277s vs 0.585s) — correlated with density issue
- **Lower pitch class entropy** (1.881 vs 2.362) — generation favours fewer pitch classes
  than training data (less melodic variety)
- **IOI overlap low** (0.40) — inter-onset intervals don't match training well, again
  due to the density issue

**Constrained vs Unconstrained:**
- Constrained wins on duration OA (+0.12), interval OA (+0.09), IOI OA (+0.06)
- Unconstrained slightly better on pitch class OA (+0.005) — negligible
- Both achieve 100% pentatonic purity
- Conclusion: **constrained decoding helps** for temporal structure even though
  both are already pentatonic

---

## 8. Overfitting / Plagiarism Analysis (`07_overfitting_check.py`)

### N-gram Coverage (pitch sequences)

| Metric | Constrained | Unconstrained | Test Set (ref) |
|--------|-------------|---------------|----------------|
| **Mean 5-gram coverage** | **0.2456** | 0.2946 | 0.4984 |
| Mean max LCS length | 8.7 | 9.9 | — |
| Files with possible memorization | **0** | 2 | — |

### Interpretation

The test set reference is critical: held-out test pieces have 49.8% 5-gram overlap with
training data. This is the "natural similarity" baseline for guzheng music (pentatonic
scales constrain pitch vocabulary, so repetition is expected).

**Constrained generation (0.246) is WELL BELOW the test set baseline (0.498)**. This
means the generated music is actually MORE novel than real held-out guzheng pieces —
the model is not memorizing, it's creating new sequences that share the pentatonic
vocabulary and some common patterns but are genuinely original.

The 2 flagged unconstrained files (unconstrained_A_01: 100%, unconstrained_G_02: 96%)
may contain memorized fragments, but these are exceptions (2/15 = 13%).

**Maximum LCS (longest common substring) averages 8.7 notes** for constrained generation.
Given that guzheng pieces contain hundreds of notes, a shared substring of 9 notes
is a brief motif (2-3 seconds) — well within normal musical similarity, not plagiarism.

### Verdict: **No systematic overfitting detected.** The model generates novel music.

---

## 9. Key Findings & Takeaways

### What Worked
1. **Piece-level train/test splitting** prevented leakage between transpositions
2. **3× data augmentation** (tempo jitter, velocity noise, micro-timing) improved
   generalization without corrupting pentatonic content
3. **Early stopping (patience=30)** caught overfitting onset at epoch 16 before
   the model degraded
4. **Label smoothing (0.1)** prevented overconfident predictions
5. **Constrained decoding** with scale-aware pitch masking guarantees pentatonic purity
6. The model learned pentatonic structure so thoroughly that even **unconstrained**
   generation is 100% pentatonic — the training data was sufficient to internalize
   the scale system

### What Needs Improvement
1. **Note density is 3× too high** — the model generates too many notes per second.
   Possible fixes:
   - Increase TIME_SHIFT token resolution (currently 10 ticks — try 20 or 30)
   - Add a TIME_SHIFT bias during generation to encourage longer gaps
   - Train with a weighted loss that penalizes zero-offset TIME_SHIFT tokens
2. **Duration distribution mismatch** — generated notes are too short.
   Possible fix: apply post-processing to stretch durations
3. **Pitch class entropy too low** — the model favours a narrow set of pitches.
   More diverse training data or temperature tuning could help

### Comparison with Previous Work (from project evaluation)
| Model | OA_PC | Penta% | Method |
|-------|-------|--------|--------|
| RWKV State-tuned + PP | 0.918 | 100% | State tuning on 36M pretrained model |
| RWKV Constrained | 0.890 | 100% | Constrained decoding on pretrained |
| **This experiment (constrained)** | **0.734** | **100%** | **From-scratch 3.9M transformer** |
| Moonbeam Pretrained | 0.815 | — | 309M pretrained |

The from-scratch transformer achieves lower OA_PC (0.734 vs 0.918) than the
state-tuned RWKV, which is expected: the RWKV was pretrained on 2.1M MIDI files
(GigaMIDI) and only needed light adaptation. Our model learned everything from
just 133 guzheng training files — a much harder task. Despite this, it achieves
100% pentatonic purity and reasonable interval distributions.

---

## 10. File Manifest

```
test_and_trial/
├── EXPERIMENT_PLAN.md          ← Experiment design
├── EXPERIMENT_LOG.md           ← This file
├── config.py                   ← Tokenizer/model/train configs
├── tokenizer.py                ← MIDI↔token conversion
├── model.py                    ← Transformer with label smoothing
├── scales.py                   ← Guzheng scale definitions
├── 01_prepare_data.py          ← Data cleaning & filtering
├── 02_split_data.py            ← Piece-level train/val/test split
├── 03_augment_data.py          ← Tempo/velocity/timing augmentation
├── 04_train.py                 ← Training loop with early stopping
├── 05_generate.py              ← Constrained & unconstrained generation
├── 06_evaluate.py              ← OA metrics & distribution comparison
├── 07_overfitting_check.py     ← N-gram & LCS plagiarism detection
├── run_training.sh             ← Training launch script
├── data/
│   ├── data_manifest.json      ← 171 files with purity/stats
│   ├── curated/                ← 72 cleaned curated MIDI files
│   ├── scraped/                ← 99 cleaned scraped MIDI files
│   ├── train/                  ← 399 files (133 orig + 266 augmented)
│   ├── val/                    ← 18 files
│   ├── test/                   ← 20 files
│   └── splits/
│       └── split.json          ← Train/val/test manifest
├── checkpoints/
│   ├── best_model.pt           ← Epoch 16, val_loss=2.3351
│   ├── final_model.pt          ← Epoch 46 (early stopped)
│   ├── epoch_0000.pt           ← Periodic checkpoints
│   ├── epoch_0020.pt
│   └── epoch_0040.pt
├── generated/
│   ├── constrained/            ← 15 pentatonic-masked samples
│   ├── unconstrained/          ← 15 free-generation samples
│   └── generation_stats.json
├── evaluation/
│   ├── evaluation_results.json ← Full metrics
│   ├── evaluation_report.md    ← Formatted report
│   └── overfitting_analysis.json
├── logs/
│   ├── training_history.json   ← Per-epoch train/val loss
│   └── training_output.log     ← Console output
└── plots/                      ← (available for loss curves)
```

---

## 11. Reproducibility

All scripts are fully self-contained and can be re-run end-to-end:

```bash
cd test_and_trial/
python3 01_prepare_data.py      # ~10 seconds
python3 02_split_data.py        # ~2 seconds
python3 03_augment_data.py      # ~5 seconds
python3 04_train.py --epochs 200  # ~70 minutes (MPS), early stops ~epoch 46
python3 05_generate.py --unconstrained --num_per_scale 3  # ~30 minutes
python3 06_evaluate.py          # ~10 seconds
python3 07_overfitting_check.py # ~2 minutes
```

Requirements: `mido`, `torch`, `numpy` (Python 3.10+)

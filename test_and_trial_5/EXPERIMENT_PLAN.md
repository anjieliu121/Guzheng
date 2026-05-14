# Trial 5: MIDI-RWKV State-Tuning on Full MIDI_transposed Dataset

## Problem Statement

Previous trials used a mixed dataset of 147 files from three sources (curated, web-scraped, PittState), with augmentation bringing the total to ~441. Results were mixed:
- **Trial 3** (147 files, no augmentation): Short pieces (~49 notes), memorization via training-file prompts (LCS 97.3%)
- **Trial 4** (441 files with augmentation): Fixed memorization via prompt strategy, but still trained on heterogeneous data of varying quality

Meanwhile, the **original best model** (72 curated files, OA_PC 0.918) remains the quality benchmark. Its success suggests that **data quality matters more than quantity** for state tuning.

Since Trial 4, the curated `MIDI_transposed/` dataset has grown from 72 to **590 files** (125 unique pieces × up to 5 pentatonic keys). All files are hand-validated with 100% pentatonic purity (276 files cleaned pre-training: 10,608 non-pentatonic notes snapped to nearest valid pitch). This is an 8× increase over the original best model's dataset — enough data to train without augmentation noise, and all consistently high quality.

## Goals

1. **Generate authentic pentatonic guzheng music** (OA_PC > 0.90, matching original best)
2. **No overfitting** (LCS ratio < 0.15 against training set)
3. **No unreasonable repetition** (self-repetition 4-gram: 0.3–0.6, matching training data)
4. **Longer pieces** (target: 200+ notes)
5. **100% pentatonic purity** (after post-processing)

## Key Changes from Trial 4

| Aspect | Trial 4 | Trial 5 |
|--------|---------|---------|
| **Data source** | 147 files (3 sources: curated + scraped + PittState) | 590 files (1 source: MIDI_transposed/, all hand-validated) |
| **Training files** | 441 (147 original + 294 augmented) | 196 (2 transpositions per piece, scale-balanced) |
| **Data augmentation** | 2× augmented copies | None — avoids augmentation artifacts |
| **Data quality** | Mixed (some scraped files are lower quality) | Uniform high quality, 100% pentatonic purity (cleaned pre-training) |
| **Split strategy** | Inherited from Trial 3 (pre-split) | By piece name (prevents transposition leakage) |
| **Training epochs** | 24 planned (6 completed) | 8 (fixed; post-hoc checkpoint selection) |
| **LR schedule** | Flat (2e-2 entire run) | Cosine decay (2e-2 → 1e-3) |
| **Epoch saves** | Every 2 epochs | Every 2 epochs (4 checkpoints) |
| **Anti-repetition** | Temperature + top_p only | Temperature + top_p + pitch 8-gram repetition penalty |
| **Max generation tokens** | 768 | 1024 (closer to training avg) |

## Data Preparation

### Source
`MIDI_transposed/` — 590 MIDI files, 125 unique pieces, each transposed to 1–5 pentatonic keys (D, G, C, A, F).

### Pre-Training Cleaning
Before splitting, all 590 files were validated and cleaned:
- `check_midi_note_quality.py --apply`: chromatic variant correction, off-grid snapping
- Custom pentatonic snap: 276 files had 10,608 non-pentatonic notes snapped to nearest valid pitch (including pressed notes per scale)
- Post-cleaning verification: 0 non-pentatonic notes remain across all 590 files

### Train/Val/Test Split

**Split by piece name** (all transpositions of one piece go to the same split):

| Split | Pieces | Files | Purpose |
|-------|--------|-------|---------|
| Train | 100 (80%) | 196 (2 transpositions/piece, scale-balanced) | Model training |
| Val | 12 (10%) | 60 (all transpositions) | Checkpoint selection |
| Test | 13 (10%) | 59 (all transpositions) | Final evaluation (never used for selection) |

Training scale distribution: A=40, C=39, D=39, F=39, G=39 (balanced by greedy selection).

Val pieces: chu_shui_lian, fang_zhi_mang, guzheng_train_003, guzheng_train_005, guzheng_train_030, guzheng_train_040, guzheng_train_047, guzheng_train_051, guzheng_train_053, guzheng_validation_004, guzheng_validation_008, zai_bei_jing_de_jin_shan_shang

Test pieces: cai_yun_zhui_yue, gao_shan_liu_shui, guzheng_test_000, guzheng_test_003, guzheng_train_004, guzheng_train_007, guzheng_train_011, guzheng_train_045, guzheng_train_057, guzheng_train_062, guzheng_train_070, nan_zheng_gong, yu_zhou_chang_wan

### Transposition Subsampling

Rather than using all 5 transpositions per piece (~471 files), we select 2 per piece (~196 files):
- Reduces training time from ~3.2 days to ~16 hours
- Still 2.7× more data than the original best model (72 files)
- Scale-balanced selection ensures even representation across all 5 keys
- Intervals and rhythms are identical across transpositions — 2 is enough to teach the model each scale's pitch range
- Val/test retain all transpositions for thorough evaluation

### No Augmentation

With 196 training files (vs 72 in the original best model), augmentation is unnecessary. Removing it:
- Eliminates micro-timing noise that could blur rhythmic patterns
- Keeps velocity and tempo faithful to the curated data
- Simplifies the pipeline

## Training Configuration

```
Base model:       RWKV-7 (36M params, pretrained on GigaMIDI 2.1M files)
PEFT method:      State tuning (~294K trainable params, 0.8%)
Training data:    196 files (2 transpositions/piece, scale-balanced)
Epochs:           8 (fixed; best checkpoint selected post-hoc by generation quality)
LR schedule:      Cosine decay (2e-2 → 1e-3)
Warmup:           20 steps
Context length:   2048
Batch size:       1
Checkpoint saves: every 2 epochs (4 checkpoints total)
Accelerator:      MPS (Apple Silicon)
Precision:        bf16
Optimizer:        Adam (beta1=0.9, beta2=0.99)
```

### Why Fixed Epochs + Post-Hoc Selection (Not Early Stopping)

RWKV-PEFT has no native validation loss monitoring — the val dataloader is commented out in `train.py`. Rather than modifying the framework, we:
1. Train for 8 epochs (saves every 2 → 4 checkpoints)
2. Generate samples from each checkpoint
3. Evaluate OA metrics + memorization + repetition per checkpoint
4. Select the best checkpoint based on combined metrics

This is the same approach that worked in Trials 3/4. The cosine LR schedule provides implicit regularization that fixed LR did not.

### Why Cosine LR Decay

Prior trials used flat LR (2e-2 entire run). RWKV-PEFT supports cosine decay natively (`--lr_schedule cos`). Setting `lr_init=2e-2, lr_final=1e-3`:
- Early epochs explore at full LR
- Later epochs settle with 5× lower LR, reducing instability
- No code changes needed — just CLI flags

## Generation Configuration

```
Prompt sources:   Val set (16 tokens) + Test set (16 tokens) + Synthetic (BOS only)
Temperature:      0.85
Top-p:            0.9
Repetition penalty: Pitch 8-gram penalty (value TBD by prototype; candidates: 1.10, 1.15, 1.20)
Max new tokens:   1024
Min tokens before EOS: 200
Constraint:       Unconstrained generation → post-processing
Samples:          15 per checkpoint (5 val + 5 test + 5 synthetic)
```

### Repetition Penalty

Trial 4 relied solely on temperature/top_p for diversity. This trial adds a **pitch 8-gram repetition penalty** during generation:
- Track pitch 8-grams already generated in the current piece
- When a token would complete a repeated 8-gram, reduce its logit by `log(penalty)`
- The exact penalty value will be determined by `prototype_rep_penalty.py` before training begins
- Target: self-repetition 4-gram between 0.3–0.6 (matching training data distribution)

## Prompt Strategy

Same three-category approach as Trial 4 (proven effective against memorization):

### A. Validation set prompts (10 samples)
- Source: val split files (never seen during training)
- First 16 tokens only
- Purpose: generalization quality

### B. Test set prompts (10 samples)
- Source: test split files (never seen during training or checkpoint selection)
- First 16 tokens only
- Purpose: independent evaluation

### C. Synthetic prompts (5 samples)
- BOS token only (no MIDI content)
- One per scale (A, C, D, F, G)
- Purpose: strongest memorization test — no priming at all

## Post-Processing Pipeline

Same proven pipeline from Trial 4:
1. **Merge tracks** → single instrument
2. **Pentatonic snap** → nearest valid pentatonic pitch (including pressed notes)
3. **Range clamp** → MIDI 38–86 (guzheng compass)
4. **Polyphony limit** → max 4 simultaneous notes

## Pipeline

```
01_prepare_data.py          → Split MIDI_transposed/ into train/val/test by piece
02_train.sh                 → MIDI-RWKV state-tuning (16 epochs, cosine LR)
03_generate.py              → Generate from val/test/synthetic prompts (per checkpoint)
04_postprocess.py           → Pentatonic snap + range + polyphony + single track
05_evaluate.py              → OA metrics, per-category analysis
06_overfitting_check.py     → LCS + n-gram memorization detection
prototype_rep_penalty.py    → Pre-training ablation of repetition penalty values
```

## Evaluation Metrics

### Quality metrics (vs training distribution)
| Metric | Target | Rationale |
|--------|--------|-----------|
| OA pitch class | > 0.90 | Match original best (0.918) |
| OA duration | > 0.80 | Match original best (0.839) |
| OA interval | > 0.65 | Original best was 0.641 |
| OA IOI | > 0.65 | Original best was 0.690 |
| Pentatonic purity | 100% | After post-processing |
| Note density | 3.0–4.0 n/s | Training mean is 3.53 |
| Note count | > 200 | Longer than original best's 178 |
| Pitch range | MIDI 38–86 | Full guzheng compass |

### Anti-overfitting metrics (per prompt category)
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| LCS ratio (vs nearest training file) | < 0.15 | Stricter than Trial 4's 0.20; original best was 0.076 |
| Pitch 5-gram coverage | < 0.45 | Below test set baseline (~0.488) |
| Pitch 8-gram coverage | < 0.15 | Low overlap for longer patterns |
| Pitch 12-gram coverage | < 0.05 | Near-zero for long patterns |

### Anti-repetition metrics
| Metric | Target Range | Rationale |
|--------|-------------|-----------|
| Self-repetition 4-gram | 0.30–0.60 | Training mean ~0.558; too low = unnatural, too high = looping |
| Self-repetition 8-gram | 0.10–0.35 | Training mean ~0.300 |
| Self-repetition 12-gram | < 0.15 | Minimal exact long-phrase repetition |

## Comparison Targets

| Metric | Original Best (72 files) | Trial 3 Best | Trial 5 Target |
|--------|--------------------------|-------------|----------------|
| OA pitch class | 0.918 | 0.797 | > 0.90 |
| OA duration | 0.839 | 0.642 | > 0.80 |
| Note count | 178 | 49 | > 200 |
| Density (n/s) | 3.45 | 2.11 | 3.0–4.0 |
| LCS ratio | 0.076 | 0.408 | < 0.15 |
| Pentatonic purity | 100% | 97.9% | 100% |
| Self-rep 4-gram | — | 0.177–0.208 | 0.30–0.60 |

## Hypothesis

With 2.7× more high-quality curated data than the original best model (196 vs 72 training files), state tuning should:
1. **Improve generalization** — more diverse pieces reduce memorization risk
2. **Maintain or exceed quality** — all data is hand-validated with consistent formatting
3. **Produce natural repetition patterns** — the 8-gram penalty prevents degenerate loops while the larger dataset provides richer melodic vocabulary
4. **Benefit from cosine LR** — decaying from 2e-2 to 1e-3 stabilizes later epochs

The key bet is that **curated data quality + volume** eliminates the need for data augmentation, mixed sources, or aggressive regularization.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| State tuning saturates with more data (flat loss) | Monitor training loss trajectory; cosine LR gives later epochs a lower LR to refine |
| Transposition leakage inflates metrics | Split by piece name, not by file |
| 8-gram penalty is too aggressive/weak | Prototype with existing model before training (prototype_rep_penalty.py) |
| MPS training is slow (~37s/step) | 196 files × 8 epochs = 1,568 steps ≈ 16 hours |
| New val/test files differ from Trial 3's | Report metrics on both new splits and Trial 3 val/test for comparability |

## Training Time Estimate

| Step | Estimate |
|------|----------|
| 01 — Data prep | < 1 min |
| 02 — Training (196 files × 8 epochs @ ~37s/step) | ~16 hours |
| 03 — Generation (15 samples × 4 checkpoints) | ~1.5 hours |
| 04 — Post-processing | < 1 min |
| 05 — Evaluation | ~10 min |
| 06 — Overfitting check | ~20 min |
| **Total** | **~18 hours** |

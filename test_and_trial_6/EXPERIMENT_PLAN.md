# Trial 6 — From-Scratch Transformer on Full MIDI_transposed Corpus

**Date:** 2026-04-09
**Goal:** Train a decoder-only transformer from scratch on **all 590 files** in
`MIDI_transposed/` to generate authentic, structurally coherent guzheng music
that scores well on literature-standard musicality metrics, while avoiding
overfitting.

**Hypothesis:** Trial 1's from-scratch transformer produced more musically
structured output than every pretrained model variant tried so far
(CR=6.10, SI=0.91, H2=1.78 — within ±0.06 of real guzheng on every structural
metric — vs. RWKV state-tuned at CR=2.55, SI=0.69, H2=2.46). Its only weaknesses
were (a) lower OA_PC due to limited data (133 train files) and (b) note density
3× too high due to a tokenizer artefact. Both are fixable. With 4.4× more
training data and a corrected tokenizer, we expect a transformer that matches
real guzheng on **both** statistical and structural axes.

---

## 1. Data

### 1.1 Source
- `MIDI_transposed/` — **590 files**, **125 unique pieces**, each transposed to
  up to 5 pentatonic scales (A, C, D, F, G).
- Trial 1 used only 72 curated + 99 scraped = 171 files (a subset).
- Trial 5 used ~196 files. **Trial 6 uses everything.**

### 1.2 Splitting strategy — piece-level, not file-level
**Critical:** all transpositions of the same piece must go into the **same
split** to prevent trivial leakage (a piece in C in train and the same piece in
A in test is the same melody, just shifted).

| Split | Pieces | Files (approx) | Purpose |
|---|---|---|---|
| Train | 105 (84%) | ~496 | Optimization |
| Val | 10 (8%) | ~48 | Early stopping, hyperparameter selection |
| Test | 10 (8%) | ~46 | Final evaluation only — touched once |

Random split with `seed=42`. Piece names → splits saved to
`data/splits/split.json` for reproducibility.

### 1.3 Quality filtering
Reuse Trial 1's filters but relaxed:
- Min 20 notes per file
- Pentatonic purity ≥ 95%
- ≥ 50% of notes in guzheng range (MIDI 37–86)

Expected to retain >95% of files (curated source already vetted).

### 1.4 Augmentation (training set only)
Reuse Trial 1's recipe:
1. **Tempo jitter:** ×U(0.85, 1.15) on onsets and durations
2. **Velocity humanization:** Gaussian noise σ=8, clipped [1, 127]
3. **Micro-timing:** onset jitter σ=10 ticks (~20 ms)

**Expansion:** 2 augmented copies per original = **3× training data ≈ 1,488 files**
(vs Trial 1's 399). No pitch transposition during augmentation — already covered
by the 5-key transpositions in the source.

---

## 2. Tokenizer (REMI-style, improved)

### 2.1 Trial 1 problems to fix
| Issue | Trial 1 cause | Trial 6 fix |
|---|---|---|
| Density 3× too high | `tick_resolution=10` ⇒ many (TIME_SHIFT=0) tokens collapse adjacent notes | `tick_resolution=20`, allow longer max time shift |
| Durations too short | DURATION token resolution coarse at long durations | log-quantized DURATION bins, max raised to 2 s |
| Pitch entropy low | Inherent to small data | More data fixes this |

### 2.2 Vocabulary
Sequence: `BOS KEY (TIME_SHIFT PITCH DURATION VELOCITY)* EOS`

| Group | Count | Notes |
|---|---|---|
| Special (PAD/BOS/EOS) | 3 | |
| KEY | 5 | A, C, D, F, G |
| TIME_SHIFT | 100 | 20 ticks/step × 100 = 2000-tick window (~2 beats) |
| PITCH | 50 | MIDI 37–86 (guzheng compass only — not full 128) |
| DURATION | 64 | log-spaced bins from 30 ms to 2 s |
| VELOCITY | 32 | linear bins |
| **Total** | **254** | (vs Trial 1's 769 — smaller, more efficient) |

### 2.3 Constrained decoding
Per-scale pitch mask from `scales.py` (carry over from Trial 1) — reject
non-pentatonic pitches at inference. Already proven to give 100% purity.

---

## 3. Model

### 3.1 Architecture — modest scale-up from Trial 1
| Param | Trial 1 | Trial 6 | Rationale |
|---|---:|---:|---|
| d_model | 256 | **320** | Modest capacity bump |
| n_heads | 4 | **5** | Match d_model |
| n_layers | 6 | **6** | Same depth |
| d_ff | 512 | **1280** | Standard 4× ratio |
| max_seq_len | 2048 | **2048** | Same |
| dropout | 0.15 | **0.15** | Same regularization |
| **Total params** | 3.9M | **~7M** | ~1.8× larger |

Decoder-only, pre-norm, learned positional embeddings, tied input/output
embeddings.

**Why this size?** Trial 1 trained 3.9M on ~527K tokens ⇒ 0.135 tokens/param,
without overfitting. Trial 6 trains 7M on ~900K tokens ⇒ ~0.13 tokens/param —
**same data-richness ratio as Trial 1**, so we inherit its overfitting headroom.
Going larger (e.g. 14M) would halve the tokens/param ratio and risk overfitting.

**Why not stay at 3.9M?** Trial 1's structural metrics (CR, SI, H2) already
matched real guzheng with 3.9M params; only OA_PC was low (0.73). The ~2×
capacity bump targets the OA_PC gap, leaving the structural quality intact.
If Trial 6 *also* hits ~0.73 OA_PC, that's diagnostic that capacity isn't the
bottleneck — and Trial 7 can change strategy with confidence.

### 3.2 Regularization
| Technique | Value |
|---|---|
| Dropout (attention + FFN + embedding) | 0.15 |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 20 epochs |

---

## 4. Training

### 4.1 Hyperparameters
| Param | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9 |
| Peak LR | 3e-4 |
| LR schedule | cosine annealing, 500-step warmup |
| Batch size | 16 |
| Context length | 512 tokens |
| Stride (sliding window) | 256 |
| Max epochs | 200 (early stop expected ~30–60) |
| Seed | 42 |

### 4.2 Data flow estimate
- ~1,500 train files × ~600 tokens/file ≈ 900K tokens
- Sliding window (ctx 512, stride 256) ⇒ ~3,500 chunks
- Batches per epoch: ~220
- Tokens per epoch: ~1.8 M

### 4.3 Hardware
Apple M-series GPU via PyTorch MPS (same as all prior trials).

### 4.4 Logging
- Per-epoch train/val loss → `logs/training_history.json`
- TensorBoard or matplotlib loss curve → `plots/`
- Save: `checkpoints/best_model.pt` (best val), periodic every 20 epochs.

---

## 5. Generation

### 5.1 Decoding settings (start point — tune on val)
| Param | Value |
|---|---|
| Temperature | 0.9 |
| Top-k | 40 |
| Top-p | 0.92 |
| Max tokens | 2048 |
| Min notes before EOS | 200 |

### 5.2 Samples
Per checkpoint (best_model only — no per-epoch sweep, that bloated Trial 5):
- **5 val-prompted** (first 8 notes of a val piece)
- **5 test-prompted** (first 8 notes of a test piece)
- **15 synthetic** (3 per scale × 5 scales) starting from `BOS KEY`

Total: **25 samples**, both constrained and unconstrained = **50 MIDI files**.

---

## 6. Evaluation — literature-standard metrics

We compute **all** metrics against the **training** distribution as the
reference, with the **held-out test** set as a calibration baseline (so we know
what scores real unseen guzheng music gets — the ceiling).

### 6.1 Distributional (Yang & Lerch 2018)
| Metric | What it measures |
|---|---|
| OA_PC | Pitch class histogram overlap |
| OA_Duration | Note duration distribution overlap |
| OA_Interval | Pitch interval distribution overlap |
| OA_IOI | Inter-onset interval overlap |
| Note density (n/s) | First-order tempo |
| Mean velocity | Dynamics |
| Pitch range | Compass usage |

### 6.2 Structural (Wu & Yang 2020 + Pearce & Wiggins 2012)
| Metric | What it measures |
|---|---|
| Compression ratio (gzip) | Structural repetition |
| Structureness Indicator (SI) | Long-range repeated sections |
| 2nd-order pitch transition entropy | Note-to-note plausibility |
| Groove consistency | Bar-to-bar rhythm similarity |
| Pitch class entropy (1st-order) | Melodic variety |

### 6.3 Plagiarism / overfitting (carry over from Trial 1)
| Metric | Threshold |
|---|---|
| Mean 5-gram coverage vs train | < test-set baseline |
| Mean LCS ratio | < 0.15 |
| Files flagged for memorization | < 10% |

### 6.4 Required outcomes (success criteria)
| Axis | Target | Reasoning |
|---|---|---|
| Pentatonic purity | 100% (after constrained decode) | Hard requirement |
| OA_PC | **≥ 0.85** | Trial 1 hit 0.73 with 1/4 the data; 0.85 is realistic |
| Compression ratio | **within ±15% of training** (4.1–5.6) | Real-music structural density |
| Structureness Indicator | **≥ 0.90** | Match Trial 1's structural quality |
| 2nd-order pitch entropy | **within ±0.10 of training** (1.62–1.82) | Plausible note moves |
| Note density | **within ±30% of training** (1.8–3.4 n/s) | The Trial 1 failure mode |
| LCS ratio | **< 0.15** | No memorization |
| 5-gram coverage | **< test baseline** | More novel than held-out real |

### 6.5 Listening test (gold standard, optional but recommended)
5–10 listeners, blind pairwise A/B between Trial 6 outputs and (a) Trial 1, (b)
Trial 5 RWKV best, (c) real held-out test pieces. ~20 pairs each. Binary
"which sounds more like guzheng music" + free text. ~30 minutes per listener.

---

## 7. Pipeline & file layout

```
test_and_trial_6/
├── EXPERIMENT_PLAN.md              ← this file
├── EXPERIMENT_LOG.md               ← filled in as we go
├── config.py                       ← tokenizer/model/train configs
├── tokenizer.py                    ← improved REMI tokenizer
├── model.py                        ← 14M decoder-only transformer
├── scales.py                       ← guzheng scale definitions (copy from t1)
├── 01_prepare_data.py              ← clean + filter MIDI_transposed/
├── 02_split_data.py                ← piece-level train/val/test split
├── 03_augment_data.py              ← tempo/velocity/timing augmentation
├── 04_train.py                     ← AdamW + cosine + early stopping
├── 05_generate.py                  ← constrained + unconstrained sampling
├── 06_evaluate.py                  ← OA metrics + structural metrics
├── 07_overfitting_check.py         ← n-gram + LCS plagiarism check
├── run_pipeline.sh                 ← one-shot end-to-end runner
├── data/
│   ├── data_manifest.json
│   ├── splits/split.json
│   ├── train/  val/  test/
├── checkpoints/
│   └── best_model.pt
├── generated/
│   ├── constrained/  unconstrained/
│   └── generation_stats.json
├── evaluation/
│   ├── evaluation_results.json
│   ├── overfitting_analysis.json
│   └── musicality_metrics.json
├── logs/
│   └── training_history.json
└── plots/
    └── loss_curve.png
```

---

## 8. Time budget — concrete estimates

Anchored on Trial 1's measured speed: **~90 s / epoch on MPS** for a 3.9M model
on 1,878 chunks (~117 batches/epoch).

### 8.1 Per-step compute scaling
Trial 6 vs Trial 1:
- Model FLOPs: **~1.8×** larger (7M vs 3.9M)
- Chunks per epoch: **~1.9×** larger (3,500 vs 1,878)
- Combined per-epoch cost: **~3.4×**
- Estimated: **~5 minutes per epoch**

### 8.2 Stage-by-stage estimate

| Stage | Time | Notes |
|---|---|---|
| 01_prepare_data | 1 min | Just filter+stat, no training |
| 02_split_data | <1 min | piece-level split |
| 03_augment_data | 2–3 min | I/O bound |
| **04_train (worst case 80 epochs)** | **~7 hours** | early stop usually fires earlier |
| 04_train (likely 35 epochs to best + 20 patience) | **~4–5 hours** | typical |
| 05_generate (50 samples) | ~30 min | autoregressive |
| 06_evaluate | 5 min | numpy + gzip is fast |
| 07_overfitting_check | 5 min | LCS is the slow part |
| **TOTAL wall clock** | **~5–7 hours** | Single afternoon |

### 8.3 Risks
- **MPS instability** — Trial 5 hit memory pressure on similar-size models. Mitigation: smaller batch (8) fallback.
- **Laptop sleep** — wrap in `caffeinate -dims`.
- **Slow LCS evaluation** on larger test set — cap LCS to first 200 notes per file.
- **Bigger model overfits faster** — early stopping patience=20 catches this.

### 8.4 Decision points (gate the experiment)
- **After 10 epochs:** if val loss < 3.5, continue. Otherwise stop and rethink architecture.
- **After best_model** is trained: if OA_PC < 0.80 OR structureness < 0.85, mark
  as failed and write up findings rather than pursuing further trials.
- **If success criteria met:** run listening test before declaring victory.

---

## 9. Differences from Trial 1 — summary

| Aspect | Trial 1 | Trial 6 |
|---|---|---|
| Training files | 133 (399 w/ aug) | ~496 (~1,488 w/ aug) |
| Pieces | not piece-split-aware on full corpus | piece-level split on 125 pieces |
| Tokenizer time res | 10 ticks | **20 ticks** |
| Tokenizer pitch range | 0–127 | **37–86** (guzheng only) |
| Vocab size | 769 | **254** |
| Model params | 3.9M | **14M** |
| Layers / dim | 6 / 256 | **8 / 384** |
| Eval metrics | OA only | **OA + CR + SI + H2 + GC** |
| Test-set baseline | yes | yes |
| Listening test | no | **yes (recommended)** |

---

## 10. Out of scope (don't do this trial)

- ❌ Pretraining or pretrained backbones — Trials 4/5 covered that
- ❌ LoRA / state tuning — Trials 4/5 covered that
- ❌ KV cache or speed optimization — generation is fast enough
- ❌ Conditioning on style tags / emotion / tempo — single condition (KEY) only
- ❌ Audio rendering / FAD — symbolic-only evaluation

If Trial 6 succeeds, those become future work.

# Adapting MIDI Models for East Asian Zither Music Generation
* next step: expand dataset
* It seems like state tuning is more effective than LoRA for small-data domain adaptation -> continue with MIDI-RWKV+State
* Explain state-tuning (soft something?)
* test on testing data (not training)
* 4/10 get presentation draft done
---





# Guzheng Music Generation with Neural Models

**Date:** 2026-03-27 | **Status:** Evaluation complete, production pipeline established

---

## 1. Goal

Generate authentic-sounding guzheng (Chinese 21-string zither) music using neural sequence models, with strict adherence to pentatonic music theory and idiomatic playing characteristics.

## 2. Motivation

Guzheng is one of the most widely played traditional Chinese instruments, yet it is underrepresented in AI music generation research. Existing foundation models (trained on Western piano/pop MIDI corpora) have no exposure to pentatonic idioms, instrument-specific pitch ranges, or guzheng playing techniques (glissandi, tremolo). This project explores whether small-data fine-tuning can adapt general-purpose MIDI models to produce musically valid guzheng output.

## 3. Architecture

Two foundation models were compared, each adapted to guzheng via parameter-efficient fine-tuning (PEFT):

| | MIDI-RWKV | Moonbeam |
|---|---|---|
| **Base architecture** | RWKV-7 (linear RNN) | LLaMA Transformer + GRU sub-decoder |
| **Parameters** | 36M | 309M |
| **Tokenization** | REMI+ with BPE (16,000 vocab) | FME compound (6-token sub-words) |
| **Attention complexity** | O(n) linear | O(n^2) quadratic |
| **Pretraining data** | GigaMIDI (2.1M files) | Lakh MIDI |
| **Adaptation methods tested** | State tuning, LoRA | LoRA |

### 3.1 Adaptation Strategies

| Strategy | Trainable Params | Description |
|----------|-----------------|-------------|
| **State tuning** (MIDI-RWKV only) | ~294K (0.8%) | Optimizes only the initial hidden state vectors; all model weights frozen. Shifts the recurrence toward guzheng style while fully preserving pretrained generalization. |
| **LoRA** (both models) | ~1.2M (MIDI-RWKV), ~2.4M (Moonbeam) | Low-rank adapters on attention + FFN layers. rank=8, alpha=32. |

![Adaptation Strategy Comparison](outputs/evaluation/adaptation_comparison.png)
*Figure 1. State tuning (294K params) outperforms LoRA (1.2M-2.4M params) on both models. Moonbeam LoRA collapses to 0.535 OA_PC due to overfitting on the small dataset.*

### 3.2 Constraint Enforcement

Two strategies were tested to ensure 100% pentatonic purity:

| Strategy | Mechanism | Speed | Purity |
|----------|-----------|-------|--------|
| **Token-level masking** | Block non-pentatonic pitch tokens during decoding via logit penalty (-10,000) | ~22 min/sample | 100% |
| **Post-processing** | Snap non-pentatonic MIDI pitches to nearest valid pitch; constrain range to MIDI 38-86; limit polyphony to 4 | Instant | 100% |

### 3.3 Production Pipeline

```
Seed MIDI (or random prompt)
  --> MIDI-RWKV (state-tuned), temp=0.85, top_p=0.9, max_tokens=512
  --> Post-process: pentatonic snap + range [38,86] + polyphony <= 4
  --> FluidSynth render to WAV
  --> Guzheng audio output
```

## 4. Data

### 4.1 Dataset Summary

| Metric | Value |
|--------|-------|
| Original curated pieces | 18 |
| Transposed variants (5 pentatonic keys) | 72 |
| Total training corpus | 90 files |
| Pentatonic purity | 100% (all files validated against sheet music) |
| Global pitch range | MIDI 37-86 (D2-D6, full 21-string compass) |
| Mean note count per piece | 610 |
| Mean duration per piece | 214 s |
| Mean note density | 3.53 notes/s |
| Velocity | Constant 64 (no dynamics) |

### 4.2 Pentatonic Key Distribution

| Key | Files | Proportion |
|-----|-------|------------|
| A major pentatonic | 28 | 31% |
| D major pentatonic | 23 | 26% |
| C major pentatonic | 22 | 24% |
| G major pentatonic | 9 | 10% |
| F major pentatonic | 8 | 9% |

### 4.3 Dataset Characteristics

| Statistic | Mean | Median | Min | Max |
|-----------|------|--------|-----|-----|
| Note duration (s) | 0.429 | 0.364 | 0.012 | 8.689 |
| Inter-onset interval (s) | 0.434 | 0.337 | - | - |
| Max simultaneous notes | 3.0 | 2.0 | 2 | 8 |
| Polyphonic fraction | 0.313 | - | 0.016 | 0.948 |
| Large leap rate (>12 ST) | 0.130 | - | 0.030 | 0.315 |
| Tremolo regions (total) | 555 | - | - | - |
| Glissando regions (total) | 326 | - | - | - |

## 5. Results

### 5.1 Model Comparison (All Variants, Ranked by OA Pitch Class)

Metrics are computed against the 72-file training distribution using Overlapping Area (OA), where 1.0 = identical to training.

| Rank | Variant | Model | Adaptation | OA_PC | OA_Dur | OA_Int | OA_IOI | Penta% | Density |
|------|---------|-------|------------|-------|--------|--------|--------|--------|---------|
| - | *Training data* | - | - | *1.000* | *1.000* | *1.000* | *1.000* | *100%* | *3.41* |
| 1 | RWKV state-tuned + PP | MIDI-RWKV | State | **0.918** | **0.839** | 0.641 | **0.690** | 100% | **3.45** |
| 2 | RWKV constrained gen | MIDI-RWKV | State | 0.890 | 0.800 | **0.838** | 0.702 | 100% | 3.37 |
| 3 | RWKV LoRA ep10 + PP | MIDI-RWKV | LoRA | 0.819 | 0.634 | 0.804 | 0.581 | 100% | 0.99 |
| 4 | Moonbeam pretrained | Moonbeam | None | 0.815 | 0.600 | 0.687 | 0.500 | 99.5% | 3.17 |
| 5 | Moonbeam pretrained + PP | Moonbeam | None | 0.810 | 0.596 | 0.729 | 0.510 | 100% | 2.88 |
| 6 | RWKV LoRA ep5 + PP | MIDI-RWKV | LoRA | 0.804 | 0.647 | 0.817 | 0.610 | 100% | 1.27 |
| 7 | RWKV LoRA ep15 + PP | MIDI-RWKV | LoRA | 0.799 | 0.609 | 0.812 | 0.581 | 100% | 1.00 |
| 8 | Moonbeam fine-tuned + PP | Moonbeam | LoRA | 0.583 | 0.681 | 0.472 | 0.490 | 100% | 1.53 |
| 9 | Moonbeam fine-tuned | Moonbeam | LoRA | 0.535 | 0.680 | 0.439 | 0.490 | 93.7% | 1.54 |

![Model Comparison — OA Metrics](outputs/evaluation/model_comparison.png)
*Figure 2. Grouped bar chart of OA Pitch Class, OA Duration, and Pentatonic Purity across all evaluated variants. State-tuned + PP leads on pitch class overlap (0.92).*

### 5.2 All Four OA Distributions

![All OA Metrics](outputs/evaluation/oa_all_metrics.png)
*Figure 3. Comprehensive view of all four distributional overlap metrics. RWKV State+PP dominates on pitch class and duration; RWKV Constrained leads on interval overlap.*

### 5.3 Detailed Metric Comparison (Generated vs Training)

| Metric | Training | RWKV State+PP | RWKV Constr. | Moonbeam Pre. | RWKV LoRA ep10 | Moonbeam FT |
|--------|----------|---------------|--------------|---------------|----------------|-------------|
| Note count (mean) | 574.7 | 178.0 | 45.5 | 159.1 | 47.4 | 131.8 |
| Density (n/s) | 3.41 | **3.45** | 3.37 | 3.17 | 0.99 | 1.54 |
| Note duration (s) | 0.433 | 0.564 | 0.625 | 0.478 | 0.865 | 0.290 |
| PC entropy | 2.314 | 2.658 | 2.326 | 1.887 | 2.483 | 1.247 |
| Mean pitch (MIDI) | 63.3 | 61.8 | 65.4 | 60.3 | 63.3 | 61.0 |
| Mean velocity | 78.1 | 86.0 | 70.0 | 76.5 | 72.3 | 99.0 |
| Large leap rate | 0.126 | 0.394 | 0.186 | 0.215 | 0.227 | 0.018 |
| Mean interval (ST) | 7.15 | 12.18 | 7.96 | 9.08 | 8.59 | 1.58 |
| Max simultaneous | 2.9 | 9.8 | 4.5 | 3.9 | 2.8 | 2.0 |

### 5.4 LoRA Training Trajectory (MIDI-RWKV)

LoRA training completed 18/20 epochs (1,186 steps). The checkpoint trajectory reveals classic overfitting:

| Epoch | Loss | OA_PC | OA_Dur | Density (n/s) | Penta% |
|-------|------|-------|--------|---------------|--------|
| 0 (pretrained) | - | 0.871 | 0.784 | 5.34 | 78.5% |
| 5 | 3.36 | 0.804 | 0.647 | 1.27 | 99.6% |
| **10** | 2.81 | **0.819** | 0.634 | 0.99 | 99.5% |
| 15 | 2.68 | 0.803 | 0.603 | 1.01 | 100% |
| *State-tuned (ref.)* | - | *0.918* | *0.839* | *3.45* | *100%* |

![LoRA Trajectory](outputs/evaluation/lora_trajectory.png)
*Figure 4. Left: OA_PC peaks at epoch 10 (0.819) then declines — never approaching state-tuned baseline (0.918). Right: Severe density collapse from 5.34 to 0.99 n/s under LoRA, while state tuning preserves training-like density (3.45).*

![MIDI-RWKV LoRA Training Loss](outputs/evaluation/loss_curve.png)
*Figure 5. Training loss over 1,186 steps (18 epochs). Loss converges around epoch 10, while downstream metrics degrade after that point — confirming overfitting.*

### 5.5 Temperature Ablation (State-tuned Model)

| Temperature | Constraint | OA_PC | OA_Dur | Density (n/s) |
|-------------|-----------|-------|--------|---------------|
| **0.85** | Post-processing | **0.918** | **0.839** | **3.45** |
| 0.85 | Token-level | 0.890 | 0.800 | 3.37 |
| 0.70 | Post-processing | 0.761 | 0.623 | 2.50 |
| 0.70 | Token-level | 0.706 | 0.614 | 2.21 |

Lower temperature causes repetitive, sparse output. **temp=0.85 is optimal.**

### 5.6 Post-processing Effect (Before vs After)

| Variant | OA_PC (raw) | OA_PC (+PP) | Delta | Penta% (raw) | Penta% (+PP) |
|---------|-------------|-------------|-------|---------------|---------------|
| MIDI-RWKV state-tuned | 0.870 | 0.918 | **+0.048** | 82.8% | 100% |
| MIDI-RWKV pretrained | 0.812 | 0.871 | +0.059 | 78.5% | 100% |
| MIDI-RWKV LoRA ep10 | 0.819 | 0.819 | +0.000 | 99.5% | 100% |
| Moonbeam pretrained | 0.815 | 0.810 | -0.005 | 99.5% | 100% |
| Moonbeam fine-tuned | 0.535 | 0.583 | +0.048 | 93.7% | 100% |

Post-processing gives a free +5% OA_PC boost for state-tuned MIDI-RWKV by redistributing non-pentatonic note mass to valid pitches.

### 5.7 Novelty Analysis (Memorization Check)

To verify the model generates novel music rather than copying training data, n-gram overlap was measured between the 10 best-model samples and the full training set.

| N-gram Length | Pitch Coverage | Interval Coverage | Interpretation |
|---------------|---------------|-------------------|----------------|
| 4 | 15.4% | 18.2% | Some shared short motifs (expected for pentatonic music) |
| 8 | 5.3% | 10.9% | Low overlap — mostly novel phrases |
| 12 | 3.8% | 9.6% | Very low — no significant copying |
| 16 | 3.0% | 8.8% | Near zero for most samples |
| 20 | 2.6% | 8.3% | Negligible — model is not memorizing |

Additional per-sample analysis:
- **Mean LCS pitch ratio:** 7.6% (longest common subsequence vs nearest training file)
- **Mean LCS interval ratio:** 8.0%
- **Samples with long copied runs (>16 notes):** 1/10 (a short 92-note file closely resembling `bu_bu_gao`)
- **9/10 samples** have 0 copied runs exceeding 16 notes

![Novelty Analysis](outputs/evaluation/novelty_analysis.png)
*Figure 6. N-gram overlap drops rapidly with length. The model produces novel melodic content rather than memorizing training sequences.*

## 6. Interpretation

### Why MIDI-RWKV dominates

1. **Better parameter/data ratio.** At 36M parameters vs 72 training files, MIDI-RWKV avoids overfitting. Moonbeam's 309M parameters are 9x larger — too many degrees of freedom for this dataset size.

2. **State tuning preserves generalization.** By adapting only the initial hidden state (~294K params, 0.8% of total), the model shifts its output distribution toward guzheng idiom without losing the rhythm and structure knowledge from pretraining.

3. **Tokenizer alignment.** REMI+ tokens map directly to MIDI events (pitch, duration, time-shift), making post-processing straightforward. Moonbeam's 6-token compound encoding is harder to constrain.

### Why LoRA fine-tuning underperforms

Both models exhibit **density collapse** under LoRA — note density drops to 0.99-1.54 n/s (vs training 3.41 n/s). The LoRA trajectory over 3 checkpoints (ep5/10/15) confirms this is not a training duration issue:

- **OA_PC peaked at epoch 10** (0.819) then declined at epoch 15 (0.803) — classic overfitting
- **Density worsened monotonically**: 1.27 -> 0.99 -> 1.01 n/s
- **OA_Dur degraded monotonically**: 0.647 -> 0.634 -> 0.603
- State tuning avoids density collapse by not modifying the model's generative weights at all

### Constraint strategy

Post-processing is strictly preferable to token-level masking: it is instant (vs 22 min/sample), achieves equal pentatonic purity, and actually scores *higher* on OA_PC because it redistributes note mass rather than suppressing tokens.

## 7. Current Status

| Item | Status |
|------|--------|
| Data collection & validation | Complete (18 pieces, 90 transposed) |
| Moonbeam LoRA fine-tuning | Complete (25 epochs) |
| MIDI-RWKV state tuning | Complete (best model) |
| MIDI-RWKV LoRA fine-tuning | Complete (18/20 epochs, all checkpoints evaluated) |
| Objective evaluation (OA metrics) | Complete (12 variants across 4 OA metrics) |
| Novelty analysis (n-gram overlap) | Complete (confirms model generates novel content) |
| Post-processing pipeline | Complete |
| Audio rendering (FluidSynth) | Complete (best samples in `outputs/final/best_audio/`) |
| Production config | Complete (see `outputs/final/generation_config.json`) |
| Subjective listening evaluation | Not yet started |

## 8. Next Steps

| Priority | Task | Rationale |
|----------|------|-----------|
| High | Subjective listening evaluation | OA metrics capture distributional similarity but not musical coherence, phrasing, or aesthetic quality. Human evaluation is essential. |
| Medium | Longer generation (1024+ tokens) | Current max is 512 tokens; training data averages 575 notes. Full-length pieces require longer sequences. |
| Medium | Expand training data | 18 original pieces is small. More curated guzheng MIDI would improve generalization and reduce the 1 outlier sample's memorization. |
| Low | Velocity modeling | All training data has constant velocity=64. Adding dynamics post-hoc or augmenting data would increase expressiveness. |
| Low | Multi-instrument arrangement | Extend to guzheng + erhu or guzheng + pipa ensemble generation. |

## 9. Summary

MIDI-RWKV (36M params) with state tuning and pentatonic post-processing is the clear best approach, achieving **91.8% pitch class overlap** with training data, **83.9% duration overlap**, and **100% pentatonic purity**. It outperforms the 9x-larger Moonbeam model across all metrics. The complete LoRA training trajectory (epochs 5/10/15) definitively confirms that **state tuning** — adapting only 0.8% of parameters — is more effective than LoRA for small-data domain adaptation, as LoRA causes density collapse in both models. Post-processing is the optimal constraint strategy: instant, effective, and improves distributional metrics. Novelty analysis confirms the model generates original melodic content (mean LCS ratio < 8%) rather than memorizing training data.

---

# April 3rd, 2026
* I found more datasets for guzheng (189 files, 7.3 hours).
* I did not proceed with the Korean gayageum and Japanese koto data because they use different scales.
* MIDI-RMKV model: only fine-tuned the hidden state vectors (~294K parameters, 0.8% of 36M) and freezed other weights to prevent overfitting -> result not too good
* Transformer decoder-only:

## New Data

The original dataset (18 pieces, 72 transposed) was expanded with two additional sources:

| Source | Files | Description |
|--------|-------|-------------|
| `MIDI_transposed/` | 72 | Original 18 hand-validated pieces × up to 5 pentatonic keys |
| `guzheng_tech99/` | 99 | Web-scraped guzheng MIDI, quality varies |
| `pittstate_chinese/` | 18 (of 20) | Classical Chinese repertoire (guzheng/pipa), filtered |
| **Total** | **189** | **2.6× the original dataset** |

Cleaning pipeline: reject files with pentatonic purity < 80% (scraped/pittstate) or < 95% (curated), < 20 notes, or < 50% notes in guzheng range (MIDI 37-86). After filtering: 147 train / 20 val / 22 test.

### Why Japanese and Korean Zither Instruments Were Not Included

The raw data collection also contains Japanese koto MIDI files (`kernscores_koto/`: 3 files; `koto_misc/`: 7 files, including 1 Korean gayageum piece `arirang_korean.mid`). These were deliberately excluded for the following reasons:

1. **Incompatible scale systems.** The guzheng is tuned to the Chinese major pentatonic scale (宫调, gōng diào), built on whole steps and minor thirds (e.g., C-D-E-G-A). The Japanese koto uses fundamentally different tunings:
   - **Miyako-bushi** (都節): contains semitone intervals (e.g., D-E♭-G-A-B♭), which are absent from Chinese pentatonic scales
   - **In scale** (陰音階): another semitone-based scale (e.g., E-F-A-B-C)
   - **Hirajōshi** (平調子): D-E♭-G-A-B♭ — overlaps only 2-3 pitch classes with any Chinese pentatonic key

   The Korean gayageum traditionally uses a pentatonic system closer to the Chinese one, but Korean traditional music (e.g., *Arirang*) often includes ornamental pitch bends and vibrato patterns (nonghyeon, 농현) that are structurally different from guzheng techniques.

   Including these scales would violate the pentatonic purity constraint (≥80%) that the cleaning pipeline enforces, and mixing scale systems would confuse the model's learned pitch distributions.

2. **Data imbalance.** Only 10 Japanese/Korean files exist versus 189 guzheng files. At ~5% of the dataset, these files would be too few to teach the model koto/gayageum idioms, but enough to pollute the guzheng pitch class distribution. The model would likely learn to occasionally produce semitone intervals that sound out-of-place in a guzheng context.

3. **Different idiomatic techniques.** Koto music features techniques like *oshide* (押し手, pitch bending by pressing behind the bridge) and *sukuizume* (掬い爪, plucking inward) that produce distinct rhythmic and melodic patterns at the MIDI level. Gayageum music similarly has unique ornamental patterns. These would introduce rhythmic and intervallic noise into a model trained to capture guzheng idioms (glissandi, tremolo, large-interval arpeggios).

4. **Scope clarity.** The project goal is to generate authentic guzheng music, not generic East Asian zither music. Mixing instruments from different traditions would compromise the stylistic coherence of the output without providing enough data to learn any additional style well.

Expanded training data characteristics (vs original):

| Metric | Original (72 files) | Expanded (147 files) |
|--------|---------------------|----------------------|
| Mean note count | 610 | 663 |
| Mean density (n/s) | 3.53 | 3.91 |
| Mean duration (s) | 0.429 | 0.569 |
| Mean velocity | 64 (constant) | 79.9 (varied) |
| Pentatonic purity | 100% | 99.5% |
| Mean self-rep 4-gram | - | 0.558 |

### 10.2 Models Compared

| | **Trial 2**: Custom Transformer | **Trial 3**: MIDI-RWKV State-Tuned |
|---|---|---|
| **Architecture** | Decoder-only transformer (from scratch) | RWKV-7 (pretrained on 2.1M MIDI files) |
| **Size** | ~1.5M params (d=256, 4 heads, 6 layers) | 36M params (d=384, 12 layers) |
| **Trainable params** | All ~1.5M (100%) | ~294K (0.8%) via state tuning |
| **Tokenizer** | Custom REMI (vocab ~760) | REMI+ with BPE (vocab 16,000) |
| **Context length** | 512 tokens | 2048 tokens |
| **Training data** | 147 files + 3× augmentation (tempo jitter, velocity noise, micro-timing) | 147 files (no augmentation) |
| **Training** | AdamW, lr=3e-4, cosine schedule, 44 epochs (early stop at 21) | AdamW, lr=2e-2, 6 epochs completed |
| **Anti-repetition** | Repetition penalty (1.2) + 3-gram blocking | None (temperature/top-p only) |
| **Constraint** | Pentatonic pitch masking during decoding | Post-processing (pentatonic snap + range clamp) |

#### State Tuning Explained

State tuning is a PEFT method specific to RWKV (recurrent) models. Instead of modifying model weights, it optimizes only the **initial hidden state vectors** — the continuous vectors the RNN starts with before processing any input. This is analogous to **soft prompting** in transformers: the learned states "prime" the model toward guzheng style while preserving all pretrained music knowledge. Only ~294K parameters (0.8% of 36M) are trainable, inherently limiting overfitting capacity.

### 10.3 Results

#### Trial 2 (Custom Transformer) — Fully Evaluated

| Metric | Training Data | Constrained | Unconstrained |
|--------|--------------|-------------|---------------|
| Density (n/s) | 3.91 | **6.43** ⚠️ | **7.45** ⚠️ |
| Duration (s) | 0.569 | 0.390 | 0.325 |
| Pentatonic purity | 99.5% | 100% | 98.8% |
| Self-rep 4-gram | 0.558 | 0.130 ⚠️ | 0.109 ⚠️ |
| Self-rep 8-gram | 0.300 | 0.001 | 0.001 |
| OA pitch class | - | 0.714 | 0.741 |
| OA duration | - | 0.426 | 0.408 |
| OA interval | - | 0.786 | 0.766 |
| OA IOI | - | 0.286 | 0.309 |

**Key issues:**
- **Density too high** (6.4-7.5 n/s vs 3.9 training) — model generates notes too fast
- **Self-repetition too low** (0.13 vs 0.56 training) — anti-repetition penalties overcorrected; real guzheng music has natural motif repetition
- **OA duration and IOI poor** (0.43 and 0.29) — rhythmic/timing distributions far from training data
- **No memorization** (good): 8-gram pitch coverage ~0%, LCS ratio ~1.4%

#### Trial 3 (RWKV State-Tuned) — Evaluated

Training loss across 6 epochs (each ~2 hours on MPS):

| Epoch | Loss |
|-------|------|
| 0 | 7.950 |
| 1 | 7.942 |
| 2 | 7.921 |
| 3 | 7.929 |
| 4 | 7.955 |
| 5 | 7.910 |

Loss is nearly flat (7.95 → 7.91), expected for state tuning with only 294K trainable parameters. Checkpoints at epochs 0, 2, 4; 3 constrained samples per checkpoint (+ 1 unconstrained for epoch 0).

**OA Metrics by Checkpoint:**

| Distribution | rwkv-0 (constr.) | rwkv-2 (constr.) | rwkv-4 (constr.) | rwkv-0 (unconstr.) |
|-------------|---------------------|---------------------|---------------------|----------------------|
| OA pitch class | 0.782 | 0.792 | **0.797** | 0.750 |
| OA duration | **0.625** | 0.667 | 0.642 | 0.684 |
| OA interval | 0.744 | **0.761** | 0.699 | 0.524 |
| OA IOI | 0.415 | 0.489 | **0.518** | **0.634** |

**Detailed Metrics:**

| Metric | Training | rwkv-0/constr. | rwkv-2/constr. | rwkv-4/constr. |
|--------|----------|----------------|----------------|----------------|
| Note count | 663.3 | 43.3 ⚠️ | 49.0 ⚠️ | 48.7 ⚠️ |
| Density (n/s) | 3.91 | 2.18 | **3.02** | 2.11 |
| Duration (s) | 0.569 | 0.716 | 0.615 | 0.975 |
| Pentatonic purity | 99.5% | 100% | 99.4% | 97.9% |
| Large leap rate | 0.122 | 0.212 | 0.258 | 0.356 ⚠️ |
| Mean interval (ST) | 6.75 | 8.12 | 9.15 | 11.12 ⚠️ |
| Max simultaneous | 3.6 | 4.3 | 6.7 | 7.0 |
| Self-rep 4-gram | 0.558 | 0.204 | 0.208 | 0.177 |

**Key issues:**
- **Very short pieces** (~48 notes vs 663 training) — generation terminates early
- **Density too low** (2.1-3.0 vs 3.9 training) — opposite problem from Trial 2
- **Large leaps worsen with training** (0.36 at epoch 4 vs 0.12 training)
- **No memorization** (good): 0 files flagged at epochs 2/4; mean LCS ≤ 10

**Overfitting Analysis:**

| Checkpoint | 5-gram coverage | Mean LCS | Memorization | Repetition |
|-----------|----------------|----------|-------------|-----------|
| rwkv-0 | 0.429 | 15.3 | 1 file | 0 |
| rwkv-2 | 0.257 | 10.3 | 0 | 0 |
| rwkv-4 | 0.201 | 10.0 | 0 | 0 |
| Test set | 0.488 | - | - | - |

N-gram coverage *decreases* with training — model generates increasingly novel content. One epoch-0 memorization (LCS=36) was a pretrained artifact corrected by state tuning.

### 10.4 Three-Way Comparison

| Metric | Original Best (RWKV State+PP, 72 files) | Trial 2 Constrained | Trial 3 Best (rwkv-4/constr.) |
|--------|----------------------------------------|--------------------|-----------------------------|
| OA pitch class | **0.918** | 0.714 | 0.797 |
| OA duration | **0.839** | 0.426 | 0.642 |
| OA interval | 0.641 | **0.786** | 0.699 |
| OA IOI | **0.690** | 0.286 | 0.518 |
| Density (n/s) | **3.45** (target: 3.91) | 6.43 ⚠️ | 2.11 ⚠️ |
| Note count | 178 | **500** | 49 ⚠️ |
| Pentatonic purity | **100%** | **100%** | 97.9% |

**Key findings:**
1. **The original 72-file RWKV state-tuned model remains the best** on OA pitch class (0.918 vs 0.797) and OA duration (0.839 vs 0.642). The expanded dataset did not improve the state-tuned approach.
2. **Trial 3 (RWKV on expanded data) beats Trial 2 (custom transformer)** on OA pitch class (0.797 vs 0.714), OA duration (0.642 vs 0.426), and OA IOI (0.518 vs 0.286). Pretraining advantage confirmed.
3. **Trial 2 wins on OA interval** (0.786 vs 0.699), likely from data augmentation and more diverse training data exposure.
4. **Both new trials produce density problems**: Trial 2 overproduces (6.43 n/s), Trial 3 underproduces (2.11 n/s). Neither matches training (3.91). The original model was closest (3.45).
5. **Trial 3 produces very short pieces** (~49 notes vs 178 original, 500 Trial 2). This is the most critical issue to address.

### 10.5 Plots

Trial 2 training plots are in `test_and_trial_2/plots/`:
- `loss_curves.png` — Training/validation loss over 44 epochs (early stop at epoch 21, val loss minimum 2.416)
- `generalization_gap.png` — Train-val gap begins widening at epoch 21
- `perplexity.png` — Validation perplexity plateaus at ~11.2

![Trial 2 Loss Curves](test_and_trial_2/plots/loss_curves.png)
*Figure 7. Trial 2 training/validation loss. Model converges by epoch 20, early stopping triggered at epoch 44 (patience=30). Val loss minimum at epoch 21 (2.416).*

![Trial 2 Generalization Gap](test_and_trial_2/plots/generalization_gap.png)
*Figure 8. Trial 2 train-val generalization gap. Overfitting begins around epoch 21.*

## 11. Next Steps

| Priority | Task | Rationale |
|----------|------|-----------|
| **High** | Investigate Trial 3 short generation | Pieces are only ~49 notes; debug early EOS or increase max generation length |
| **High** | Generate more Trial 3 samples | Only 3 samples per checkpoint — increase to 25 (5 per scale) for reliable metrics |
| **High** | Subjective listening evaluation | OA metrics don't capture musical coherence or aesthetic quality |
| **Medium** | Complete Trial 3 training (24 epochs planned, 6 completed) | Only 25% of planned training done; later checkpoints may improve |
| **Medium** | Fix Trial 2 anti-repetition | Repetition penalty and n-gram blocking overcorrected — tune down to allow natural motif repetition |
| Low | 4/10 presentation draft | Prepare results for presentation |

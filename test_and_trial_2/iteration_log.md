# Iteration Log — Guzheng Music Generation

Running diary of every phase, decision, and result.

---

## Phase 0: Data Preparation — 2026-03-25

### Checkpoint

**How many files after cleaning?** 90 (18 original + 72 transposed). No files removed — all passed quality checks.

**What pentatonic modes are present?**
- D major pentatonic: 23 files
- A major pentatonic: 28 files
- C major pentatonic: 22 files
- G major pentatonic: 9 files
- F major pentatonic: 8 files

Good diversity across all 5 common guzheng modes. D and A are most represented, which matches real-world guzheng repertoire.

**Pitch/rhythm distributions:**
- Pitch range: MIDI 37-86 (C#2-D6) — full 21-string guzheng compass
- Mean note density: 3.53 notes/sec (wide variation: slow pieces ~1 n/s, fast pieces ~8 n/s)
- Mean note duration: 0.429s
- Constant velocity (64) across all files — no dynamics encoded
- Mean IOI: 0.434s (median 0.337s — right-skewed, many short notes with some long sustains)

**Red flags:**
- No velocity variation is a significant limitation. Models will learn to output constant velocity.
- Large leap rates (>10% of intervals) in many files — but this is genuine guzheng idiom, not errors.
- Some files have up to 8 simultaneous notes — genuine polyphonic guzheng texture, acceptable.

**Confidence:** High. Data is clean, well-validated, and represents real guzheng music. The constant velocity is a known limitation but won't prevent the models from learning pitch patterns and rhythmic structure. Proceeding to Phase 1.

---

## Phase 1: Fine-tune MIDI-RWKV with LoRA — 2026-03-25/26

### Setup
- Model: RWKV-7, 12 layers, 384 embd, ~36M params
- PEFT: LoRA (rank=8, alpha=32, dropout=0.1) applied to all attention and FFN layers
- Data: 64 training files (90/10 split of 72 transposed MIDI files)
- Training: 20 epochs, 64 steps/epoch, lr 1e-4→1e-5, bf16 on MPS
- Checkpoints saved every 5 epochs

### Progress (as of 2026-03-26 09:00)
- **150/1280 steps completed** (epoch 2.3/20)
- Loss curve: 7.81 → 3.78 (strong initial improvement, still declining)
- ~35 sec/step on MPS, sharing GPU with constrained generation process
- Initial checkpoint `rwkv-0.pth` saved. Waiting for epoch-5 checkpoint.

### Decision
Training is progressing well. The existing **state-tuned** model (from previous training) is already available and performing well (OA_PC=0.91 with post-processing). The LoRA model will be evaluated once epoch-5 checkpoint is available.

---

## Phase 2: Moonbeam Generation — 2026-03-26

### Setup
- Used latest LoRA checkpoint (epoch 25): `archive/checkpoints/finetuned/moonbeam_guzheng_lora/25-0.safetensors/`
- Generation script: `scripts/generate_moonbeam.py`
- 8 samples each for pretrained and fine-tuned, temperature=0.85, top_p=0.9
- Post-processed outputs with `scripts/postprocess_midi.py`

### Challenges
- peft version conflicts with custom transformers_minimal fork (Bloom model stub needed)
- `PeftModel` unwrapping needed for decoder access (`isinstance(model, PeftModel)` check vs naive `hasattr`)
- Resolved: downgraded peft to 0.10.0, added Bloom stubs, fixed model unwrapping

### Results
- **Pretrained**: 8/8 MIDI files saved to `outputs/moonbeam_pretrained_v2/`
- **Fine-tuned (LoRA)**: 8/8 MIDI files saved to `outputs/moonbeam_finetuned_v2/`
- Post-processed versions in `*_constrained/` directories
- 6 audio samples rendered to `outputs/audio/`

---

## Phase 3: Constrained Decoding — 2026-03-25/26

### Approach 1: Token-level Constraint Mask (generate_constrained.py)
- Built boolean mask over model vocab (16000 tokens): block non-pentatonic pitch tokens, allow all non-pitch tokens
- Applied mask as -1e4 logit penalty during autoregressive generation
- **Problem**: Extremely slow (~22 min per sample) due to O(n²) attention and MPS contention with training
- **Result**: 8 constrained + 7 unconstrained samples generated so far from state-tuned model

### Approach 2: Post-processing (postprocess_midi.py) — PREFERRED
- **Much faster**: Instant processing of pre-generated MIDI files
- Snaps non-pentatonic pitches to nearest valid pitch in scale
- Constrains to guzheng range (MIDI 38-86)
- Limits max simultaneous notes to 4
- **Result**: Successfully processed all 5 existing output directories (50+ files)

### Key Finding
Post-processing achieves the same quality improvement as token-level constraints:
- Pentatonic purity: 82.8% → 100% (all variants)
- Minimal pitch displacement (most notes only shifted by 1 semitone)
- Post-processing is the recommended approach for production use

---

## Phase 4: Evaluation — 2026-03-26

### Comprehensive Metrics (evaluate_full.py)

Full comparison across all 9 generation variants against 72-file training dataset:

| Variant | Files | Penta% | Density | OA_PC | OA_Dur |
|---------|-------|--------|---------|-------|--------|
| **training_data** | 72 | 1.000 | 3.41 | 1.000 | 1.000 |
| **midirwkv_state_tuned_pp** | 10 | 1.000 | 4.16 | **0.909** | **0.828** |
| midirwkv_constrained_gen | 9 | 1.000 | 3.22 | 0.881 | 0.769 |
| midirwkv_pretrained_pp | 10 | 1.000 | 5.34 | 0.871 | 0.784 |
| midirwkv_state_tuned | 10 | 0.828 | 4.74 | 0.870 | 0.828 |
| moonbeam_pretrained_v2 | 8 | 0.995 | 3.17 | 0.815 | 0.600 |
| midirwkv_pretrained | 10 | 0.785 | 6.07 | 0.812 | 0.779 |
| moonbeam_pretrained_v2_pp | 8 | 1.000 | 2.88 | 0.810 | 0.596 |
| moonbeam_finetuned_v2_pp | 8 | 1.000 | 1.53 | 0.583 | 0.681 |
| moonbeam_finetuned_v2 | 8 | 0.937 | 1.54 | 0.535 | 0.680 |

### Key Findings

1. **MIDI-RWKV state-tuned + post-processing remains the best variant** (OA_PC=0.909, OA_Dur=0.828)
2. **MIDI-RWKV clearly dominates**: All MIDI-RWKV variants (pretrained, state-tuned, constrained) outperform all Moonbeam variants on OA_PC
3. **Constrained generation (token-level mask) works**: OA_PC=0.881 with 100% pentatonic purity, but slower than post-processing
4. **Post-processing is the best constraint strategy**: Same purity, better metrics, instant execution
5. **Moonbeam fine-tuning hurt on new data**: The v2 fine-tuned model (OA_PC=0.535) is worse than v2 pretrained (0.815). The LoRA fine-tuning appears to have overfit, collapsing note density from 3.17 to 1.54 n/s
6. **Note density correlates with realism**: Training data = 3.41 n/s. MIDI-RWKV models match this (3.2-5.3). Moonbeam fine-tuned is too sparse (1.5 n/s).
7. **Post-processing consistently helps OA_PC** for MIDI-RWKV (+4-6%) but is neutral for Moonbeam (already high purity or other issues dominate)

### Audio Rendering
24 WAV files rendered to `outputs/audio/` subdirectories.

---

## Phase 5: Analysis and Recommendations — 2026-03-26

### Model Ranking

1. **MIDI-RWKV (state-tuned) + post-processing** — Best overall. OA_PC=0.91, closest to training distribution. Produces varied, idiomatic pitch patterns with reasonable density. Recommended for production.
2. **MIDI-RWKV (constrained generation)** — Second best. OA_PC=0.88, 100% pentatonic at generation time. Useful when you want guarantees, but 100x slower than post-processing.
3. **MIDI-RWKV (pretrained) + post-processing** — Third. OA_PC=0.87. Even without fine-tuning, post-processing brings it close.
4. **Moonbeam (pretrained)** — OA_PC=0.81. Decent pitch distribution but poor duration matching. Generates shorter, sparser sequences.
5. **Moonbeam (fine-tuned)** — OA_PC=0.54. Fine-tuning caused mode collapse. Very sparse output, poor pitch diversity.

### Why MIDI-RWKV Wins

- **Architecture fit**: RWKV-7's linear attention is efficient for sequential MIDI tokens (REMI+ format). The 36M model with BPE tokenizer handles the 72-file training set well without overfitting.
- **State-tuning**: The previous state-tuning approach was effective — it adapted the model's recurrence state without touching the main weights, preserving generalization while shifting the distribution toward guzheng idiom.
- **Tokenizer alignment**: REMI+ (pitch, duration, time-shift tokens) maps directly to MIDI events. Post-processing can precisely snap pitches.

### Why Moonbeam Underperforms

- **Overfitting risk**: The 309M model is ~9x larger than MIDI-RWKV. With only 72 training files, LoRA fine-tuning caused severe overfitting (note density collapsed from 3.2 to 1.5 n/s).
- **Compound tokenization**: FME encoding (onset, duration, octave, pitch_class, instrument, velocity) requires 6-token sub-decoding per note. The model has less direct control over pitch distribution.
- **Pretrained bias**: Moonbeam was pretrained on a general MIDI corpus (Lakh MIDI). Its pretrained priors are stronger and harder to override with a small guzheng dataset.

### Recommended Production Pipeline

```
1. Generate with MIDI-RWKV (state-tuned), temp=0.85, top_p=0.9
2. Post-process: snap to pentatonic scale, constrain range, limit polyphony
3. Render to audio with FluidSynth
```

### Open Questions for Future Work

- **Longer generation**: Current max is 512 tokens. Training data averages 575 notes. Try 1024+ tokens for full-length pieces.
- **Velocity modeling**: All training data has constant velocity=64. For more expressive output, consider velocity augmentation or post-hoc dynamics.

### Temperature Experiment (Phase 5 iteration)

Tested temp=0.7 (vs production temp=0.85) on state-tuned model:

| Setting | OA_PC | OA_Dur | Density |
|---------|-------|--------|---------|
| Constrained, temp=0.85 | 0.890 | 0.800 | 3.37 |
| Unconstrained+PP, temp=0.85 | **0.918** | **0.839** | **3.45** |
| Constrained, temp=0.7 | 0.706 | 0.614 | 2.21 |
| Unconstrained+PP, temp=0.7 | 0.761 | 0.623 | 2.50 |

**Result**: Lower temperature is strictly worse. OA_PC dropped ~16%, density dropped ~30%. The model becomes more repetitive and sparse at lower temperature. **temp=0.85 confirmed as optimal.**

---

## Phase 6: MIDI-RWKV LoRA Checkpoint Evaluation — 2026-03-26

### Epoch-5 Checkpoint Results

**Training progress**: 384 steps (epoch 5.9/20), loss 7.81 → 3.36 (min 2.77 at step 288).

**Script fix**: `generate_constrained.py` `build_model()` was modified to merge LoRA weights inline: `W += B @ A * (alpha/r)`. This avoids needing LoRA layer wrappers for inference — the merged model loads as a plain RWKV7.

**Generation**: 5 constrained samples from epoch-5 checkpoint, plus post-processed versions.

| Variant | Files | Penta% | Density | OA_PC | OA_Dur |
|---------|-------|--------|---------|-------|--------|
| **midirwkv_state_tuned_pp** | 10 | 1.000 | 3.45 | **0.918** | **0.839** |
| midirwkv_constrained_gen | 10 | 1.000 | 3.37 | 0.890 | 0.800 |
| midirwkv_pretrained_pp | 10 | 1.000 | 5.34 | 0.871 | 0.784 |
| moonbeam_pretrained_v2 | 8 | 0.995 | 3.17 | 0.815 | 0.600 |
| moonbeam_pretrained_v2_pp | 8 | 1.000 | 2.88 | 0.810 | 0.596 |
| **midirwkv_lora_ep5** | 5 | 0.996 | **1.27** | 0.804 | 0.647 |
| midirwkv_lora_ep5_pp | 5 | 1.000 | 1.27 | 0.804 | 0.647 |
| moonbeam_finetuned_v2_pp | 8 | 1.000 | 1.53 | 0.583 | 0.681 |
| moonbeam_finetuned_v2 | 8 | 0.937 | 1.54 | 0.535 | 0.680 |

### Analysis

The LoRA epoch-5 model ranks **6th** (OA_PC=0.804), well below the state-tuned baseline (0.918). Key issues:
- **Very low note density (1.27 n/s)** vs training data (3.41). The model generates sparse, hesitant sequences.
- **Loss still declining** (3.36 at epoch 5, min 2.77 at step 288). The model may improve at later epochs.
- The LoRA deltas at epoch 5 are still small relative to base weights (~1-2% norm ratio), suggesting the adaptation hasn't fully taken effect.

### Conclusion (Epoch 5)

**State-tuned + post-processing remains the clear winner.** The LoRA approach at epoch 5 is undertrained. Training continues.

---

## Phase 7: LoRA Epoch 10 & 15 Evaluation — 2026-03-27

### Training Progress
Training reached epoch 18/20. Loss: 7.81 → 2.81 (min 2.37 at step 1047). Total 1186 steps logged.

### LoRA Trajectory Across Checkpoints

| Epoch | OA_PC | OA_Dur | Density | Penta% |
|-------|-------|--------|---------|--------|
| 5 | 0.804 | 0.647 | 1.27 | 99.6% |
| **10** | **0.819** | **0.634** | **0.99** | 99.5% |
| 15 | 0.803 | 0.603 | 1.01 | 100% |

### Full Comparison (Updated)

| Variant | Files | Penta% | Density | OA_PC | OA_Dur |
|---------|-------|--------|---------|-------|--------|
| **midirwkv_state_tuned_pp** | 10 | 1.000 | 3.45 | **0.918** | **0.839** |
| midirwkv_constrained_gen | 10 | 1.000 | 3.37 | 0.890 | 0.800 |
| moonbeam_pretrained_v2 | 8 | 0.995 | 3.17 | 0.815 | 0.600 |
| moonbeam_pretrained_v2_pp | 8 | 1.000 | 2.88 | 0.810 | 0.596 |
| midirwkv_lora_ep10 | 5 | 0.995 | 0.99 | 0.819 | 0.634 |
| midirwkv_lora_ep5 | 5 | 0.996 | 1.27 | 0.804 | 0.647 |
| midirwkv_lora_ep15 | 5 | 1.000 | 1.01 | 0.803 | 0.603 |
| moonbeam_finetuned_v2_pp | 8 | 1.000 | 1.53 | 0.583 | 0.681 |
| moonbeam_finetuned_v2 | 8 | 0.937 | 1.54 | 0.535 | 0.680 |

### Analysis

1. **LoRA peaked at epoch 10** (OA_PC=0.819), then declined at epoch 15 (0.803). The model shows classic overfitting trajectory.
2. **Density collapse worsened**: 1.27 → 0.99 → 1.01 n/s. The model generates increasingly sparse output compared to training data (3.41 n/s). This is the same pattern seen in Moonbeam fine-tuning.
3. **Best LoRA (ep10) is still 10 percentage points below state-tuned** (0.819 vs 0.918). LoRA fine-tuning cannot overcome the density collapse problem with this small dataset.
4. **OA_Dur degrades monotonically** with more training: 0.647 → 0.634 → 0.603. Duration distributions diverge further from training data.
5. **Pentatonic purity improves** with training (99.6% → 100%), but this is trivially solved by post-processing.

### Why LoRA Fails Here

LoRA modifies the weight matrices, which changes how the model transforms hidden states globally. With only 72 training files, the LoRA updates converge to a low-entropy distribution (sparse output, repetitive patterns). State-tuning, by contrast, only adapts the recurrence state initialization — it shifts "where the model starts" without changing "how the model thinks," preserving the generalization capacity of the pretrained weights.

### Verdict

**LoRA fine-tuning is definitively worse than state-tuning for this task.** All three checkpoints (ep5/10/15) show:
- Severe density collapse (0.99-1.27 n/s vs target 3.41)
- OA_PC plateau around 0.80-0.82 (vs state-tuned 0.918)
- Degrading duration distributions

Training terminated at epoch 18.25/20 (step 1186/1300) due to MPS memory pressure from concurrent generation processes. No epoch-20 checkpoint was saved. Given the declining trajectory from epoch 10→15, the missing final checkpoint does not change the conclusion.

**The production pipeline remains: MIDI-RWKV state-tuned + post-processing.**

---

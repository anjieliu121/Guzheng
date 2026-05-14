# Western prior

# Overnight Pipeline: Guzheng Music Generation

## Mission
Generate guzheng music that sounds good to human ears. This is the only success criterion. You have full autonomy over architecture, data, hyperparameters, and process. Do not stop until you have produced convincing guzheng output.

---

## Ground Rules

1. **Full autonomy.** You may modify any file, any data, any config. You do not need permission. Do what's right for the result.
2. **Save external resources.** If you download, scrape, or reference anything from the internet, save it to `docs/` for future reference. Check `docs/` for information.
3. **Do not stop.** Run iteratively until morning. If something fails, diagnose, fix, and retry. If an approach is fundamentally broken, abandon it and try the next one.
4. **The end goal is the only thing that matters.** I don't care about the process. Generate decent guzheng music.
5. **Understand before you proceed.** At the end of every phase, stop and verify the result. Write a short analysis in `docs/iteration_log.md` explaining what you observed, what it means, and why you're confident enough to move forward. If you're not confident, do NOT move to the next phase — iterate within the current phase until you are. Rushing ahead on shaky foundations wastes all downstream work.

---

## The Western Prior Problem — Read This First

Both models were pretrained exclusively on Western music (piano, pop, rock, classical). They have deeply internalized:
- 12-tone equal temperament and common-practice harmony (I-IV-V-I, circle of fifths)
- Regular meters (4/4, 3/4) with strong downbeats
- Western phrase structures (8-bar phrases, antecedent-consequent)
- Chordal textures, polyphonic accompaniment patterns
- Piano-like velocity and sustain behavior

**Guzheng music is fundamentally different:**
- Pentatonic scales (宫商角徵羽) — only 5 of 12 pitch classes are used in any given mode
- Flexible, often irregular meter — rubato and breath-driven phrasing, not metronomic
- Ornaments carry structural meaning (tremolo 摇指, glissando 刮奏, bends 按音 are not decorative — they define the musical identity)
- Monophonic or heterophonic texture — no chords, no bass-and-melody separation
- Plucked string dynamics — sharp attack, natural decay, no sustain pedal

**The risk:** Fine-tuning on fewer than 100 files may not override these deep priors. You may get output that sounds like a piano piece cosplaying as guzheng — correct pitches but wrong phrasing, wrong texture, wrong musical logic. Be vigilant for this. If the output "feels Western" despite being pentatonic, that's a failure mode you need to address.

---

## Context: Model Architectures

You have two models available. Read carefully before starting.

### Moonbeam (~309M params, Transformer + GRU sub-decoder)
- Pretrained on Lakh MIDI (81.6K hours), Western repertoire
- **Critical flaw:** Pitch is factored into octave + pitch class as independent tokens. The GRU sub-decoder generates them separately → wild octave jumps. Inference-time octave constraint (±1) partially mitigates but is a band-aid.
- Fine-tuning: LoRA on transformer layers
- Known failure modes: octave jumps, repetitive loops from overfitting

### MIDI-RWKV (~35M params, RWKV-7 linear recurrent)
- Pretrained on GigaMIDI (1.05M files), Western repertoire
- **Advantage:** Pitch is a single unified token (0–127). No octave mismatch possible. Architecturally suited for melodic coherence.
- Previous run used state tuning → failed (multi-track output persisted). **This run: use LoRA** to give enough capacity to learn single-track guzheng style.
- RWKV's smaller parameter count (35M vs 309M) may actually help — fewer parameters to overfit on ~100 files.
- Known failure modes: multi-track bleed from pretraining priors (when using state tuning)

### Key insight
The primary variable is **tokenization**: factored (Moonbeam) vs unified (MIDI-RWKV) pitch representation. MIDI-RWKV should produce better melodic output. But run both and compare.

---

## Pipeline Steps

### Phase 0: Data Preparation
**GATE: Do not leave this phase until you have clean, validated, well-understood training data.**

1. **Inventory** all guzheng MIDI files. Count them. Log file list.
2. **Deep inspection** — for at least 10 representative files, analyze:
   - Pitch range (expected: ~D2–D6, MIDI 38–86 for 21-string guzheng)
   - Pitch class distribution — confirm pentatonic adherence. Identify which pentatonic mode(s) are present.
   - Note density over time — is it monophonic? Sparse polyphony? Dense chords? (Dense chords = probably not real guzheng)
   - Track count — must be single-track. Extract guzheng track if multi-track.
   - Ornament patterns — look for rapid repeated notes (tremolo candidates), stepwise runs (glissando candidates), pitch bends if encoded
   - Duration and velocity distributions
   - Save this analysis as `docs/data_analysis_report.md` with plots
3. **Clean the data:**
   - Remove corrupt/empty files
   - Extract single guzheng track from multi-track files
   - Remove notes outside guzheng range
   - You may modify original data directly — you have permission
4. **Augment if needed** (likely, with ~100 files):
   - Transpose to pentatonic-friendly keys (maintain pentatonic structure — don't just shift chromatically)
   - Tempo variations (±10-20%)
   - Consider Chinese traditional music MIDI as supplementary (erhu, pipa) if guzheng-specific data is very limited
5. **Split data:** Reserve 10-15 files for validation. You need this to detect memorization.
6. **Log everything** to `docs/data_preparation_log.md`

**✅ Checkpoint — write in `docs/iteration_log.md` before proceeding:**
- How many files after cleaning? What pentatonic modes are present?
- What does the pitch/rhythm distribution look like?
- Any red flags in the data?
- Are you confident this data represents real guzheng music?

---

### Phase 1: Fine-tune MIDI-RWKV (Primary Model)
**GATE: Do not leave this phase until generated output is recognizably pentatonic, single-track, and in guzheng pitch range.**

#### 1a. Tokenization — Consider Custom Tokens
Before fine-tuning, consider injecting guzheng-specific tokens into the vocabulary:
- `ORNAMENT_TREMOLO` — for rapid repeated notes (detect heuristically: repeated alternating notes within short time window)
- `ORNAMENT_GLISSANDO` — for stepwise pitch runs (detect: 5+ consecutive stepwise notes within ~100ms)
- `PHRASE_BOUNDARY` — for musical breath points (detect: gaps > threshold in note onset)

Write a preprocessing script that analyzes MIDI files with these heuristics and inserts tokens into the tokenized sequence. This lets the model learn *when* to ornament rather than reconstructing ornaments note-by-note. **But:** only do this if you understand MIDITok's vocabulary well enough to inject tokens cleanly. If it's too risky or too complex, skip it, note why, and revisit later if needed.

#### 1b. Fine-tuning Strategy
- **Use LoRA.** Start with rank r=8.
- **Learning rate: start LOW.** 1e-5 or lower. You're nudging, not overwriting. With ~100 files, aggressive learning rates will memorize.
- **Consider freezing early layers** and only training the last few blocks. The early layers encode general music knowledge (rhythm, basic pitch relationships) which is still useful. The later layers encode style and structure — that's what needs to change. Inspect the model architecture, count the layers, decide how many to freeze based on what you see. This is your call once you look at the model.
- **Dropout: 0.2–0.3.** Aggressive, to combat overfitting on small data.
- **Early stopping** based on validation loss (that's why you reserved 10-15 files).
- **Train for few epochs.** Monitor for memorization — if generated pieces start reproducing training examples verbatim, you've gone too far. Roll back.
- Cosine decay schedule.

#### 1c. Training Loop
- Generate loss plots at regular intervals → `outputs/plots/midi_rwkv_loss.png`
- Generate samples every N epochs → `outputs/samples/midi_rwkv/`
- After each sample generation, run a quick sanity check:
  - Is it single-track? (If multi-track: LoRA isn't overriding the prior. Increase rank, unfreeze more layers, or force single-track in tokenization.)
  - Is pitch range within guzheng range?
  - Is pitch distribution roughly pentatonic?
  - Is note density reasonable (not chordal mush)?

#### 1d. If LoRA alone isn't enough
Try in order:
1. LoRA + state tuning combined
2. Unfreezing more layers (if you froze some)
3. Increasing LoRA rank
4. Full fine-tuning of last N layers with very low learning rate

**✅ Checkpoint — write in `docs/iteration_log.md` before proceeding:**
- What is the best MIDI-RWKV output so far?
- Is it single-track? Pentatonic? In range?
- Does it sound Western-in-disguise or genuinely Chinese in phrasing?
- What hyperparameters worked? What failed?
- Are you confident this output is good enough to move on, or do you need another iteration?

---

### Phase 2: Fine-tune Moonbeam (Secondary Model)
**GATE: Do not leave this phase until you have comparable samples to MIDI-RWKV for evaluation.**

1. Use LoRA on transformer layers. Same freezing/dropout/LR considerations as MIDI-RWKV.
2. **Octave constraint (±1) during inference is mandatory.** If octave jumps persist:
   - Try ±0 (same octave unless explicitly learned)
   - Post-processing: median filter on octave sequence
   - Reduce temperature on octave token generation specifically
3. Tokenize using Moonbeam's compound tokenizer. Check if custom token injection is feasible — same principle as MIDI-RWKV but implementation will differ.
4. Generate loss plots → `outputs/plots/moonbeam_loss.png`
5. Generate samples → `outputs/samples/moonbeam/`
6. Same sanity checks as Phase 1.

**✅ Checkpoint — write comparison notes in `docs/iteration_log.md`.**

---

### Phase 3: Constrained Decoding — Critical for Both Models
**This is where you get the most bang for your buck with limited data. Apply after fine-tuning.**

1. **Pentatonic scale mask:** Define the target pentatonic scale (e.g., D-E-F#-A-B for D major pentatonic). Before sampling each pitch token, zero out or heavily penalize logits for all non-pentatonic pitches. This eliminates an entire class of wrong-sounding output for free, without needing the model to learn it.
   - Allow relaxation during detected glissando passages (passing tones are OK in runs)
   - Try multiple pentatonic modes and pick what sounds best

2. **Pitch range constraint:** Mask pitches outside guzheng range (MIDI 38–86). Hard constraint.

3. **Velocity constraint:** Guzheng is plucked — specific attack profiles, no sustain pedal. Constrain velocity to realistic range based on what you observed in the training data.

4. **Note density constraint:** Enforce max simultaneous notes to prevent Western-style chordal textures. Solo guzheng is mostly monophonic with occasional dyads. If the model outputs 4-note chords, something is wrong.

5. **Temperature / top-k / top-p tuning:** Experiment. Lower temperature = safer but potentially boring. Higher = riskier but potentially more musical.

**Regenerate samples with constraints applied. Compare constrained vs unconstrained. The improvement should be dramatic. Save both versions.**

**✅ Checkpoint — write in `docs/iteration_log.md`:**
- How much did constrained decoding improve the output?
- Which constraints had the biggest impact?
- What's still wrong?

---

### Phase 4: Evaluation
Run standard symbolic music AI evaluation metrics:

1. **Pitch analysis:** pitch class histogram (generated vs training), pentatonic adherence ratio, pitch range stats
2. **Rhythm analysis:** note duration distribution, inter-onset interval distribution, comparison with training data
3. **Melodic analysis:** interval distribution, pitch entropy, self-similarity matrix
4. **Overlapping Area (OA) metric:** overlap between generated and real distributions for pitch, duration, velocity — standard in MuseGAN, Music Transformer papers
5. **Structural metrics:** pitch class histogram entropy, note density over time, groove consistency
6. **Qualitative:** convert best samples to audio (FluidSynth with guzheng/koto/zheng soundfont if available, otherwise GM). Save to `outputs/audio/`
7. Save everything to `outputs/evaluation/`

---

### Phase 5: Iterate
**This is where the overnight hours get spent.** After evaluating:

#### Diagnosis table:
| Symptom | Likely cause | Fix |
|---|---|---|
| Random/atonal pitches | LR too high, insufficient training | Lower LR, more epochs (watch memorization) |
| Western harmony feel | Pretrained priors not overridden | Freeze fewer layers, stronger pentatonic mask, more augmentation |
| Chordal texture | Western polyphonic prior | Enforce monophonic constraint at decoding, check data for multi-voice contamination |
| Repetitive loops | Overfitting / low temperature | Reduce epochs, increase dropout, raise temperature |
| Multi-track output (RWKV) | LoRA rank too low / state prior too strong | Increase LoRA rank, force single-track in post-processing |
| Octave jumps (Moonbeam) | Factored pitch representation | Tighten octave constraint, median filter |
| "Piano cosplaying as guzheng" | Right pitches, wrong phrasing/texture | Inject ornament tokens, constrain note density, adjust velocity profile |
| Verbatim training reproduction | Memorization | Fewer epochs, more dropout, more augmentation, check val loss divergence |

#### Iteration strategies (in order of effort, cheapest first):
1. **Adjust decoding constraints** — try different pentatonic modes, temperature, density limits
2. **Adjust training hyperparameters** — retrain with different LR, LoRA rank, dropout, epoch count
3. **Adjust what's frozen** — unfreeze/freeze different layers based on output analysis
4. **Augment data differently** — transpose to single key, add tempo variation
5. **Custom token injection** — add ornament tokens and retrain if not done yet
6. **Architectural changes** — last resort

**Re-run training and evaluation after each change. Repeat until output sounds like guzheng music.**

---

## Output Structure

```
outputs/
├── plots/
│   ├── midi_rwkv_loss.png
│   ├── moonbeam_loss.png
│   └── ...
├── samples/
│   ├── midi_rwkv/
│   │   ├── epoch_10.mid
│   │   ├── constrained/
│   │   └── ...
│   └── moonbeam/
│       ├── epoch_10.mid
│       ├── constrained/
│       └── ...
├── audio/
│   ├── midi_rwkv_best.wav
│   ├── moonbeam_best.wav
│   └── ...
├── evaluation/
│   ├── pitch_distribution.png
│   ├── interval_distribution.png
│   ├── rhythm_analysis.png
│   ├── oa_metrics.json
│   ├── evaluation_report.md
│   └── ...
└── final/
    ├── best_model_checkpoint/
    ├── best_samples/
    └── generation_config.json

docs/
├── data_analysis_report.md
├── data_preparation_log.md
├── iteration_log.md        ← Running diary. Write at EVERY checkpoint.
└── [any downloaded references]
```

---

## Decision Framework

When stuck, use this priority order:
1. **Does it sound like guzheng?** Not just "pentatonic" — does the phrasing, texture, ornament placement, and dynamics feel like a plucked Chinese zither? A piano playing pentatonic notes is still a failure.
2. **Constrained decoding first.** Before retraining, see if inference-time constraints fix the problem. 10x faster.
3. **Data quality > model tuning.** Bad data defeats any hyperparameter search.
4. **MIDI-RWKV is the primary bet** — unified pitch tokens, smaller parameter count (less overfitting risk).
5. **Moonbeam is the comparison model.** Get it working but don't spend disproportionate time.
6. **Understand each result before moving on.** If you can't explain why the output sounds the way it does, you don't know what to fix. Diagnose first, then act.

---

## Realistic Expectations

With ~100 files, expect output that is:
- Recognizably pentatonic with guzheng-like ornamental gestures
- Musically coherent for 30–60 seconds before potentially drifting
- More like plausible guzheng *sketches* than finished compositions

Producing convincing 30-60 second sketches is a real achievement and a valid thesis result.

---

## Final Deliverables

By morning, I expect:
- [ ] `docs/data_analysis_report.md` — thorough analysis of the training data
- [ ] `docs/iteration_log.md` — running diary of every phase, every decision, every result
- [ ] Training loss plots for both models
- [ ] Generated MIDI samples (multiple, from best checkpoints, with and without constrained decoding)
- [ ] Audio renders of best samples
- [ ] Evaluation metrics and comparison plots
- [ ] A clear recommendation: which model produces better guzheng music, why, and what the remaining gaps are

**Do not stop. Understand each step. Iterate until the music is convincing or you run out of night.**

# Test and Trial 3: MIDI-RWKV State-Tuning with Combined Dataset

## Goal
Fine-tune MIDI-RWKV using state-tuning on a combined dataset from three sources
to generate high-quality guzheng music without overfitting or excessive repetition.

## Data Sources
1. **MIDI_transposed/** (72 files) - Hand-validated curated pieces transposed to 5 pentatonic keys
2. **guzheng_tech99/** (99 files) - Web-scraped guzheng MIDI, pre-split train/val/test
3. **pittstate_chinese/** (20 files) - Classical Chinese repertoire (guzheng/pipa)

Combined: ~191 files before filtering (vs 72 in previous state-tuning trial).

## Method: MIDI-RWKV State-Tuning
- **Base model**: RWKV-7 (36M params, pretrained on GigaMIDI 2.1M files)
- **PEFT method**: State-tuning (~294K trainable params, 0.8% of total)
- Only optimizes initial hidden state vectors; all weights frozen
- Preserves pretrained generalization while adapting to guzheng style

## Key Differences from Previous State-Tuning (archive/midi-rwkv)
- **3x more training data** (191 vs 72 files)
- **More diverse sources** (curated + scraped + classical repertoire)
- **Lower learning rate** (2e-2 vs 5e-2) to prevent overfitting with larger dataset
- **More epochs with frequent saves** (24 epochs, save every 2) for checkpoint selection
- **Validation monitoring** via held-out files

## Anti-Overfitting Strategy
1. State-tuning inherently limits capacity (~294K params)
2. Larger, more diverse training set reduces memorization risk
3. Lower learning rate for smoother convergence
4. Multiple checkpoint saves for selection based on generation quality
5. Evaluation pipeline checks for memorization and repetition

## Anti-Repetition Strategy (Generation)
1. Temperature = 0.85 (balanced creativity/coherence)
2. Top-p = 0.9 nucleus sampling
3. Pentatonic scale constraints (pitch mask)
4. Guzheng range constraints (MIDI 38-86)
5. Post-processing: pentatonic snap + range clamp + polyphony limit

## Pipeline
```
01_prepare_data.py   -> Filter and collect MIDI from all 3 sources
02_train.sh          -> Run MIDI-RWKV state-tuning
03_generate.py       -> Generate constrained samples from each checkpoint
04_evaluate.py       -> Compute OA metrics vs training data
05_overfitting_check.py -> Check for memorization and repetition
```

## Evaluation Metrics
- OA_pitch_class: Pitch class distribution overlap with training data
- OA_duration: Duration distribution overlap
- OA_interval / OA_ioi: Interval and IOI distribution overlap
- Pentatonic purity (target: >0.95)
- Note density (target: 2-5 notes/sec, matching training data)
- Self-repetition scores (4/8/12-gram)
- N-gram coverage against training (memorization check)
- Longest common substring (plagiarism check)

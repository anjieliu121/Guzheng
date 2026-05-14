# Guzheng Music

## Instrument Overview

The guzheng (古筝) is a Chinese plucked zither with 21 strings spanning approximately D2–D6 (three octaves).

### Pentatonic Tuning

Standard tuning uses the Chinese pentatonic scale: **C, D, E, G, A** (宫商角徵羽 — gōng, shāng, jué, zhǐ, yǔ).

The 21 strings are tuned across ~4 octaves in this pentatonic pattern. Non-pentatonic notes (F, B, C#, etc.) are produced by pressing strings with the left hand to bend pitch.

Modern compositions sometimes use seven-tone (heptatonic) systems instead of the traditional five.

### Playing Techniques

**Right-hand techniques** (plucking):
- Basic plucking (勾托抹打)
- Arpeggio (琶音) — sweeping across strings
- Tremolo/finger shake (摇指) — rapid repetition of a single note
- Pinch techniques — paired notes at specific beat positions

**Left-hand techniques** (pressing/ornamentation):
- **Pressing (按音)** — pressing string to raise pitch (produces non-pentatonic notes)
- **Sliding (滑音)** — pitch bend up or down
- **Vibrato (颤音)** — oscillating pitch variation

These techniques are what give guzheng its distinctive character but are difficult to represent in standard MIDI.

---

## Prior AI Work on Guzheng

### 1. LSTM + DQN Reinforcement Learning (Chen et al. 2022)

- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9500105/
- **Method:** LSTM melody generation + DQN for guzheng playing techniques

**Architecture:**
- 3 stacked LSTM layers (512 hidden units each)
- 2 dropout layers (30%)
- 2 dense layers + softmax output
- Sequences of 100 timesteps, lr=0.0004, 5.4M+ params

**Music representation:**
- Numbered notation → staff notation → MIDI
- Piano timbre as proxy (MIDI lacks native guzheng)
- 128-dimensional pitch vectors, 10 frames/second

**DQN technique rewards:**
| Technique | Reward | Condition |
|-----------|--------|-----------|
| Arpeggio | +0.2 | At even beats |
| Finger shake | +0.3 | Identical sequential notes |
| Finger shake | −1.0 | Exceeding 6 repetitions |
| Pinch | +0.7 | Paired notes at appropriate beats |

DQN params: α=0.15, γ=0.8.

**Dataset:** 31 guzheng pieces (30s–4min each), segmented into 1000-line/10s sections.

**Evaluation:** Note accuracy (80%/75%/68% for deleting 10/20/25 notes) + subjective ranking by 10 guzheng players.

### 2. Firefly Algorithm + LSTM (Han et al. 2024)

- **Paper:** https://www.sciencedirect.com/science/article/pii/S2405844024081234
- Firefly algorithm + stacked LSTM with residual connections
- VAE embeddings for guzheng tune switching

### 3. GZGEN: Poetry-to-Guzheng (Diffusion Transformer)

- **Code/Data:** https://huggingface.co/NMLAB8/GZGEN
- Diffusion Transformer (DiT) with T5 text encoder
- Cross-attention: Chinese classical poetry → guzheng audio

### 4. CPTGZ: Painting-to-Guzheng

- **Paper:** https://www.researchgate.net/publication/384776505
- ViT + Llama 2 + latent diffusion
- Chinese painting → guzheng audio generation

### 5. Humming-to-Guzheng Melody

- **Paper:** https://link.springer.com/article/10.1007/s00530-025-01734-4
- YIN pitch detection + rule-based guzheng melody transcription from vocal humming

### 6. Guzheng Playing Technique Detection

- **CCMUSIC Database:** https://ccmusic-database.github.io/en/database/csmtd.html (~300 audios, ~20hrs)
- **GuzhengTech99:** https://lidcc.github.io/GuzhengTech99/
- Li et al., "Playing technique detection by fusing note onset information in guzheng performance," ISMIR 2022

---

## MIDI Representation Challenges for Guzheng

1. **No standard GM program** — guzheng is typically approximated with koto (program 107) or other plucked strings
2. **Left-hand ornaments** — pressing, sliding, vibrato require pitch bend or CC messages, not standard note events
3. **Pentatonic bias** — most notes fall on C/D/E/G/A; non-pentatonic notes are ornamental
4. **Sparse texture** — typically monophonic or lightly polyphonic, unlike piano
5. **Timbre variation** — different plucking positions/techniques produce different timbres (not captured in MIDI)

---

## Guzheng in Available Datasets

| Dataset | Guzheng Content |
|---------|----------------|
| XMIDI | One of 17 instrument categories (108K total files) |
| CCMUSIC | ~300 audio recordings, ~20hrs, technique labels |
| GuzhengTech99 | Playing technique detection |
| This project | Custom MIDI files in `MIDI/` and `MIDI_transposed/` |

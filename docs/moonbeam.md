# Moonbeam MIDI Foundation Model

- **Paper:** https://arxiv.org/abs/2505.15559 | [HTML](https://arxiv.org/html/2505.15559v1)
- **Code:** https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model
- **Authors:** Zixun Guo, Simon Dixon (Queen Mary University of London)

## Architecture Overview

Transformer-based autoregressive decoder with two key modifications:
1. **Multidimensional Relative Attention (MRA)** replaces standard multi-head attention
2. **GRU sub-decoder** sequentially decodes 6 attributes per musical event

## Tokenization: Fundamental Music Embedding (FME)

Each musical event is a 6-tuple: `x = (o, d, oct, p, i, v)`

| Attribute | Encoding | Description |
|-----------|----------|-------------|
| `o` (onset) | FME (continuous sinusoidal) | Absolute time position |
| `d` (duration) | FME | Note length |
| `oct` (octave) | FME | Octave number |
| `p` (pitch class) | FME | Pitch class (0-11) |
| `i` (instrument) | Standard embedding lookup | GM instrument program |
| `v` (velocity) | FME | Dynamic level |

**Key design choice:** Absolute onsets instead of timeshifts, because transformers struggle with arithmetic (aggregating deltas).

**FME advantages:**
- Fewer parameters than lookup tables (~hidden_size² vs dictionary_size × hidden_size)
- Handles interpolated/extrapolated inputs (microtonal, pitch bends, longer durations)
- After pretraining, quantization constraints can be relaxed for fine-tuning

### Vocabulary Sizes

| Token Type | Moonbeam (S) | Moonbeam (M) |
|------------|-------------|-------------|
| Time shift | 1,024 | 4,097 |
| Duration | 1,024 | 4,097 |
| Octave | 11 | 11 |
| Pitch class | 12 | 12 |
| Instrument | 129 | 129 |
| Velocity | 128 | 128 |
| **GRU output size** | **2,341** | **8,487** |

Special tokens: `<sos_gru>`, `<sos_x̃>`, `<eos_x̃>` per attribute (12 total), plus `<start-of-sequence>`, `<end-of-sequence>`, `<classification>`.

### MIDI Preprocessing

- Time quantized to **10ms intervals**
- Max timeshift/duration: 10,240ms (S) / 40,960ms (M) — files exceeding limits discarded
- Only note-level MIDI info tokenized (no CC, metadata, system messages)
- Input format uses absolute onsets; target format uses delta onsets (Δo)

## Multidimensional Relative Attention (MRA)

Extension of Rotary Position Embedding (RoPE) to N=5 music dimensions.

H attention heads partitioned into G=6 groups, each assigned a dimension:

| Group | Dimension | RoPE Base (θ) |
|-------|-----------|---------------|
| g=1 | onset | 199,999 |
| g=2 | duration | 1,031 |
| g=3 | octave | 19 |
| g=4 | pitch class | 20 |
| g=5 | onset (shared) | 199,999 |
| g=6 | velocity | 131 |

For group g:
```
f_q^g(Q, t_q) = Q_𝒢g · e^(i · v_g(t_q) · θ_g)
f_k^g(K, t_k) = K_𝒢g · e^(i · v_g(t_k) · θ_g)
```

**Key advantage:** Encodes absolute AND relative positional info with zero additional trainable parameters.

## GRU Sub-Decoder

1. Transformer outputs a single latent vector per timestep
2. This initializes the GRU hidden state
3. GRU sequentially predicts each of 6 attributes
4. Each prediction conditioned on previously generated attributes

Rationale: Only 6 tokens per step — GRU sufficient, avoids transformer overhead.

## Model Sizes

| Parameter | Moonbeam (S) | Moonbeam (M) |
|-----------|-------------|-------------|
| **Total params** | **309M** | **839M** |
| Hidden size | 1,536 | 1,920 |
| FFN size | 5,376 | 6,720 |
| Query heads | 12 | 12 |
| KV heads | 6 | 6 |
| Transformer layers | 9 | 15 |
| GRU layers | 2 | 4 |
| GRU hidden size | 1,024 | 1,536 |

## Training

### Pretraining
- **Data:** 81.6K hours (~18B tokens) from 20+ datasets
  - AriaMIDI (piano): 57,380 hrs / 8.40B tokens
  - MetaMIDI (multitrack): 18,189 hrs / 7.26B tokens
  - Maestro (piano): 197 hrs / 42.24M tokens
  - SymphonyNet (classical): 3,135 hrs / 1.61B tokens
- **Sequence length:** L=1024 (LLaMA concatenation with block-diagonal causal mask)
- **Optimizer:** Adam, lr=3e-4, decay 0.85/epoch, <9 epochs
- **Hardware:** Moonbeam (S): 2×A100-40GB → 54 hrs; Moonbeam (M): 2×A100-80GB → ~15 days
- **Loss:** Standard cross-entropy per sub-event

### Fine-tuning: Classification
- Append `<cls>` token, replace GRU with linear classification layer
- LoRA on transformer weights; `<cls>` embedding + classifier fully trainable
- Datasets: PiJAMA30, Pianist8, Emopia, GPM30

### Fine-tuning: Conditional Generation
- Non-temporal conditions `m` (genre, key, etc.) prepended with `<soc>`/`<eoc>` delimiters
- Temporal conditions `c` (chords) share event format, input directly to transformer
- Conditions `m` added to GRU via embedding + linear feature extractor
- **Sampling:** top-p=0.6, temperature=0.7

## Repository Structure

```
generation/          # Generation modules
recipes/             # Training/inference scripts
scripts/             # Utilities
src/llama_recipes/   # Core model config and components
tests/               # Test suite
data_preprocess.py   # MIDI preprocessing script
```

### Dependencies
- Python 3.12 (conda)
- `pip install .` then `pip install src/llama_recipes/transformers_minimal/.`
- Distributed training via torchrun

### Checkpoints
- **Pretrained:** `guozixunnicolas/moonbeam-midi-foundation-model` on HuggingFace
- Unconditional generation (ATEPP-Bach): TODO
- Conditional generation (CoMMU): TODO

### Data Preprocessing
`data_preprocess.py` discovers MIDI files, splits 90/10 train/test, outputs tokenized datasets with metadata CSV.

### Inference Parameters
- **Unconditional:** top_p=0.95, temperature=0.9, max 512 tokens, 50-token prompt
- **Conditional:** top_p=0.6, temperature=0.7, max 600 tokens

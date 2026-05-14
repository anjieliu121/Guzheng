# MIDI-RWKV

- **Paper:** https://arxiv.org/abs/2506.13001 | [HTML](https://arxiv.org/html/2506.13001v2)
- **Code:** https://github.com/christianazinn/MIDI-RWKV
- **Authors:** Christian Zhou-Zheng, Philippe Pasquier (Metacreation Lab, Simon Fraser University)

## Architecture Overview

RWKV-7 linear RNN for multi-track symbolic music infilling with controllable generation and state tuning for style adaptation.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Layers | 12 RWKV-7 layers |
| Head size | 64 |
| Hidden dim | 384 |
| FFN dim | 1,536 |
| **Total params** | **~35M** |
| Training seq length | 4,096 tokens |

Key advantage over transformers: O(n) complexity, can process infinitely long sequences (up to physical memory).

## Tokenization: REMI+ with BPE

### Base Encoding (REMI+)
Extends REMI with multi-track support. Tokens include:
- `Track_Start`, `Track_End` — track boundaries
- `Bar_Break` — measure boundaries
- `Position` — position within bar
- `Pitch`, `Velocity`, `Duration` — note attributes
- `Tempo`, `TimeSignature`, `Program` — global/track metadata

Tracks are placed sequentially in the token sequence.

### BPE Compression
- **Base vocabulary:** 663 REMI+ tokens
- **Post-BPE vocabulary:** 16,000 tokens
- Motivation: Reduce sequence length (like subword tokenization in NLP)
- Implementation: MidiTok library

## State Tuning

Core innovation for style adaptation. Instead of zero-initialized hidden states, learns initial conditions h₀,ᵢ for each layer.

### How It Works
1. Freeze all model weights
2. Optimize only initial state vectors using cross-entropy loss
3. Extracts information the model already learned — doesn't teach new info

### Training Procedure
- **Learning rate:** 5e-2 (high but stable)
- **Duration:** 16 epochs (~4 minutes)
- **Parameters trained:** 294K (L×d for vectors, L×d² for matrices)
- **No** learning rate decay or dropout

### Theoretical Rationale
From dynamical systems perspective: biases the model's trajectory through state space by adjusting initial conditions. Most effective for domains with global + local attributes (music: structure constant, style varies).

### Comparison
State tuning significantly outperformed both base model and LoRA fine-tuning on POP909 melody infilling (p<0.05 on CP, GS, PCHE across N=2,4,8 bar conditions).

## Infilling Approach

### Single-Section Infilling Format
1. Bars to infill are masked with `Infill_Bar` tokens (one per bar)
2. Original content moved to end of sequence, marked as infill content
3. Converts seq2seq → decoder-only generation (compatible with RWKV-7)

### Context Window
Default: C = 4N where N = infilling width (tested N = 2, 4, 8 bars).

### Attribute Controls (per-bar)
| Control | Description |
|---------|-------------|
| Note Density | 1–18 notes per bar (bins for 18+) |
| Note Duration | Binary tokens for {whole, half, quarter, eighth, sixteenth} |
| Polyphony | Min and max simultaneous notes at any onset |

Controls computed per-bar using MidiTok.

### Inference Parameters
- temperature=1.0
- repetition_penalty=1.2
- top-k=20
- top-p=0.95

Multiple masked regions handled via multiple model calls.

## Training

- **Dataset:** GigaMIDI train set (1.05M MIDI files)
- **Optimizer:** Adam, cosine lr 1e-4 → 1e-5
- **Weight decay:** 0.1
- **Batch size:** 16
- **Epochs:** 48
- **Hardware:** 1× RTX 4090
- **Wall time:** 64 hours
- **No dropout**

## Repository Structure

```
train/               # Pretraining scripts and dataset code
RWKV-PEFT/           # Submodule: LoRA and state tuning
rwkv.cpp/            # Submodule: inference engine
MIDIMetrics/         # Submodule: evaluation metrics
midi_rwkv.pth        # Pretrained base model
```

### Setup
```bash
conda create -n midirwkv python=3.11
conda activate midirwkv
pip install -r requirements.txt
# Critical: pytorch-lightning==1.9.5 for training
```

### Training
```bash
export PROJECT_ROOT=...
# Authenticate with HuggingFace for GigaMIDI access
./train/train.sh
```

### State Tuning
```bash
./RWKV-PEFT/scripts/run-state-tuning.sh
```

### LoRA Fine-tuning
```bash
./RWKV-PEFT/scripts/run-lora.sh
# Paper uses rank=alpha=4 and rank=alpha=32
```

### Inference
```bash
# 1. Build rwkv.cpp
# 2. Convert model:
./train/convert_model_to_cpp.sh midi_rwkv.pth
# 3. Configure rwkv.cpp/python/evaluate.sh
# 4. Run evaluation — outputs to MIDIMetrics/output/
```

### Custom Data
Modify `train/src/dataset.py` for custom MIDI collections. Data loading code can be injected into existing RWKV-LM or RWKV-PEFT working copies.

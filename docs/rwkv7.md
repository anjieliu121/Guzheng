# RWKV-7 "Goose" Architecture

- **Paper:** https://arxiv.org/abs/2503.14456
- **Code:** https://github.com/BlinkDL/RWKV-LM
- **Models:** https://huggingface.co/RWKV
- **Wiki:** https://wiki.rwkv.com/

## Overview

RWKV-7 is a linear RNN architecture that combines transformer-level parallelizable training with RNN-level O(n) inference. It achieves constant memory usage and constant inference time per token.

## Key Innovations

### 1. Generalized Delta Rule
A newly generalized formulation of the delta rule with:
- **Vector-valued gating** — extends scalar gating to per-dimension control
- **In-context learning rates** — dynamic adaptation during inference
- **Relaxed value replacement** — modification to standard state update

### 2. Expressive State Dynamics
The hidden state evolves more expressively than traditional RNNs. The state vectors encode previous token information with sufficient capacity for complex pattern recognition.

### 3. Computational Properties
- Can perform **state tracking** and recognize **all regular languages**
- Exceeds transformer capabilities under standard complexity conjectures (transformers limited to TC⁰)
- **O(n)** time and **O(1)** memory per token at inference

## Comparison to Transformers

| Property | Transformer | RWKV-7 |
|----------|------------|--------|
| Training | Parallelizable | Parallelizable |
| Inference complexity | O(n²) attention | O(n) linear |
| Memory per token | O(n) KV cache | O(1) fixed state |
| Sequence length | Limited by positional encoding | Unlimited (physical memory) |
| Computational class | TC⁰ | ≥ Regular languages |

## State Mechanism

At each timestep, RWKV-7 maintains fixed-size hidden state vectors per layer. These states are updated via the generalized delta rule:
- New information is written via gated updates
- Old information is selectively forgotten
- The state serves as compressed memory of all prior context

This is what enables **state tuning** (see [midi-rwkv.md](midi-rwkv.md)) — by learning optimal initial states, the model's entire generation trajectory can be biased toward a specific style.

## Model Sizes

The 2.9B parameter model achieves 3B SoTA on multilingual tasks despite being trained on dramatically fewer tokens than competitors.

Training corpus: 3.1 trillion token multilingual dataset (Apache 2.0 license).

## Related Tools

| Tool | Purpose | Link |
|------|---------|------|
| RWKV-PEFT | Fine-tuning: LoRA, state tuning | https://github.com/RWKV/RWKV-PEFT |
| rwkv.cpp | Fast inference | https://github.com/RWKV/rwkv.cpp |
| RWKV-LM | Training/inference code | https://github.com/BlinkDL/RWKV-LM |

## Earlier Versions

- **RWKV-4** ("Reinventing RNNs for the Transformer Era"): https://arxiv.org/abs/2305.13048
- Architecture history: https://wiki.rwkv.com/basic/architecture.html

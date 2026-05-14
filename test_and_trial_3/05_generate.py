#!/usr/bin/env python3
"""
Step 5: Generate MIDI samples from trained checkpoint.

- Generates constrained (pentatonic-masked) samples with repetition penalty
- Uses multiple scales for diversity
- Also generates unconstrained samples for comparison
- Reports generation statistics
"""

import argparse
import os
import time

import torch

from config import TokenizerConfig, ModelConfig, trial_root
from model import GuzhengTransformer
from scales import midi_pitches_for_scale, pitch_token_ids_for_scale
from tokenizer import MidiTokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Generate Guzheng MIDI")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_per_scale", type=int, default=5,
                        help="Number of samples per scale")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--tempo_bpm", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repeated tokens (1.0=none, >1.0=penalize)")
    parser.add_argument("--ngram_block_size", type=int, default=3,
                        help="Block repeated note-group n-grams (0=disabled)")
    parser.add_argument("--unconstrained", action="store_true",
                        help="Also generate unconstrained samples")
    args = parser.parse_args()

    root = trial_root()
    if args.checkpoint is None:
        args.checkpoint = os.path.join(root, "checkpoints", "best_model.pt")
    if args.output_dir is None:
        args.output_dir = os.path.join(root, "generated")

    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # Load model
    load_kw = {"map_location": device}
    try:
        ckpt = torch.load(args.checkpoint, weights_only=False, **load_kw)
    except TypeError:
        ckpt = torch.load(args.checkpoint, **load_kw)

    tok_cfg = TokenizerConfig(**ckpt["config"]["tokenizer"])
    model_cfg = ModelConfig(**ckpt["config"]["model"])

    model = GuzhengTransformer(
        vocab_size=tok_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=0.0,
        pad_token=tok_cfg.pad_token,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    print(f"Parameters: {model.param_count():,}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"N-gram block size: {args.ngram_block_size}")

    tokenizer = MidiTokenizer(tok_cfg)
    tempo_us = int(60_000_000 / args.tempo_bpm)

    scales = list(tok_cfg.key_scale_letters)

    # Generate constrained samples
    constrained_dir = os.path.join(args.output_dir, "constrained")
    os.makedirs(constrained_dir, exist_ok=True)

    print(f"\nGenerating {args.num_per_scale} constrained samples per scale ({scales})")
    all_stats = []

    for scale in scales:
        key_id = tok_cfg.key_token_id(scale)
        allowed_pitch = pitch_token_ids_for_scale(scale, tok_cfg)
        n_pitches = len(midi_pitches_for_scale(scale))

        for i in range(args.num_per_scale):
            t0 = time.time()
            prompt = [tok_cfg.bos_token, key_id]

            tokens = model.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token=tok_cfg.eos_token,
                tok_cfg=tok_cfg,
                allowed_pitch_token_ids=allowed_pitch,
                prefix_len=2,
                repetition_penalty=args.repetition_penalty,
                ngram_block_size=args.ngram_block_size,
            )

            elapsed = time.time() - t0
            notes = tokenizer.decode_tokens(tokens)

            out_path = os.path.join(constrained_dir, f"constrained_{scale}_{i:02d}.mid")
            tokenizer.tokens_to_midi(tokens, out_path, ticks_per_beat=480, tempo=tempo_us)

            stat = {
                "file": os.path.basename(out_path),
                "scale": scale,
                "n_notes": len(notes),
                "n_tokens": len(tokens),
                "constrained": True,
                "elapsed": round(elapsed, 1),
            }
            if notes:
                pitches = [n.pitch for n in notes]
                stat["pitch_range"] = [min(pitches), max(pitches)]
            all_stats.append(stat)
            print(f"  {scale}_{i:02d}: {len(notes)} notes, {len(tokens)} tokens, {elapsed:.1f}s")

    # Generate unconstrained samples for comparison
    if args.unconstrained:
        unconstrained_dir = os.path.join(args.output_dir, "unconstrained")
        os.makedirs(unconstrained_dir, exist_ok=True)
        print(f"\nGenerating {args.num_per_scale} unconstrained samples per scale")

        for scale in scales:
            key_id = tok_cfg.key_token_id(scale)

            for i in range(args.num_per_scale):
                t0 = time.time()
                prompt = [tok_cfg.bos_token, key_id]

                tokens = model.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    eos_token=tok_cfg.eos_token,
                    tok_cfg=tok_cfg,
                    allowed_pitch_token_ids=None,
                    prefix_len=2,
                    repetition_penalty=args.repetition_penalty,
                    ngram_block_size=args.ngram_block_size,
                )

                elapsed = time.time() - t0
                notes = tokenizer.decode_tokens(tokens)

                out_path = os.path.join(unconstrained_dir, f"unconstrained_{scale}_{i:02d}.mid")
                tokenizer.tokens_to_midi(tokens, out_path, ticks_per_beat=480, tempo=tempo_us)

                stat = {
                    "file": os.path.basename(out_path),
                    "scale": scale,
                    "n_notes": len(notes),
                    "n_tokens": len(tokens),
                    "constrained": False,
                    "elapsed": round(elapsed, 1),
                }
                if notes:
                    pitches = [n.pitch for n in notes]
                    stat["pitch_range"] = [min(pitches), max(pitches)]
                all_stats.append(stat)
                print(f"  {scale}_{i:02d}: {len(notes)} notes, {len(tokens)} tokens, {elapsed:.1f}s")

    # Save generation stats
    import json
    stats_path = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    total_constrained = sum(1 for s in all_stats if s["constrained"])
    total_unconstrained = sum(1 for s in all_stats if not s["constrained"])
    print(f"\nGeneration complete:")
    print(f"  Constrained: {total_constrained} samples in {constrained_dir}")
    if args.unconstrained:
        print(f"  Unconstrained: {total_unconstrained} samples in {unconstrained_dir}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()

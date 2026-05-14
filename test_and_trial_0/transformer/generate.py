#!/usr/bin/env python3
"""Generate guzheng MIDI from a trained transformer checkpoint."""

import argparse
import os
import time

import torch

from config import TokenizerConfig, ModelConfig
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
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="output/generated")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tempo_bpm", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--scale",
        default="D",
        help="Guzheng key (A, C, D, F, G). Generation uses BOS+KEY and pitch mask from guzheng_scales.json.",
    )
    parser.add_argument(
        "--no_pitch_mask",
        action="store_true",
        help="Do not restrict pitch tokens to scale (still uses key conditioning).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

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

    tokenizer = MidiTokenizer(tok_cfg)
    tempo_us = int(60_000_000 / args.tempo_bpm)
    os.makedirs(args.output_dir, exist_ok=True)

    scale = args.scale.strip().upper()
    if scale not in tok_cfg.key_scale_letters:
        raise SystemExit(
            f"--scale must be one of {tok_cfg.key_scale_letters}, got {args.scale!r}"
        )
    key_id = tok_cfg.key_token_id(scale)
    allowed_pitch = None if args.no_pitch_mask else pitch_token_ids_for_scale(scale, tok_cfg)
    n_scale_pitches = len(midi_pitches_for_scale(scale))
    print(
        f"Key token {key_id} ({scale}), "
        f"{n_scale_pitches} allowed MIDI pitches"
        + (" (mask off)" if args.no_pitch_mask else "")
    )

    for i in range(args.num_samples):
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
        )

        elapsed = time.time() - t0
        notes = tokenizer.decode_tokens(tokens)
        n_tokens = len(tokens)

        out_path = os.path.join(args.output_dir, f"guzheng_{i:02d}.mid")
        tokenizer.tokens_to_midi(tokens, out_path, ticks_per_beat=480, tempo=tempo_us)

        if notes:
            pitches = [n.pitch for n in notes]
            durs = [n.duration for n in notes]
            print(
                f"  Sample {i}: {len(notes)} notes, {n_tokens} tokens, "
                f"pitch [{min(pitches)}-{max(pitches)}], "
                f"dur [{min(durs)}-{max(durs)}] ticks, "
                f"{elapsed:.1f}s"
            )
        else:
            print(
                f"  Sample {i}: 0 notes decoded from {n_tokens} tokens "
                f"(try more training epochs), {elapsed:.1f}s"
            )

    print(f"\nGenerated {args.num_samples} samples in {args.output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 3: Generate guzheng MIDI from state-tuned MIDI-RWKV checkpoints.

Generates constrained (pentatonic-masked) samples from each saved checkpoint.
Supports batch generation across multiple checkpoints for comparison.

Run with midi_rwkv conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/archive/midi-rwkv/RWKV-PEFT
  /opt/miniconda3/envs/midi_rwkv/bin/python3 ../../test_and_trial_3/03_generate.py
"""

import os
import sys
import random
import json
import argparse
import glob
import time

import torch
import numpy as np

# ── disable torch.compile on MPS/CPU ─────────────────────────────────────────
if not torch.cuda.is_available():
    torch.compile = lambda f, **kwargs: f

# ── env vars for RWKV ────────────────────────────────────────────────────────
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_TRAIN_TYPE", "state")
os.environ.setdefault("FUSED_KERNEL", "0")
os.environ.setdefault("WKV", "torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── paths ─────────────────────────────────────────────────────────────────────
TRIAL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRIAL_ROOT)
RWKV_PEFT = os.path.join(REPO_ROOT, "archive/midi-rwkv/RWKV-PEFT")
sys.path.insert(0, RWKV_PEFT)

BASE_MODEL = os.path.join(REPO_ROOT, "archive/midi-rwkv/midi_rwkv.pth")
TOKENIZER = os.path.join(REPO_ROOT, "archive/midi-rwkv/train/tokenizer/tokenizer.json")
MIDI_DIR = os.path.join(REPO_ROOT, "MIDI_transposed")

# Pentatonic scale definitions
PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},
    "G": {7, 9, 11, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "A": {9, 11, 1, 4, 6},
    "F": {5, 7, 9, 0, 2},
}
PRESSED_PCS = {
    "D": {7, 1}, "G": {0, 6}, "C": {5, 11}, "A": {2, 8}, "F": {10, 4},
}
GUZHENG_PITCH_MIN = 38
GUZHENG_PITCH_MAX = 86


def build_model(peft_path=None):
    """Build MIDI-RWKV model with optional state-tuning checkpoint."""
    from rwkvt.args_type import TrainingArgs
    args = TrainingArgs(
        n_layer=12, n_embd=384, dim_att=384, dim_ffn=1344,
        vocab_size=16000, ctx_len=2048, head_size_a=64,
        head_size_divisor=8, train_type="state",
    )
    args.my_testing = os.environ.get("RWKV_MY_TESTING", "x070")
    args.my_timestamp = "inference"

    from rwkvt.rwkv7.model import RWKV7
    model = RWKV7(args)

    base_sd = torch.load(BASE_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(base_sd, strict=False)

    if peft_path and os.path.isfile(peft_path):
        peft_sd = torch.load(peft_path, map_location="cpu", weights_only=True)
        peft_sd = {(k[6:] if k.startswith("model.") else k): v for k, v in peft_sd.items()}
        model.load_state_dict(peft_sd, strict=False)
        print(f"Loaded state checkpoint: {peft_path}")

    model = model.to(torch.bfloat16).to(DEVICE).eval()
    return model


def load_tokenizer():
    from miditok import MMM
    return MMM(params=TOKENIZER)


def midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=64):
    from symusic import Score
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    score = Score.from_midi(midi_bytes)
    seq = tok.encode(score)
    bos_id = tok.vocab["BOS_None"]
    ids = [bos_id] + seq.ids[:max_prompt_tokens]
    return ids


def build_pitch_token_map(tok):
    """Map token IDs to MIDI pitch values."""
    pitch_map = {}
    for token_str, token_id in tok.vocab.items():
        if token_str.startswith("Pitch_"):
            try:
                pitch = int(token_str.split("_")[1])
                pitch_map[token_id] = pitch
            except (ValueError, IndexError):
                pass
    return pitch_map


def build_constraint_mask(tok, scale_name="D", allow_pressed=True, model_vocab_size=16000):
    """Build boolean mask: True = allowed, False = blocked."""
    tok_vocab_size = len(tok.vocab)
    mask = torch.zeros(model_vocab_size, dtype=torch.bool)
    mask[:tok_vocab_size] = True

    pitch_map = build_pitch_token_map(tok)
    penta_pcs = PENTATONIC_SCALES.get(scale_name, PENTATONIC_SCALES["D"])
    if allow_pressed:
        penta_pcs = penta_pcs | PRESSED_PCS.get(scale_name, set())

    blocked = 0
    for token_id, midi_pitch in pitch_map.items():
        pc = midi_pitch % 12
        if pc not in penta_pcs:
            mask[token_id] = False
            blocked += 1
        elif midi_pitch < GUZHENG_PITCH_MIN or midi_pitch > GUZHENG_PITCH_MAX:
            mask[token_id] = False
            blocked += 1

    print(f"  Constraint mask ({scale_name}): {blocked} pitch tokens blocked, "
          f"{sum(1 for t, p in pitch_map.items() if mask[t])} pitch allowed")
    return mask


def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(ps, dim=-1)
    ps[cum - ps > p] = 0.0
    ps.div_(ps.sum(dim=-1, keepdim=True))
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok).squeeze(-1)


@torch.inference_mode()
def generate_constrained(model, prompt_ids, tok, max_new_tokens=512,
                         temperature=0.85, top_p=0.9,
                         scale_name="D", allow_pressed=True,
                         constrain=True):
    """Autoregressive generation with pentatonic + range constraints."""
    eos_id = tok.vocab.get("EOS_None", 2)
    constraint_mask = None
    if constrain:
        constraint_mask = build_constraint_mask(tok, scale_name, allow_pressed).to(DEVICE)

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    for step in range(max_new_tokens):
        if step % 100 == 0 and step > 0:
            print(f"    step {step}/{max_new_tokens}")

        logits = model.forward_normal(tokens)
        next_logits = logits[0, -1, :].float()

        if constraint_mask is not None:
            next_logits[~constraint_mask] -= 1e4

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = sample_top_p(probs.unsqueeze(0), top_p).item()
        else:
            next_tok = next_logits.argmax().item()

        if next_tok == eos_id:
            break

        tokens = torch.cat([
            tokens,
            torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        ], dim=1)

    return tokens[0].tolist()


def tokens_to_midi(tok, token_ids, out_path):
    from miditok.classes import TokSequence
    bos_id = tok.vocab.get("BOS_None", 1)
    eos_id = tok.vocab.get("EOS_None", 2)
    clean = [t for t in token_ids if t not in (bos_id, eos_id)]
    if len(clean) < 5:
        return False
    seq = TokSequence(ids=clean, are_ids_encoded=True)
    score = tok.decode(seq)
    score.dump_midi(out_path)
    return True


def detect_scale_from_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in PENTATONIC_SCALES:
        return parts[1]
    return "D"


def find_checkpoints(ckpt_dir):
    """Find all state-tuning checkpoints in directory."""
    pattern = os.path.join(ckpt_dir, "rwkv-*.pth")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return ckpts


def main():
    parser = argparse.ArgumentParser(description="Generate guzheng MIDI from state-tuned MIDI-RWKV")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint path (default: latest in checkpoints/)")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Generate from all checkpoints for comparison")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=str, default=None,
                        help="Force a specific scale (D/G/C/A/F)")
    parser.add_argument("--unconstrained_too", action="store_true",
                        help="Also generate unconstrained samples")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")

    tok = load_tokenizer()
    print(f"Tokenizer loaded: {len(tok.vocab)} tokens")

    # Determine checkpoints to use
    ckpt_dir = os.path.join(TRIAL_ROOT, "checkpoints")
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    elif args.all_checkpoints:
        checkpoints = find_checkpoints(ckpt_dir)
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {ckpt_dir}")
            return
        print(f"Found {len(checkpoints)} checkpoints: {[os.path.basename(c) for c in checkpoints]}")
    else:
        checkpoints = find_checkpoints(ckpt_dir)
        if checkpoints:
            checkpoints = [checkpoints[-1]]  # Use latest
        else:
            print(f"ERROR: No checkpoints found in {ckpt_dir}")
            return

    # Get prompt MIDI files
    midi_files = sorted([
        os.path.join(MIDI_DIR, f)
        for f in os.listdir(MIDI_DIR) if f.endswith(".mid")
    ])
    sampled = random.sample(midi_files, min(args.n_samples, len(midi_files)))

    all_results = {}

    for ckpt_path in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'='*60}")
        print(f"Generating from checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        model = build_model(peft_path=ckpt_path)

        out_dir = os.path.join(TRIAL_ROOT, "generated", ckpt_name, "constrained")
        os.makedirs(out_dir, exist_ok=True)

        results = []
        for i, midi_path in enumerate(sampled):
            fname = os.path.basename(midi_path)
            scale = args.scale or detect_scale_from_filename(fname)
            print(f"\n  [{i+1}/{len(sampled)}] {fname} (scale: {scale})")

            t0 = time.time()
            prompt_ids = midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=args.prompt_tokens)

            gen_ids = generate_constrained(
                model, prompt_ids, tok,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                scale_name=scale, constrain=True,
            )
            elapsed = time.time() - t0

            out_path = os.path.join(out_dir, f"constrained_{i:02d}_{scale}.mid")
            ok = tokens_to_midi(tok, gen_ids, out_path)
            if ok:
                results.append({
                    "file": os.path.basename(out_path),
                    "source_prompt": fname,
                    "scale": scale,
                    "type": "constrained",
                    "tokens": len(gen_ids),
                    "elapsed": round(elapsed, 1),
                })
                print(f"    Saved: {out_path} ({len(gen_ids)} tokens, {elapsed:.1f}s)")

            # Unconstrained comparison
            if args.unconstrained_too:
                unc_dir = os.path.join(TRIAL_ROOT, "generated", ckpt_name, "unconstrained")
                os.makedirs(unc_dir, exist_ok=True)

                t0 = time.time()
                gen_ids_unc = generate_constrained(
                    model, prompt_ids, tok,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    scale_name=scale, constrain=False,
                )
                elapsed = time.time() - t0

                out_path_unc = os.path.join(unc_dir, f"unconstrained_{i:02d}_{scale}.mid")
                ok = tokens_to_midi(tok, gen_ids_unc, out_path_unc)
                if ok:
                    results.append({
                        "file": os.path.basename(out_path_unc),
                        "source_prompt": fname,
                        "scale": scale,
                        "type": "unconstrained",
                        "tokens": len(gen_ids_unc),
                        "elapsed": round(elapsed, 1),
                    })

        # Save generation log
        log_dir = os.path.join(TRIAL_ROOT, "generated", ckpt_name)
        log_path = os.path.join(log_dir, "generation_log.json")
        log_data = {
            "checkpoint": ckpt_path,
            "config": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens": args.prompt_tokens,
                "seed": args.seed,
                "n_samples": args.n_samples,
            },
            "results": results,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        all_results[ckpt_name] = results
        print(f"\nCheckpoint {ckpt_name}: generated {len(results)} samples")

        # Free model memory before loading next checkpoint
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for ckpt_name, results in all_results.items():
        constrained = [r for r in results if r["type"] == "constrained"]
        unconstrained = [r for r in results if r["type"] == "unconstrained"]
        print(f"  {ckpt_name}: {len(constrained)} constrained, {len(unconstrained)} unconstrained")


if __name__ == "__main__":
    main()

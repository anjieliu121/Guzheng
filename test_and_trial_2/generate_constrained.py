#!/usr/bin/env python3
"""
Constrained decoding for MIDI-RWKV: pentatonic scale mask, pitch range,
velocity constraint, note density limit.

Run with midi_rwkv conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/archive/midi-rwkv/RWKV-PEFT
  /opt/miniconda3/envs/midi_rwkv/bin/python3 ../../scripts/generate_constrained.py
"""

import os, sys, random, json, argparse
import torch
import numpy as np

# ── disable torch.compile on MPS/CPU ─────────────────────────────────────────
if not torch.cuda.is_available():
    torch.compile = lambda f, **kwargs: f

# ── env vars for RWKV ────────────────────────────────────────────────────────
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_TRAIN_TYPE", "state")
os.environ.setdefault("FUSED_KERNEL",    "0")
os.environ.setdefault("WKV",             "torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = "/Users/anjie/Documents/MyGuzheng/Guzheng"
RWKV_PEFT    = os.path.join(ROOT, "archive/midi-rwkv/RWKV-PEFT")
sys.path.insert(0, RWKV_PEFT)

BASE_MODEL   = os.path.join(ROOT, "archive/midi-rwkv/midi_rwkv.pth")
TOKENIZER    = os.path.join(ROOT, "archive/midi-rwkv/train/tokenizer/tokenizer.json")
MIDI_DIR     = os.path.join(ROOT, "MIDI_transposed")

# Pentatonic scale definitions (pitch class sets, MIDI note % 12)
PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},    # D E F# A B
    "G": {7, 9, 11, 2, 4},    # G A B D E
    "C": {0, 2, 4, 7, 9},     # C D E G A
    "A": {9, 11, 1, 4, 6},    # A B C# E F#
    "F": {5, 7, 9, 0, 2},     # F G A C D
}
# Pressed strings (scale degrees 4 and 7)
PRESSED_PCS = {
    "D": {7, 1},   # G, C#
    "G": {0, 6},   # C, F#
    "C": {5, 11},  # F, B
    "A": {2, 8},   # D, G#
    "F": {10, 4},  # A#, E
}
# Guzheng pitch range
GUZHENG_PITCH_MIN = 38  # D2
GUZHENG_PITCH_MAX = 86  # D6


def build_model(peft_path=None, peft_type="state", lora_alpha=32):
    """Build MIDI-RWKV model with optional PEFT weights.

    For peft_type="state": loads state-tuning checkpoint via strict=False.
    For peft_type="lora": merges LoRA weights (B @ A * alpha/r) into base
    model weights so the plain RWKV7 architecture can be used for inference.
    """
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

        if peft_type == "lora":
            # Merge LoRA weights into base model: W += B @ A * (alpha / r)
            merged = 0
            model_sd = model.state_dict()
            lora_keys = {k for k in peft_sd if ".lora_A" in k}
            for la_key in lora_keys:
                lb_key = la_key.replace(".lora_A", ".lora_B")
                w_key = la_key.replace(".lora_A", ".weight")
                if lb_key in peft_sd and w_key in model_sd:
                    lora_A = peft_sd[la_key].float()
                    lora_B = peft_sd[lb_key].float()
                    lora_r = lora_A.shape[0]
                    scale = lora_alpha / lora_r
                    model_sd[w_key] = model_sd[w_key].float() + (lora_B @ lora_A) * scale
                    merged += 1
            # Load non-LoRA params (ln weights, time_state, etc.)
            non_lora = {k: v for k, v in peft_sd.items() if "lora_" not in k}
            model_sd.update(non_lora)
            model.load_state_dict(model_sd, strict=False)
            print(f"Merged {merged} LoRA pairs (alpha={lora_alpha}) + {len(non_lora)} non-LoRA params from: {peft_path}")
        else:
            model.load_state_dict(peft_sd, strict=False)
            print(f"Loaded PEFT checkpoint: {peft_path}")

    model = model.to(torch.bfloat16).to(DEVICE).eval()
    return model


def load_tokenizer():
    from miditok import MMM
    return MMM(params=TOKENIZER)


def midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=128):
    from symusic import Score
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    score = Score.from_midi(midi_bytes)
    seq = tok.encode(score)
    bos_id = tok.vocab["BOS_None"]
    ids = [bos_id] + seq.ids[:max_prompt_tokens]
    return ids


def build_pitch_token_map(tok):
    """Map token IDs to MIDI pitch values for constraint application."""
    pitch_map = {}  # token_id -> midi_pitch or None
    for token_str, token_id in tok.vocab.items():
        # REMI+ tokens: "Pitch_X" where X is MIDI number
        if token_str.startswith("Pitch_"):
            try:
                pitch = int(token_str.split("_")[1])
                pitch_map[token_id] = pitch
            except (ValueError, IndexError):
                pass
    return pitch_map


def build_constraint_mask(tok, scale_name="D", allow_pressed=True, model_vocab_size=16000):
    """Build a boolean mask over model vocabulary: True = allowed, False = blocked.

    The tokenizer vocab is smaller (e.g. 663) than the model vocab (16000).
    We mask non-pentatonic pitch tokens and block unused token IDs beyond the
    tokenizer vocab.
    """
    tok_vocab_size = len(tok.vocab)
    # Start with all tokens allowed for the tokenizer range, blocked beyond
    mask = torch.zeros(model_vocab_size, dtype=torch.bool)
    mask[:tok_vocab_size] = True  # Allow all valid tokenizer tokens

    pitch_map = build_pitch_token_map(tok)

    penta_pcs = PENTATONIC_SCALES.get(scale_name, PENTATONIC_SCALES["D"])
    if allow_pressed:
        penta_pcs = penta_pcs | PRESSED_PCS.get(scale_name, set())

    blocked = 0
    for token_id, midi_pitch in pitch_map.items():
        pc = midi_pitch % 12
        # Block non-pentatonic pitches
        if pc not in penta_pcs:
            mask[token_id] = False
            blocked += 1
        # Block out-of-range pitches
        elif midi_pitch < GUZHENG_PITCH_MIN or midi_pitch > GUZHENG_PITCH_MAX:
            mask[token_id] = False
            blocked += 1

    print(f"  Constraint mask ({scale_name}): {blocked} pitch tokens blocked, "
          f"{sum(1 for t, p in pitch_map.items() if mask[t])} pitch allowed, "
          f"{mask.sum().item()} total allowed / {model_vocab_size}")
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
    """Autoregressive generation with optional pentatonic + range constraints."""
    eos_id = tok.vocab.get("EOS_None", 2)
    constraint_mask = None
    if constrain:
        constraint_mask = build_constraint_mask(tok, scale_name, allow_pressed).to(DEVICE)

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    for step in range(max_new_tokens):
        if step % 100 == 0:
            print(f"  Token {step}/{max_new_tokens} (seq len: {tokens.shape[1]})")

        logits = model.forward_normal(tokens)
        next_logits = logits[0, -1, :].float()

        # Apply constraint mask: heavily penalize blocked tokens
        if constraint_mask is not None:
            next_logits[~constraint_mask] -= 1e4

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = sample_top_p(probs.unsqueeze(0), top_p).item()
        else:
            next_tok = next_logits.argmax().item()

        if next_tok == eos_id:
            print("  EOS reached.")
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
    """Detect target pentatonic scale from transposed filename like 'piece_D.mid'."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in PENTATONIC_SCALES:
        return parts[1]
    return "D"  # default


def main():
    parser = argparse.ArgumentParser(description="Constrained MIDI-RWKV generation")
    parser.add_argument("--peft_dir", type=str, default=None,
                        help="Directory containing PEFT checkpoints")
    parser.add_argument("--peft_type", type=str, default="state",
                        choices=["state", "lora"],
                        help="Type of PEFT checkpoint")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "outputs/midirwkv_constrained"))
    parser.add_argument("--unconstrained_too", action="store_true",
                        help="Also generate unconstrained samples for comparison")
    parser.add_argument("--scale", type=str, default=None,
                        help="Force a specific pentatonic scale (D/G/C/A/F)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")

    # Load tokenizer
    tok = load_tokenizer()
    print(f"Tokenizer loaded: {len(tok.vocab)} tokens")

    # Find PEFT checkpoint
    peft_path = None
    if args.peft_dir and os.path.isdir(args.peft_dir):
        ckpts = sorted([
            os.path.join(args.peft_dir, f)
            for f in os.listdir(args.peft_dir)
            if f.startswith("rwkv-") and f.endswith(".pth")
        ], key=os.path.getmtime)
        if ckpts:
            peft_path = ckpts[-1]
            print(f"Using PEFT checkpoint: {peft_path}")

    # Build model
    model = build_model(peft_path=peft_path, peft_type=args.peft_type)
    print("Model loaded.")

    # Get prompt MIDI files
    midi_files = sorted([
        os.path.join(MIDI_DIR, f)
        for f in os.listdir(MIDI_DIR) if f.endswith(".mid")
    ])
    sampled = random.sample(midi_files, min(args.n_samples, len(midi_files)))

    # Generate constrained samples
    os.makedirs(args.out_dir, exist_ok=True)
    if args.unconstrained_too:
        os.makedirs(args.out_dir + "_unconstrained", exist_ok=True)

    results = []
    for i, midi_path in enumerate(sampled):
        fname = os.path.basename(midi_path)
        scale = args.scale or detect_scale_from_filename(fname)
        print(f"\n[{i+1}/{len(sampled)}] {fname} (scale: {scale})")

        prompt_ids = midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=args.prompt_tokens)

        # Constrained generation
        gen_ids = generate_constrained(
            model, prompt_ids, tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            scale_name=scale, constrain=True,
        )
        out_path = os.path.join(args.out_dir, f"constrained_{i:02d}_{scale}.mid")
        ok = tokens_to_midi(tok, gen_ids, out_path)
        if ok:
            print(f"  Saved constrained: {out_path}")
            results.append({"file": out_path, "scale": scale, "type": "constrained", "tokens": len(gen_ids)})

        # Unconstrained for comparison
        if args.unconstrained_too:
            gen_ids_unc = generate_constrained(
                model, prompt_ids, tok,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                scale_name=scale, constrain=False,
            )
            out_path_unc = os.path.join(args.out_dir + "_unconstrained", f"unconstrained_{i:02d}_{scale}.mid")
            ok = tokens_to_midi(tok, gen_ids_unc, out_path_unc)
            if ok:
                print(f"  Saved unconstrained: {out_path_unc}")
                results.append({"file": out_path_unc, "scale": scale, "type": "unconstrained", "tokens": len(gen_ids_unc)})

    # Save generation log
    log_path = os.path.join(args.out_dir, "generation_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "peft_path": peft_path,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens": args.prompt_tokens,
                "seed": args.seed,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nGeneration log saved to {log_path}")
    print(f"Generated {len(results)} samples total.")


if __name__ == "__main__":
    main()

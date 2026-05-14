#!/usr/bin/env python3
"""
Generate MIDI from MIDI-RWKV7 (pretrained and state-tuned fine-tuned).
Run with midi_rwkv conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/midi-rwkv/RWKV-PEFT
  conda run -n midi_rwkv python3 ../../outputs/generate_midirwkv.py
"""

import os, sys, random
import torch

# ── disable torch.compile on MPS/CPU ─────────────────────────────────────────
if not torch.cuda.is_available():
    torch.compile = lambda f, **kwargs: f

# ── env vars must be set BEFORE importing rwkvt modules ──────────────────────
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_TRAIN_TYPE", "state")   # enables time_state + RUN_RWKV7_STATE
os.environ.setdefault("FUSED_KERNEL",    "0")
os.environ.setdefault("WKV",             "torch")    # pure PyTorch impl for MPS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = "/Users/anjie/Documents/MyGuzheng/Guzheng"
RWKV_PEFT    = os.path.join(ROOT, "midi-rwkv/RWKV-PEFT")
sys.path.insert(0, RWKV_PEFT)

BASE_MODEL   = os.path.join(ROOT, "midi-rwkv/midi_rwkv.pth")
PEFT_DIR     = os.path.join(ROOT, "midi-rwkv/RWKV-PEFT/peft_model_guzheng")
TOKENIZER    = os.path.join(ROOT, "midi-rwkv/train/tokenizer/tokenizer.json")
MIDI_DIR     = os.path.join(ROOT, "MIDI_transposed")

OUT_PRETRAINED = os.path.join(ROOT, "outputs/midirwkv_pretrained")
OUT_FINETUNED  = os.path.join(ROOT, "outputs/midirwkv_finetuned")

N_SAMPLES   = 10
PROMPT_BARS = 4        # use first ~4 bars as prompt
MAX_NEW_TOK = 512
TEMPERATURE = 0.85
TOP_P       = 0.9
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ── model builder ─────────────────────────────────────────────────────────────
def build_model(state_ckpt_path=None):
    from rwkvt.args_type import TrainingArgs

    # my_testing and train_type come from env vars (set above), not TrainingArgs
    args = TrainingArgs(
        n_layer          = 12,
        n_embd           = 384,
        dim_att          = 384,
        dim_ffn          = 1344,
        vocab_size       = 16000,
        ctx_len          = 2048,
        head_size_a      = 64,
        head_size_divisor= 8,
        train_type       = "state",   # must match RWKV_TRAIN_TYPE env var
    )
    # Extra attrs required by model code but not in TrainingArgs dataclass
    args.my_testing  = os.environ.get("RWKV_MY_TESTING", "x070")
    args.my_timestamp = "inference"

    from rwkvt.rwkv7.model import RWKV7
    model = RWKV7(args)

    # Load base weights (no 'model.' prefix in .pth file)
    base_sd = torch.load(BASE_MODEL, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"Base model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Load state-tuning checkpoint (only has time_state keys)
    if state_ckpt_path is not None and os.path.isfile(state_ckpt_path):
        state_sd = torch.load(state_ckpt_path, map_location="cpu", weights_only=True)
        # Strip 'model.' prefix if present
        state_sd = {(k[6:] if k.startswith("model.") else k): v for k, v in state_sd.items()}
        missing2, unexpected2 = model.load_state_dict(state_sd, strict=False)
        print(f"State checkpoint loaded from {state_ckpt_path}.")
        print(f"  Missing: {len(missing2)}, Unexpected: {len(unexpected2)}")

    model = model.to(torch.bfloat16).to(DEVICE).eval()
    return model


# ── tokenizer ─────────────────────────────────────────────────────────────────
def load_tokenizer():
    from miditok import MMM
    return MMM(params=TOKENIZER)


# ── build prompt from a MIDI file ─────────────────────────────────────────────
def midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=128):
    from symusic import Score
    from miditok.classes import TokSequence
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    score = Score.from_midi(midi_bytes)
    seq = tok.encode(score)   # returns TokSequence (one_token_stream=True)
    # Prepend BOS
    bos_id = tok.vocab["BOS_None"]
    ids = [bos_id] + seq.ids[:max_prompt_tokens]
    return ids


# ── nucleus sampling ──────────────────────────────────────────────────────────
def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(ps, dim=-1)
    ps[cum - ps > p] = 0.0
    ps.div_(ps.sum(dim=-1, keepdim=True))
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok).squeeze(-1)


# ── autoregressive generation ─────────────────────────────────────────────────
@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens=512,
             temperature=0.85, top_p=0.9, eos_id=2):
    """Simple autoregressive generation (O(n^2) but works for short sequences)."""
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)  # (1, T)
    v_first = torch.empty(1, tokens.shape[1], model.args.n_embd, device=DEVICE, dtype=torch.bfloat16)

    for step in range(max_new_tokens):
        if step % 100 == 0:
            print(f"  Token {step}/{max_new_tokens} (total seq len: {tokens.shape[1]})")

        logits = model.forward_normal(tokens)   # (1, T, vocab_size)
        next_logits = logits[0, -1, :]           # (vocab_size,)

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


# ── decode tokens to MIDI and save ───────────────────────────────────────────
def tokens_to_midi(tok, token_ids, out_path):
    from miditok.classes import TokSequence
    # Remove BOS/EOS if present
    bos_id = tok.vocab.get("BOS_None", 1)
    eos_id = tok.vocab.get("EOS_None", 2)
    clean = [t for t in token_ids if t not in (bos_id, eos_id)]
    if len(clean) < 5:
        return False
    seq = TokSequence(ids=clean, are_ids_encoded=True)
    score = tok.decode(seq)
    score.dump_midi(out_path)
    return True


# ── get all MIDI files for prompts ───────────────────────────────────────────
def get_midi_files():
    files = []
    for fn in os.listdir(MIDI_DIR):
        if fn.endswith(".mid"):
            files.append(os.path.join(MIDI_DIR, fn))
    return sorted(files)


# ── run generation for one model variant ─────────────────────────────────────
def run(model, tok, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    midi_files = get_midi_files()
    if not midi_files:
        print(f"No MIDI files found in {MIDI_DIR}")
        return

    sampled = random.sample(midi_files, min(N_SAMPLES, len(midi_files)))
    eos_id  = tok.vocab.get("EOS_None", 2)

    print(f"\n=== Generating {len(sampled)} samples for {label} ===")
    saved = 0
    for i, midi_path in enumerate(sampled):
        print(f"\n  Sample {i+1}/{len(sampled)}: {os.path.basename(midi_path)}")
        prompt_ids = midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=64)
        gen_ids    = generate(model, prompt_ids, max_new_tokens=MAX_NEW_TOK,
                              temperature=TEMPERATURE, top_p=TOP_P, eos_id=eos_id)
        out_path   = os.path.join(out_dir, f"{label}_{i:02d}.mid")
        ok = tokens_to_midi(tok, gen_ids, out_path)
        if ok:
            print(f"  Saved {out_path} ({len(gen_ids)} tokens)")
            saved += 1
        else:
            print(f"  Sample {i} too short, skipped.")

    print(f"\nSaved {saved}/{len(sampled)} MIDI files to {out_dir}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok = load_tokenizer()

    # 1. Pretrained model
    print("\n>>> Building pretrained MIDI-RWKV ...")
    model_pre = build_model(state_ckpt_path=None)
    run(model_pre, tok, OUT_PRETRAINED, "midirwkv_pretrained")
    del model_pre
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # 2. Fine-tuned model (latest state checkpoint)
    # Sort by modification time to avoid lexicographic pitfall (rwkv-8 > rwkv-14 lexicographically)
    state_ckpts = sorted([
        os.path.join(PEFT_DIR, fn)
        for fn in os.listdir(PEFT_DIR)
        if fn.startswith("rwkv-") and fn.endswith(".pth")
    ], key=os.path.getmtime)
    if state_ckpts:
        latest = state_ckpts[-1]
        print(f"\n>>> Building fine-tuned MIDI-RWKV (state ckpt: {latest}) ...")
        model_ft = build_model(state_ckpt_path=latest)
        run(model_ft, tok, OUT_FINETUNED, "midirwkv_finetuned")
        del model_ft
    else:
        print("No state checkpoints found; skipping fine-tuned generation.")

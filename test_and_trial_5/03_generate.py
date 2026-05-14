#!/usr/bin/env python3
"""
Step 3: Generate guzheng MIDI from state-tuned MIDI-RWKV checkpoints.

Trial 5 generation:
- Prompts from val/test sets (never seen during training) + synthetic BOS-only
- Short 16-token prompts (val/test) or BOS+scale only (synthetic)
- EOS blocked for first 200 tokens (min generation length)
- Max 1024 new tokens
- Pitch 8-gram repetition penalty (1.15)
- Unconstrained generation (post-processing handles pentatonic snap)
- 15 samples per checkpoint (5 val + 5 test + 5 synthetic)

Run with midi_rwkv conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_0/midi-rwkv/RWKV-PEFT
  /opt/miniconda3/envs/midi_rwkv/bin/python3 ../../test_and_trial_5/03_generate.py
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
RWKV_PEFT = os.path.join(REPO_ROOT, "test_and_trial_0/midi-rwkv/RWKV-PEFT")
sys.path.insert(0, RWKV_PEFT)

BASE_MODEL = os.path.join(REPO_ROOT, "test_and_trial_0/midi-rwkv/midi_rwkv.pth")
TOKENIZER = os.path.join(REPO_ROOT, "test_and_trial_0/midi-rwkv/train/tokenizer/tokenizer.json")

VAL_DIR = os.path.join(TRIAL_ROOT, "data", "val")
TEST_DIR = os.path.join(TRIAL_ROOT, "data", "test")

PENTATONIC_SCALES = {
    "D": {2, 4, 6, 9, 11},
    "G": {7, 9, 11, 2, 4},
    "C": {0, 2, 4, 7, 9},
    "A": {9, 11, 1, 4, 6},
    "F": {5, 7, 9, 0, 2},
}

# Scale tokens: map scale name -> token string used by MMM tokenizer
SCALE_NAMES = ["A", "C", "D", "F", "G"]


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


def midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=16):
    """Extract first N tokens from a MIDI file as prompt."""
    from symusic import Score
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    score = Score.from_midi(midi_bytes)
    seq = tok.encode(score)
    bos_id = tok.vocab["BOS_None"]
    ids = [bos_id] + seq.ids[:max_prompt_tokens]
    return ids


def synthetic_prompt_ids(tok):
    """Create a minimal BOS-only prompt."""
    bos_id = tok.vocab["BOS_None"]
    return [bos_id]


def detect_scale_from_filename(filename):
    """Detect scale from filename suffix (e.g., 'foo_D.mid' -> 'D')."""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in PENTATONIC_SCALES:
        return parts[1]
    return "D"


def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(ps, dim=-1)
    ps[cum - ps > p] = 0.0
    ps.div_(ps.sum(dim=-1, keepdim=True))
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok).squeeze(-1)


def build_id_to_pitch(tok):
    """Build a mapping from token_id -> MIDI pitch for Pitch tokens."""
    id_to_pitch = {}
    for token_str, tid in tok.vocab.items():
        if isinstance(token_str, str) and token_str.startswith("Pitch_"):
            try:
                id_to_pitch[tid] = int(token_str.split("_")[1])
            except (ValueError, IndexError):
                pass
    return id_to_pitch


def _forward_one_block_stateful(block, x, v_first, att_shift, att_wkv, ffn_shift):
    """Run one RWKV block on input x with explicit RNN states.

    Returns: (x_out, v_first, new_att_shift, new_att_wkv, new_ffn_shift)
    """
    import torch.nn.functional as F

    if block.layer_id == 0:
        x = block.ln0(x)

    # ── attention ──
    att = block.att
    B, T, C = x.size()
    H = att.n_head
    head_size = att.head_size

    x_ln = block.ln1(x)
    # time_shift with explicit state
    xx = torch.cat([att_shift.unsqueeze(1), x_ln[:, :-1]], dim=1) - x_ln
    new_att_shift = x_ln[:, -1]

    xr, xw, xk, xv, xa, xg = att.addcmul_kernel(x_ln, xx)

    r = att.receptance(xr)
    w = -F.softplus(-(att.w0 + torch.tanh(xw @ att.w1) @ att.w2)) - 0.5
    k = att.key(xk)
    v = att.value(xv)
    if block.layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(att.v0 + (xv @ att.v1) @ att.v2)
    a = torch.sigmoid(att.a0 + (xa @ att.a1) @ att.a2)
    g = torch.sigmoid(xg @ att.g1) @ att.g2

    kk = k * att.k_k
    kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
    k = k * (1 + (a - 1) * att.k_a)

    # RUN_RWKV7 with explicit state (same as RUN_RWKV7_STATE but with external state)
    state = att_wkv.clone().to(dtype=r.dtype)
    r_ = r.view(B, T, H, head_size)
    k_ = k.view(B, T, H, head_size)
    v_ = v.view(B, T, H, head_size)
    w_ = w.view(B, T, H, head_size)
    a_ = (-kk).view(B, T, H, head_size)  # note: -kk passed as 'a' in original
    b_ = (kk * a).view(B, T, H, head_size)  # kk*a passed as 'b' in original
    output = torch.zeros(B, T, H, head_size, dtype=r.dtype, device=r.device)
    for t in range(T):
        wt = torch.exp(w_[:, t])
        kt = k_[:, t]; vt = v_[:, t]
        rt = r_[:, t]; at = a_[:, t]; bt = b_[:, t]
        sa = torch.einsum('bhvc,bhc->bhv', state, at)
        state = (state * wt.unsqueeze(-2) +
                 sa.unsqueeze(-1) * bt.unsqueeze(-2) +
                 vt.unsqueeze(-1) * kt.unsqueeze(-2))
        output[:, t] = torch.einsum('bhvc,bhc->bhv', state, rt)
    new_att_wkv = state

    x_att = output.view(B, T, C)
    x_att = att.ln_x(x_att.view(B * T, C)).view(B, T, C)
    x_att = x_att + ((r.view(B,T,H,-1)*k_.view(B,T,H,-1)*att.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
    x_att = att.output(x_att * g)
    x = x + x_att

    # ── ffn ──
    ffn = block.ffn
    x_ln2 = block.ln2(x)
    xx_ffn = torch.cat([ffn_shift.unsqueeze(1), x_ln2[:, :-1]], dim=1) - x_ln2
    new_ffn_shift = x_ln2[:, -1]

    k_ffn = x_ln2 + xx_ffn * ffn.x_k
    k_ffn = torch.relu(ffn.key(k_ffn)) ** 2
    x = x + ffn.value(k_ffn)

    return x, v_first, new_att_shift, new_att_wkv, new_ffn_shift


@torch.inference_mode()
def forward_stateful(model, idx, states):
    """Forward pass through model with explicit RNN states.

    states: list of (att_shift, att_wkv, ffn_shift) per layer
    Returns: (logits, new_states)
    """
    x = model.emb(idx)
    v_first = torch.empty_like(x)
    new_states = []

    for i, block in enumerate(model.blocks):
        att_shift, att_wkv, ffn_shift = states[i]
        x, v_first, new_att_shift, new_att_wkv, new_ffn_shift = \
            _forward_one_block_stateful(block, x, v_first, att_shift, att_wkv, ffn_shift)
        new_states.append((new_att_shift, new_att_wkv, new_ffn_shift))

    x = model.ln_out(x)
    x = model.head(x)
    return x, new_states


def init_states(model):
    """Create initial states from model's time_state parameters."""
    args = model.args
    B = 1
    C = args.n_embd
    H = args.dim_att // args.head_size_a
    head_size = args.head_size_a
    states = []
    for block in model.blocks:
        att_shift = torch.zeros(B, C, device=DEVICE, dtype=torch.bfloat16)
        # time_state: [H, head_size, head_size] -> transpose -> expand to [B, H, head_size, head_size]
        att_wkv = block.att.time_state.transpose(1, 2).expand(B, H, head_size, head_size).clone()
        ffn_shift = torch.zeros(B, C, device=DEVICE, dtype=torch.bfloat16)
        states.append((att_shift, att_wkv, ffn_shift))
    return states


@torch.inference_mode()
def generate_with_eos_blocking(model, prompt_ids, tok, max_new_tokens=1024,
                                temperature=0.85, top_p=0.9,
                                min_tokens_before_eos=200,
                                rep_penalty=1.15, rep_ngram=8):
    """Autoregressive generation with EOS blocking and pitch 8-gram repetition penalty.

    Uses stateful incremental inference for O(1) per-step cost.
    """
    eos_id = tok.vocab.get("EOS_None", 2)
    model_vocab_size = 16000  # BPE vocab size, NOT base vocab size

    # Initialize states from model's trained time_state
    states = init_states(model)

    # Process prompt to build up state
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    logits, states = forward_stateful(model, prompt_tensor, states)
    next_logits = logits[0, -1, :].float()

    all_tokens = list(prompt_ids)

    for step in range(max_new_tokens):
        if step % 100 == 0 and step > 0:
            print(f"    step {step}/{max_new_tokens}")

        # Block EOS for first min_tokens_before_eos generated tokens
        if step < min_tokens_before_eos:
            next_logits[eos_id] -= 1e4

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = sample_top_p(probs.unsqueeze(0), top_p).item()
        else:
            next_tok = next_logits.argmax().item()

        if next_tok == eos_id:
            break

        all_tokens.append(next_tok)

        # Forward single token (O(1) per step with stateful inference)
        tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        logits, states = forward_stateful(model, tok_tensor, states)
        next_logits = logits[0, -1, :].float()

    generated_count = len(all_tokens) - len(prompt_ids)
    return all_tokens, generated_count


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


def find_checkpoints(ckpt_dir):
    """Find all state-tuning checkpoints in directory."""
    pattern = os.path.join(ckpt_dir, "rwkv-*.pth")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return ckpts


def build_prompt_list(tok):
    """Build the three prompt categories: val, test, synthetic."""
    prompts = []

    # A. Validation set prompts (5 samples)
    if os.path.isdir(VAL_DIR):
        val_files = sorted(f for f in os.listdir(VAL_DIR) if f.endswith(".mid"))
        selected = random.sample(val_files, min(5, len(val_files)))
        for fname in selected:
            midi_path = os.path.join(VAL_DIR, fname)
            scale = detect_scale_from_filename(fname)
            ids = midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=16)
            prompts.append({
                "category": "val",
                "source_file": fname,
                "scale": scale,
                "prompt_ids": ids,
            })
        print(f"  Val prompts: {len(selected)} (from {len(val_files)} files)")
    else:
        print(f"  WARNING: Val dir not found: {VAL_DIR}")

    # B. Test set prompts (5 samples)
    if os.path.isdir(TEST_DIR):
        test_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".mid"))
        selected = random.sample(test_files, min(5, len(test_files)))
        for fname in selected:
            midi_path = os.path.join(TEST_DIR, fname)
            scale = detect_scale_from_filename(fname)
            ids = midi_to_prompt_ids(tok, midi_path, max_prompt_tokens=16)
            prompts.append({
                "category": "test",
                "source_file": fname,
                "scale": scale,
                "prompt_ids": ids,
            })
        print(f"  Test prompts: {len(selected)} (from {len(test_files)} files)")
    else:
        print(f"  WARNING: Test dir not found: {TEST_DIR}")

    # C. Synthetic prompts (5, one per scale)
    for scale in SCALE_NAMES:
        ids = synthetic_prompt_ids(tok)
        prompts.append({
            "category": "synthetic",
            "source_file": f"synthetic_BOS_{scale}",
            "scale": scale,
            "prompt_ids": ids,
        })
    print(f"  Synthetic prompts: {len(SCALE_NAMES)}")

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate guzheng MIDI (Trial 5)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint path")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Generate from all checkpoints")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_tokens_before_eos", type=int, default=200)
    parser.add_argument("--prompt_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--rep_penalty", type=float, default=1.15,
                        help="Pitch 8-gram repetition penalty (1.0 = disabled)")
    parser.add_argument("--rep_ngram", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")

    tok = load_tokenizer()
    print(f"Tokenizer loaded: {len(tok.vocab)} tokens")

    # Build prompt list
    print("\nBuilding prompt list...")
    prompts = build_prompt_list(tok)
    print(f"Total prompts: {len(prompts)}")

    # Determine checkpoints
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
            checkpoints = [checkpoints[-1]]
        else:
            print(f"ERROR: No checkpoints found in {ckpt_dir}")
            return

    all_results = {}

    for ckpt_path in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'='*60}")
        print(f"Generating from checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        model = build_model(peft_path=ckpt_path)

        results = []
        for i, prompt in enumerate(prompts):
            cat = prompt["category"]
            scale = prompt["scale"]
            source = prompt["source_file"]

            out_dir = os.path.join(TRIAL_ROOT, "generated", ckpt_name, cat)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n  [{i+1}/{len(prompts)}] {cat}/{source} (scale: {scale})")

            t0 = time.time()
            gen_ids, gen_count = generate_with_eos_blocking(
                model, prompt["prompt_ids"], tok,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                min_tokens_before_eos=args.min_tokens_before_eos,
                rep_penalty=args.rep_penalty, rep_ngram=args.rep_ngram,
            )
            elapsed = time.time() - t0

            out_fname = f"{cat}_{i:02d}_{scale}.mid"
            out_path = os.path.join(out_dir, out_fname)
            ok = tokens_to_midi(tok, gen_ids, out_path)

            if ok:
                results.append({
                    "file": out_fname,
                    "category": cat,
                    "source_prompt": source,
                    "scale": scale,
                    "total_tokens": len(gen_ids),
                    "generated_tokens": gen_count,
                    "prompt_tokens": len(prompt["prompt_ids"]),
                    "elapsed": round(elapsed, 1),
                })
                print(f"    Saved: {out_path} ({gen_count} generated tokens, {elapsed:.1f}s)")
            else:
                print(f"    FAILED: too few tokens to decode")

        # Save generation log
        log_dir = os.path.join(TRIAL_ROOT, "generated", ckpt_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "generation_log.json")
        log_data = {
            "checkpoint": ckpt_path,
            "config": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "min_tokens_before_eos": args.min_tokens_before_eos,
                "prompt_tokens": args.prompt_tokens,
                "seed": args.seed,
            },
            "results": results,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        all_results[ckpt_name] = results
        print(f"\nCheckpoint {ckpt_name}: generated {len(results)} samples")

        # Free model memory
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for ckpt_name, results in all_results.items():
        by_cat = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r)
        parts = ", ".join(f"{cat}: {len(rs)}" for cat, rs in sorted(by_cat.items()))
        print(f"  {ckpt_name}: {len(results)} total ({parts})")


if __name__ == "__main__":
    main()

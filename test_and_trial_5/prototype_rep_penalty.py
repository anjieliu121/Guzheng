#!/usr/bin/env python3
"""
Prototype: Pitch 8-gram repetition penalty for MIDI-RWKV generation.

Tests different penalty values (1.0, 1.10, 1.15, 1.20) on the base model
with synthetic BOS-only prompts (one per scale: A, C, D, F, G).

The 8-gram penalty works as follows:
- Track only Pitch_* tokens in the generated sequence
- Maintain the last 7 pitch tokens as context
- When a candidate pitch token would complete an 8-gram that has already
  appeared, reduce its logit by log(penalty)

Run with:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_0/midi-rwkv/RWKV-PEFT
  /opt/miniconda3/envs/midi_rwkv/bin/python3 ../../test_and_trial_5/prototype_rep_penalty.py
"""

import os
import sys
import math
import time
from collections import defaultdict

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

SCALE_NAMES = ["A", "C", "D", "F", "G"]
PENALTY_VALUES = [1.0, 1.10, 1.15, 1.20]

# Generation hyperparams (matching 03_generate.py)
TEMPERATURE = 0.85
TOP_P = 0.9
MAX_NEW_TOKENS = 768
MIN_TOKENS_BEFORE_EOS = 200
SEED = 42


def build_model():
    """Build base MIDI-RWKV model (no fine-tuning checkpoint)."""
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

    model = model.to(torch.bfloat16).to(DEVICE).eval()
    return model


def load_tokenizer():
    from miditok import MMM
    return MMM(params=TOKENIZER)


def get_pitch_token_ids(tok):
    """Return a set of token IDs whose name starts with 'Pitch_'."""
    pitch_ids = set()
    for token_name, token_id in tok.vocab.items():
        if token_name.startswith("Pitch_"):
            pitch_ids.add(token_id)
    return pitch_ids


def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(ps, dim=-1)
    ps[cum - ps > p] = 0.0
    ps.div_(ps.sum(dim=-1, keepdim=True))
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok).squeeze(-1)


@torch.inference_mode()
def generate_with_rep_penalty(model, prompt_ids, tok, pitch_token_ids,
                              rep_penalty=1.0, ngram_n=8,
                              max_new_tokens=MAX_NEW_TOKENS,
                              temperature=TEMPERATURE, top_p=TOP_P,
                              min_tokens_before_eos=MIN_TOKENS_BEFORE_EOS):
    """Autoregressive generation with pitch 8-gram repetition penalty.

    Args:
        rep_penalty: Repetition penalty multiplier (1.0 = no penalty).
            When a pitch token would complete a previously seen n-gram,
            its logit is reduced by log(penalty).
        ngram_n: The n-gram size to penalize (default 8).
    """
    eos_id = tok.vocab.get("EOS_None", 2)
    vocab_size = 16000
    tok_vocab_size = len(tok.vocab)

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    n_prompt = len(prompt_ids)

    # Track pitch tokens for n-gram penalty
    pitch_history = []  # list of pitch token IDs in generation order
    # Map from (n-1)-gram context tuple -> set of seen completions
    seen_ngrams = defaultdict(set)
    log_penalty = math.log(rep_penalty) if rep_penalty > 1.0 else 0.0
    penalty_applied_count = 0

    for step in range(max_new_tokens):
        if step % 200 == 0 and step > 0:
            print(f"    step {step}/{max_new_tokens}")

        logits = model.forward_normal(tokens)
        next_logits = logits[0, -1, :].float()

        # Block tokens beyond tokenizer vocab
        next_logits[tok_vocab_size:] -= 1e4

        # Block EOS for first min_tokens_before_eos generated tokens
        if step < min_tokens_before_eos:
            next_logits[eos_id] -= 1e4

        # Apply pitch n-gram repetition penalty
        if log_penalty > 0 and len(pitch_history) >= (ngram_n - 1):
            context = tuple(pitch_history[-(ngram_n - 1):])
            if context in seen_ngrams:
                banned = seen_ngrams[context]
                for pid in banned:
                    next_logits[pid] -= log_penalty
                    penalty_applied_count += 1

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = sample_top_p(probs.unsqueeze(0), top_p).item()
        else:
            next_tok = next_logits.argmax().item()

        if next_tok == eos_id:
            break

        # Update pitch history and n-gram tracking
        if next_tok in pitch_token_ids:
            pitch_history.append(next_tok)
            # Register all completed n-grams ending at this new pitch token
            if len(pitch_history) >= ngram_n:
                ctx = tuple(pitch_history[-ngram_n:-1])
                seen_ngrams[ctx].add(next_tok)

        tokens = torch.cat([
            tokens,
            torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        ], dim=1)

    generated_count = tokens.shape[1] - n_prompt
    return tokens[0].tolist(), generated_count, pitch_history, penalty_applied_count


def compute_self_repetition(pitch_sequence, n):
    """Compute n-gram self-repetition ratio for a pitch sequence.

    Returns the fraction of n-grams that are repeated (seen more than once).
    """
    if len(pitch_sequence) < n:
        return 0.0
    ngrams = []
    for i in range(len(pitch_sequence) - n + 1):
        ngrams.append(tuple(pitch_sequence[i:i + n]))
    total = len(ngrams)
    unique = len(set(ngrams))
    if total == 0:
        return 0.0
    # repetition ratio = 1 - (unique / total)
    return 1.0 - (unique / total)


def main():
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Penalty values: {PENALTY_VALUES}")
    print(f"Scales: {SCALE_NAMES}")
    print(f"N-gram size for penalty: 8")
    print()

    tok = load_tokenizer()
    print(f"Tokenizer loaded: {len(tok.vocab)} tokens")

    pitch_token_ids = get_pitch_token_ids(tok)
    print(f"Pitch tokens in vocab: {len(pitch_token_ids)}")

    bos_id = tok.vocab["BOS_None"]

    print("\nLoading base model (no fine-tuning)...")
    model = build_model()
    print("Model loaded.\n")

    # Results storage: penalty -> list of dicts
    all_results = {}

    for penalty in PENALTY_VALUES:
        print(f"{'='*70}")
        print(f"  PENALTY = {penalty:.2f}")
        print(f"{'='*70}")

        # Reset seed for each penalty so prompts are comparable
        torch.manual_seed(SEED)

        results = []
        for scale in SCALE_NAMES:
            prompt_ids = [bos_id]
            print(f"\n  Generating scale={scale}, penalty={penalty:.2f} ...")

            t0 = time.time()
            gen_ids, gen_count, pitch_hist, penalty_count = generate_with_rep_penalty(
                model, prompt_ids, tok, pitch_token_ids,
                rep_penalty=penalty,
            )
            elapsed = time.time() - t0

            # Compute self-repetition metrics
            rep4 = compute_self_repetition(pitch_hist, 4)
            rep8 = compute_self_repetition(pitch_hist, 8)
            rep12 = compute_self_repetition(pitch_hist, 12)

            result = {
                "scale": scale,
                "penalty": penalty,
                "total_tokens": len(gen_ids),
                "generated_tokens": gen_count,
                "pitch_tokens": len(pitch_hist),
                "penalty_applications": penalty_count,
                "rep_4gram": rep4,
                "rep_8gram": rep8,
                "rep_12gram": rep12,
                "elapsed": round(elapsed, 1),
            }
            results.append(result)

            print(f"    {gen_count} tokens ({len(pitch_hist)} pitch), "
                  f"penalties applied: {penalty_count}, "
                  f"rep4={rep4:.3f} rep8={rep8:.3f} rep12={rep12:.3f}, "
                  f"{elapsed:.1f}s")

        all_results[penalty] = results

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("RESULTS: Pitch 8-gram Repetition Penalty Comparison")
    print(f"{'='*90}")

    # Per-scale detail table
    header = f"{'Penalty':>8} {'Scale':>6} {'Pitch#':>7} {'PenApp':>7} {'Rep-4':>7} {'Rep-8':>7} {'Rep-12':>7} {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for penalty in PENALTY_VALUES:
        for r in all_results[penalty]:
            print(f"{r['penalty']:>8.2f} {r['scale']:>6} {r['pitch_tokens']:>7} "
                  f"{r['penalty_applications']:>7} {r['rep_4gram']:>7.3f} "
                  f"{r['rep_8gram']:>7.3f} {r['rep_12gram']:>7.3f} {r['elapsed']:>5.1f}s")
        print()

    # Summary table (averaged across scales)
    print(f"\n{'='*70}")
    print("SUMMARY (averaged across 5 scales)")
    print(f"{'='*70}")
    summary_header = f"{'Penalty':>8} {'AvgPitch':>9} {'AvgPenApp':>10} {'AvgRep4':>8} {'AvgRep8':>8} {'AvgRep12':>9}"
    print(summary_header)
    print("-" * len(summary_header))
    for penalty in PENALTY_VALUES:
        rs = all_results[penalty]
        avg_pitch = np.mean([r["pitch_tokens"] for r in rs])
        avg_pen = np.mean([r["penalty_applications"] for r in rs])
        avg_r4 = np.mean([r["rep_4gram"] for r in rs])
        avg_r8 = np.mean([r["rep_8gram"] for r in rs])
        avg_r12 = np.mean([r["rep_12gram"] for r in rs])
        print(f"{penalty:>8.2f} {avg_pitch:>9.1f} {avg_pen:>10.1f} "
              f"{avg_r4:>8.3f} {avg_r8:>8.3f} {avg_r12:>9.3f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()

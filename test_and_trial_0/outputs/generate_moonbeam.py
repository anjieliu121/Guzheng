#!/usr/bin/env python3
"""
Generate MIDI from Moonbeam (pretrained and LoRA fine-tuned).
Run with moonbeam conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/moonbeam
  conda run -n moonbeam python3 ../outputs/generate_moonbeam.py
"""

import os, sys, time, random
import numpy as np
import pandas as pd
import torch

# ── MPS device setup ──────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = "/Users/anjie/Documents/MyGuzheng/Guzheng"
MOONBEAM_SRC  = os.path.join(ROOT, "moonbeam/src")
TRANSFORMERS  = os.path.join(ROOT, "moonbeam/src/llama_recipes/transformers_minimal/src")
sys.path.insert(0, MOONBEAM_SRC)
sys.path.insert(0, TRANSFORMERS)

CKPT_PATH          = os.path.join(ROOT, "checkpoints/pretrained/moonbeam_309M.pt")
# model_config_small.json has onset/dur vocab_size=1026 matching decode_vocab_size=2341 in checkpoint.
# model_config_309M.json has onset/dur vocab_size=4099 which overflows the 2341-token decoder.
MODEL_CONFIG_PATH  = os.path.join(ROOT, "moonbeam/src/llama_recipes/configs/model_config_small.json")
LORA_BASE_DIR      = os.path.join(ROOT, "checkpoints/finetuned/moonbeam_guzheng_lora")
CSV_FILE           = os.path.join(ROOT, "data/moonbeam_preprocessed/train_test_split.csv")
DATA_DIR           = os.path.join(ROOT, "data/moonbeam_preprocessed/processed")

OUT_PRETRAINED = os.path.join(ROOT, "outputs/moonbeam_pretrained")
OUT_FINETUNED  = os.path.join(ROOT, "outputs/moonbeam_finetuned")

N_SAMPLES    = 10
PROMPT_LEN   = 5
MAX_GEN_LEN  = 300
TEMPERATURE  = 0.85
TOP_P        = 0.9
SEED         = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ── model + tokenizer builders ────────────────────────────────────────────────
def build_model(lora_dir=None):
    from transformers import LlamaConfig, LlamaForCausalLM
    from llama_recipes.datasets.music_tokenizer import MusicTokenizer

    llama_config = LlamaConfig.from_pretrained(MODEL_CONFIG_PATH)
    llama_config.use_cache = True

    model = LlamaForCausalLM(llama_config)

    raw = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = raw["model_state_dict"]
    new_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if lora_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"LoRA loaded from {lora_dir}")

    model = model.to(torch.bfloat16).to(DEVICE).eval()

    tokenizer = MusicTokenizer(
        timeshift_vocab_size  = llama_config.onset_vocab_size,
        dur_vocab_size        = llama_config.dur_vocab_size,
        octave_vocab_size     = llama_config.octave_vocab_size,
        pitch_class_vocab_size= llama_config.pitch_class_vocab_size,
        instrument_vocab_size = llama_config.instrument_vocab_size,
        velocity_vocab_size   = llama_config.velocity_vocab_size,
    )
    return model, tokenizer, llama_config


# ── nucleus sampling ──────────────────────────────────────────────────────────
def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(ps, dim=-1)
    ps[cumsum - ps > p] = 0.0
    ps = ps / ps.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok)


# ── MPS-compatible generate ───────────────────────────────────────────────────
@torch.inference_mode()
def generate(model, tokenizer, prompt_tokens, max_gen_len,
             temperature=0.85, top_p=0.9, max_len=2048):
    bsz = len(prompt_tokens)
    min_plen = min(len(t) for t in prompt_tokens)
    max_plen = max(len(t) for t in prompt_tokens)
    total_len = min(max_len, max_gen_len + max_plen)

    pad_id  = tokenizer.pad_token_compound
    pad_t   = torch.tensor(pad_id, dtype=torch.long, device=DEVICE).unsqueeze(0).unsqueeze(0)
    tokens  = pad_t.expand(bsz, total_len, -1).clone()

    for k, t in enumerate(prompt_tokens):
        tt = torch.tensor(t, dtype=torch.long, device=DEVICE)
        tokens[k, :len(t)] = tt

    prev_pos    = 0
    eos_reached = torch.tensor([False] * bsz, device=DEVICE)
    input_mask  = torch.all(tokens != pad_t, dim=-1).unsqueeze(-1)
    past_kv     = None

    for cur_pos in range(min_plen, total_len):
        if cur_pos % 100 == 0:
            print(f"  Token {cur_pos}/{total_len}")

        out = model.forward(
            input_ids       = tokens[:, prev_pos:cur_pos],
            past_key_values = past_kv,
            use_cache       = True,
            attention_mask  = None,
        )

        # Decoder GRU: autoregressively decode 6 compound attributes.
        # Only use last-position hidden state (all prior positions are wasted work).
        dec_tok = (torch.tensor(tokenizer.sos_out, device=DEVICE)
                   .to(tokens)
                   .expand(bsz, 1))
        dec_tok_out = dec_tok

        hidden = out.logits[:, -1, :]  # (bsz, decoder_hidden) — last position only
        hidden = (hidden.unsqueeze(0)
                  .expand(model.decoder.num_hidden_layers, -1, -1)
                  .contiguous())

        attrs = ["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode",
                 "pitch_dict_decode", "instrument_dict_decode", "velocity_dict_decode"]
        for attr in attrs:
            dec_out = model.forward(
                decoded_hidden_state    = hidden,
                decoded_language_tokens = dec_tok,
                attention_mask          = None,
            )
            gen_logits = dec_out.generation_logits
            hidden     = dec_out.generation_hidden_state

            sample_ids = list(getattr(tokenizer, attr).keys())

            if temperature > 0:
                lg = gen_logits[:, -1, :].float()
                lg = torch.clamp(lg / temperature, min=-60.0, max=60.0)
                # Pre-mask: only keep valid tokens for this attribute.
                # Uses -1e4 (not -inf) to guarantee finite softmax output.
                masked_lg = torch.full_like(lg, -1e4)
                masked_lg[:, sample_ids] = lg[:, sample_ids]
                probs   = torch.softmax(masked_lg, dim=-1)
                dec_tok = sample_top_p(probs, top_p)
            else:
                si_t    = torch.tensor(sample_ids, device=gen_logits.device)
                lg_last = gen_logits[:, -1, :]
                dec_tok = si_t[lg_last[:, si_t].argmax(dim=-1)].unsqueeze(-1)

            dec_tok_out = torch.cat([dec_tok_out, dec_tok], dim=-1)

        # Reshape and convert to compound tokens (shape: bsz × 1 × 6)
        dec_reshaped  = dec_tok_out[:, 1:].unsqueeze(1)  # (bsz, 1, 6)
        dec_lang      = tokenizer.convert_from_language_tokens(dec_reshaped).to(DEVICE)

        prev_onset  = tokens[:, cur_pos - 1, 0]
        new_onset   = prev_onset + dec_lang.clone().detach()[:, -1, 0].to(prev_onset)
        next_cmp    = torch.cat([new_onset.unsqueeze(-1),
                                 dec_lang.clone().detach()[:, -1, 1:]], dim=-1).to(tokens)

        next_tok = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_cmp)
        tokens[:, cur_pos] = next_tok

        # EOS check
        eos_conds = torch.stack([
            dec_lang.detach()[:, -1, 0] == tokenizer.eos_timeshift,
            dec_lang.detach()[:, -1, 1] == tokenizer.eos_dur,
            dec_lang.detach()[:, -1, 2] == tokenizer.eos_octave,
            dec_lang.detach()[:, -1, 3] == tokenizer.eos_pitch_class,
            dec_lang.detach()[:, -1, 4] == tokenizer.eos_instrument,
            dec_lang.detach()[:, -1, 5] == tokenizer.eos_velocity,
        ], dim=-1)
        eos_any = torch.any(eos_conds, dim=-1).to(input_mask)
        eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_any

        prev_pos = cur_pos
        past_kv  = out.past_key_values

        if all(eos_reached):
            print("  EOS reached.")
            break

    tokens = tokens[:, 1:, :]   # remove SOS

    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[:len(prompt_tokens[i]) + max_gen_len]
        eos_attrs = [tokenizer.eos_dur, tokenizer.eos_octave, tokenizer.eos_pitch_class,
                     tokenizer.eos_instrument, tokenizer.eos_velocity]
        for j, stop in enumerate(eos_attrs, start=1):
            try:
                idx = [row[j] for row in toks].index(stop)
                toks = toks[:idx]
            except ValueError:
                pass
        out_tokens.append(toks)
    return out_tokens


# ── load test prompts ─────────────────────────────────────────────────────────
def load_prompts(tokenizer):
    df = pd.read_csv(CSV_FILE)
    test_files = df[df["split"] == "test"]["file_base_name"].tolist()
    sampled = random.sample(test_files, min(N_SAMPLES, len(test_files)))
    prompts = []
    for fn in sampled:
        data = np.load(os.path.join(DATA_DIR, fn))
        encoded = tokenizer.encode_series(data, if_add_sos=True, if_add_eos=False)
        prompts.append(encoded[:PROMPT_LEN])
    return prompts


# ── run generation ────────────────────────────────────────────────────────────
def run(model, tokenizer, llama_config, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_prompts(tokenizer)
    print(f"\n=== Generating {len(prompts)} samples for {label} ===")

    out_tokens = generate(
        model, tokenizer, prompts,
        max_gen_len=MAX_GEN_LEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_len=llama_config.max_len if hasattr(llama_config, "max_len") else 2048,
    )

    saved = 0
    for i, (toks, prompt) in enumerate(zip(out_tokens, prompts)):
        if len(toks) < 2:
            print(f"  Sample {i}: too short, skipping")
            continue
        midi = tokenizer.compound_to_midi(toks)
        path = os.path.join(out_dir, f"{label}_{i:02d}.mid")
        midi.save(path)
        print(f"  Saved {path} ({len(toks)} tokens)")
        saved += 1
    print(f"Saved {saved}/{len(prompts)} MIDI files to {out_dir}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Pretrained model
    print("\n>>> Building pretrained Moonbeam ...")
    model_pre, tok_pre, cfg_pre = build_model(lora_dir=None)
    run(model_pre, tok_pre, cfg_pre, OUT_PRETRAINED, "moonbeam_pretrained")
    del model_pre
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # 2. Fine-tuned model (latest LoRA checkpoint)
    # Sort by modification time so epoch 10 sorts after epoch 9 (avoids lexicographic pitfall)
    lora_dirs = sorted([
        os.path.join(LORA_BASE_DIR, d)
        for d in os.listdir(LORA_BASE_DIR)
        if os.path.isdir(os.path.join(LORA_BASE_DIR, d))
           and os.path.isfile(os.path.join(LORA_BASE_DIR, d, "adapter_config.json"))
    ], key=os.path.getmtime)
    if lora_dirs:
        latest_lora = lora_dirs[-1]
        print(f"\n>>> Building fine-tuned Moonbeam (LoRA: {latest_lora}) ...")
        model_ft, tok_ft, cfg_ft = build_model(lora_dir=latest_lora)
        run(model_ft, tok_ft, cfg_ft, OUT_FINETUNED, "moonbeam_finetuned")
        del model_ft
    else:
        print("No LoRA checkpoints found; skipping fine-tuned generation.")

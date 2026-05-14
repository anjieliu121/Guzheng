#!/usr/bin/env python3
"""
Generate MIDI from Moonbeam (pretrained and LoRA fine-tuned).
Updated for archive/ directory structure.

Run with moonbeam conda env:
  cd /Users/anjie/Documents/MyGuzheng/Guzheng/archive/moonbeam
  PYTHONPATH="src:src/llama_recipes/transformers_minimal/src" \
  /opt/miniconda3/envs/moonbeam/bin/python3 ../../scripts/generate_moonbeam.py
"""

import os, sys, time, random, argparse
import numpy as np
import pandas as pd
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ROOT          = "/Users/anjie/Documents/MyGuzheng/Guzheng"
MOONBEAM_SRC  = os.path.join(ROOT, "archive/moonbeam/src")
TRANSFORMERS  = os.path.join(ROOT, "archive/moonbeam/src/llama_recipes/transformers_minimal/src")
sys.path.insert(0, MOONBEAM_SRC)
sys.path.insert(0, TRANSFORMERS)

CKPT_PATH          = os.path.join(ROOT, "archive/checkpoints/pretrained/moonbeam_309M.pt")
MODEL_CONFIG_PATH  = os.path.join(ROOT, "archive/moonbeam/src/llama_recipes/configs/model_config_small.json")
LORA_BASE_DIR      = os.path.join(ROOT, "archive/checkpoints/finetuned/moonbeam_guzheng_lora")
CSV_FILE           = os.path.join(ROOT, "outputs/moonbeam_preprocessed/train_test_split.csv")
DATA_DIR           = os.path.join(ROOT, "outputs/moonbeam_preprocessed/processed")


def build_model(lora_dir=None):
    from transformers import LlamaConfig, LlamaForCausalLM

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

    from llama_recipes.datasets.music_tokenizer import MusicTokenizer
    tokenizer = MusicTokenizer(
        timeshift_vocab_size  = llama_config.onset_vocab_size,
        dur_vocab_size        = llama_config.dur_vocab_size,
        octave_vocab_size     = llama_config.octave_vocab_size,
        pitch_class_vocab_size= llama_config.pitch_class_vocab_size,
        instrument_vocab_size = llama_config.instrument_vocab_size,
        velocity_vocab_size   = llama_config.velocity_vocab_size,
    )
    return model, tokenizer, llama_config


def sample_top_p(probs, p):
    ps, pi = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(ps, dim=-1)
    ps[cumsum - ps > p] = 0.0
    ps = ps / ps.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    next_tok = torch.multinomial(ps, num_samples=1)
    return torch.gather(pi, -1, next_tok)


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
        if cur_pos % 50 == 0:
            print(f"  Token {cur_pos}/{total_len}")

        out = model.forward(
            input_ids       = tokens[:, prev_pos:cur_pos],
            past_key_values = past_kv,
            use_cache       = True,
            attention_mask  = None,
        )

        dec_tok = (torch.tensor(tokenizer.sos_out, device=DEVICE)
                   .to(tokens)
                   .expand(bsz, 1))
        dec_tok_out = dec_tok

        hidden = out.logits[:, -1, :]

        # Unwrap PeftModel to get the original LlamaForCausalLM
        from peft import PeftModel as _PM
        _unwrapped = model.base_model.model if isinstance(model, _PM) else model

        hidden = (hidden.unsqueeze(0)
                  .expand(_unwrapped.decoder.num_hidden_layers, -1, -1)
                  .contiguous())

        attrs = ["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode",
                 "pitch_dict_decode", "instrument_dict_decode", "velocity_dict_decode"]

        model_for_decode = _unwrapped
        for attr in attrs:
            dec_out = model_for_decode.forward(
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
                masked_lg = torch.full_like(lg, -1e4)
                masked_lg[:, sample_ids] = lg[:, sample_ids]
                probs   = torch.softmax(masked_lg, dim=-1)
                dec_tok = sample_top_p(probs, top_p)
            else:
                si_t    = torch.tensor(sample_ids, device=gen_logits.device)
                lg_last = gen_logits[:, -1, :]
                dec_tok = si_t[lg_last[:, si_t].argmax(dim=-1)].unsqueeze(-1)

            dec_tok_out = torch.cat([dec_tok_out, dec_tok], dim=-1)

        dec_reshaped  = dec_tok_out[:, 1:].unsqueeze(1)
        dec_lang      = tokenizer.convert_from_language_tokens(dec_reshaped).to(DEVICE)

        prev_onset  = tokens[:, cur_pos - 1, 0]
        new_onset   = prev_onset + dec_lang.clone().detach()[:, -1, 0].to(prev_onset)
        next_cmp    = torch.cat([new_onset.unsqueeze(-1),
                                 dec_lang.clone().detach()[:, -1, 1:]], dim=-1).to(tokens)

        next_tok = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_cmp)
        tokens[:, cur_pos] = next_tok

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

    tokens = tokens[:, 1:, :]
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


def load_prompts(tokenizer, n_samples=10, prompt_len=5):
    df = pd.read_csv(CSV_FILE)
    test_files = df[df["split"] == "test"]["file_base_name"].tolist()
    if not test_files:
        test_files = df["file_base_name"].tolist()
    sampled = random.sample(test_files, min(n_samples, len(test_files)))
    prompts = []
    for fn in sampled:
        data = np.load(os.path.join(DATA_DIR, fn))
        encoded = tokenizer.encode_series(data, if_add_sos=True, if_add_eos=False)
        prompts.append(encoded[:prompt_len])
    return prompts


def run(model, tokenizer, llama_config, out_dir, label, n_samples=10,
        max_gen_len=300, temperature=0.85, top_p=0.9):
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_prompts(tokenizer, n_samples=n_samples)
    print(f"\n=== Generating {len(prompts)} samples for {label} ===")

    out_tokens = generate(
        model, tokenizer, prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_len=llama_config.max_len if hasattr(llama_config, "max_len") else 2048,
    )

    saved = 0
    for i, (toks, prompt) in enumerate(zip(out_tokens, prompts)):
        if len(toks) < 2:
            continue
        midi = tokenizer.compound_to_midi(toks)
        path = os.path.join(out_dir, f"{label}_{i:02d}.mid")
        midi.save(path)
        print(f"  Saved {path} ({len(toks)} tokens)")
        saved += 1
    print(f"Saved {saved}/{len(prompts)} MIDI files to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_gen_len", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    # 1. Pretrained (skip if requested)
    if not args.skip_pretrained:
        print("\n>>> Building pretrained Moonbeam ...")
        model_pre, tok_pre, cfg_pre = build_model(lora_dir=None)
        run(model_pre, tok_pre, cfg_pre,
            os.path.join(ROOT, "outputs/moonbeam_pretrained_v2"),
            "moonbeam_pretrained",
            n_samples=args.n_samples,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p)
        del model_pre
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # 2. Fine-tuned (latest LoRA)
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
        run(model_ft, tok_ft, cfg_ft,
            os.path.join(ROOT, "outputs/moonbeam_finetuned_v2"),
            "moonbeam_finetuned",
            n_samples=args.n_samples,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p)
        del model_ft
    else:
        print("No LoRA checkpoints found.")


if __name__ == "__main__":
    main()

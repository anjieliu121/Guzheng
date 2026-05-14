"""Hierarchical Patch-Character GPT for guzheng ABC generation (PatchGPT).

Same architecture as NotaGen (hierarchical patch decoder + character decoder)
but trained FROM SCRATCH on guzheng data only — no pre-training.

This isolates the contribution of NotaGen's pre-training on 1M+ Western scores.
If PatchGPT << NotaGen, pre-training matters.
If PatchGPT >> CharGPT (flat), hierarchy matters.

Architecture (matching NotaGen medium):
  Patch decoder: causal GPT over 16-char patch embeddings
  Char decoder:  causal GPT generating characters within each patch,
                 conditioned on the patch decoder's output

Config presets:
  medium  ~231M params (matches NotaGen medium: d=1024, 16+3 layers)
  small   ~50M  params (d=512, 8+2 layers) — faster training
  tiny    ~10M  params (d=256, 4+2 layers) — quick experiments

Run:
    python test_and_trial_9/train.py                    # medium (default)
    python test_and_trial_9/train.py --config small     # 50M
    python test_and_trial_9/train.py --config tiny      # 10M
    python test_and_trial_9/train.py --smoke            # 2-epoch test
"""

import argparse
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "test_and_trial_7" / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────
PATCH_SIZE = 16
BOS_ID = 1      # begin of sequence token
EOS_ID = 2      # end of sequence token
PAD_ID = 0      # padding / special token
VOCAB_SIZE = 128  # ASCII

# ── Args ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="quick 2-epoch test")
ap.add_argument("--config", choices=["medium", "small", "tiny"], default="medium")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--batch_size", type=int, default=None,
                help="override batch size (default: auto per config)")
ap.add_argument("--max_patches", type=int, default=256,
                help="max patches per sequence")
ap.add_argument("--grad_accum", type=int, default=1,
                help="gradient accumulation steps")
args = ap.parse_args()

# Config presets
CONFIGS = {
    "medium": dict(d_model=1024, patch_layers=16, char_layers=3, d_ff=4096,
                   n_heads=16, batch_size=1, dropout=0.1),
    "small":  dict(d_model=512,  patch_layers=8,  char_layers=2, d_ff=2048,
                   n_heads=8,  batch_size=4, dropout=0.1),
    "tiny":   dict(d_model=256,  patch_layers=4,  char_layers=2, d_ff=1024,
                   n_heads=4,  batch_size=8, dropout=0.1),
}
CFG = CONFIGS[args.config]
if args.batch_size is not None:
    CFG["batch_size"] = args.batch_size

# ── Reproducibility ───────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Device ────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")


# ══════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════

class PatchDecoder(nn.Module):
    """Causal transformer over patch-level embeddings.

    Each patch (16 chars) is one-hot encoded and linearly projected to d_model,
    then processed by a causal transformer. Same as NotaGen's PatchLevelDecoder.
    """
    def __init__(self, d_model, n_heads, n_layers, d_ff, max_patches, dropout):
        super().__init__()
        self.patch_proj = nn.Linear(PATCH_SIZE * VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_patches, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, patches):
        """patches: (B, N, PATCH_SIZE) int tensor with values 0-127."""
        B, N, P = patches.shape
        # One-hot → linear projection
        one_hot = F.one_hot(patches.long(), num_classes=VOCAB_SIZE).float()
        one_hot = one_hot.view(B, N, P * VOCAB_SIZE)
        h = self.patch_proj(one_hot)

        pos = torch.arange(N, device=patches.device).unsqueeze(0)
        h = self.drop(h + self.pos_emb(pos))

        mask = nn.Transformer.generate_square_subsequent_mask(N, device=patches.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.ln_f(h)  # (B, N, d_model)


class CharDecoder(nn.Module):
    """Generates characters within a patch, conditioned on the patch embedding.

    Same as NotaGen's CharLevelDecoder: the encoded patch replaces the first
    token embedding, then the model predicts characters autoregressively.
    """
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(PATCH_SIZE + 1, d_model)  # +1 for BOS position
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, encoded_patch, target_chars):
        """Training forward pass.

        encoded_patch: (B, d_model) — output from patch decoder
        target_chars:  (B, PATCH_SIZE) — ground truth characters (0-127)

        Input to transformer: [encoded_patch, emb(c0), emb(c1), ..., emb(c_{P-2})]
        Predicts:              [c0, c1, ..., c_{P-1}]
        """
        B, P = target_chars.shape
        char_embs = self.token_emb(target_chars)  # (B, P, d_model)
        # Shift right: [enc_patch, c0, c1, ..., c_{P-2}]
        input_embs = torch.cat([
            encoded_patch.unsqueeze(1),  # (B, 1, d_model)
            char_embs[:, :-1, :],        # (B, P-1, d_model)
        ], dim=1)  # (B, P, d_model)

        pos = torch.arange(P, device=target_chars.device).unsqueeze(0)
        h = self.drop(input_embs + self.pos_emb(pos))

        mask = nn.Transformer.generate_square_subsequent_mask(P, device=target_chars.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        logits = self.head(self.ln_f(h))  # (B, P, VOCAB_SIZE)

        # Mask padding positions in target
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                               target_chars.view(-1), ignore_index=PAD_ID)
        return logits, loss

    @torch.no_grad()
    def generate_patch(self, encoded_patch, temperature=1.2, top_k=9, top_p=0.9):
        """Generate one 16-char patch autoregressively."""
        self.eval()
        dev = encoded_patch.device
        # Start with just the encoded_patch
        input_embs = encoded_patch.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        chars = []
        for i in range(PATCH_SIZE):
            T = i + 1
            pos = torch.arange(T, device=dev).unsqueeze(0)
            h = input_embs + self.pos_emb(pos[:, :T])
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=dev)
            h = self.transformer(h, mask=mask, is_causal=True)
            logits = self.head(self.ln_f(h[0, -1:])) / temperature  # (1, VOCAB)

            if top_k > 0:
                v, _ = torch.topk(logits[0], min(top_k, logits.size(-1)))
                logits[0][logits[0] < v[-1]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float('-inf')
                logits[0] = sorted_logits.scatter(0, sorted_idx, sorted_logits)

            probs = F.softmax(logits[0], dim=-1)
            tok = torch.multinomial(probs, 1).item()

            chars.append(tok)
            if tok == EOS_ID:
                break

            tok_emb = self.token_emb(torch.tensor([tok], device=dev)).unsqueeze(0)
            input_embs = torch.cat([input_embs, tok_emb], dim=1)

        return chars


class PatchGPT(nn.Module):
    """Hierarchical Patch + Character GPT.

    Architecture matches NotaGen: patch-level decoder encodes sequences of
    16-char patches, character-level decoder generates characters within
    each patch conditioned on the patch encoding.

    Trained end-to-end from scratch (no pre-training).
    """
    def __init__(self, d_model=1024, n_heads=16,
                 patch_layers=16, char_layers=3, d_ff=4096,
                 max_patches=256, dropout=0.1):
        super().__init__()
        self.patch_decoder = PatchDecoder(
            d_model=d_model, n_heads=n_heads, n_layers=patch_layers,
            d_ff=d_ff, max_patches=max_patches, dropout=dropout)
        self.char_decoder = CharDecoder(
            d_model=d_model, n_heads=n_heads, n_layers=char_layers,
            d_ff=d_ff, dropout=dropout)

    def forward(self, patches):
        """
        patches: (B, N, PATCH_SIZE) — sequence of patches per sample
        Returns: scalar loss
        """
        B, N, P = patches.shape

        # Encode all patches
        encoded = self.patch_decoder(patches)  # (B, N, d_model)

        # Autoregressive: encoded[i] predicts patches[i+1]
        enc_shifted = encoded[:, :-1, :]  # (B, N-1, d_model)
        tgt_shifted = patches[:, 1:, :]   # (B, N-1, P)

        # Flatten batch and patch dimensions
        B2 = B * (N - 1)
        enc_flat = enc_shifted.reshape(B2, -1)
        tgt_flat = tgt_shifted.reshape(B2, P)

        # Filter padding patches (all zeros)
        valid = tgt_flat.sum(dim=-1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=patches.device, requires_grad=True)

        _, loss = self.char_decoder(enc_flat[valid], tgt_flat[valid])
        return loss

    @torch.no_grad()
    def generate(self, prompt_patches, max_patches=128,
                 temperature=1.2, top_k=9, top_p=0.9, window=64):
        """Auto-regressive patch-by-patch generation with sliding window.

        prompt_patches: list of patches (each a list of PATCH_SIZE ints)
        window: max patches to feed through patch decoder (for speed)
        Returns: list of generated patches (lists of ints)
        """
        self.eval()
        dev = next(self.parameters()).device
        patches = [list(p) for p in prompt_patches]

        for _ in range(max_patches):
            # Sliding window: only encode last `window` patches
            context = patches[-window:]
            inp = torch.tensor([context], device=dev)  # (1, W, P)
            encoded = self.patch_decoder(inp)  # (1, W, d_model)
            last_enc = encoded[0, -1]  # (d_model,)

            # Generate next patch character by character
            new_patch = self.char_decoder.generate_patch(
                last_enc, temperature=temperature, top_k=top_k, top_p=top_p)

            # Check for EOS patch (starts with BOS, rest EOS)
            if len(new_patch) >= 2 and new_patch[0] == BOS_ID and new_patch[1] == EOS_ID:
                break
            # Check if first char is EOS
            if len(new_patch) > 0 and new_patch[0] == EOS_ID:
                break

            # Pad to PATCH_SIZE
            while len(new_patch) < PATCH_SIZE:
                new_patch.append(PAD_ID)
            new_patch = new_patch[:PATCH_SIZE]
            patches.append(new_patch)

        return patches


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

from abctoolkit.transpose import Key2index
Index2Key = {idx: k for k, idx in Key2index.items() if idx not in [1, 11]}


def text_to_patches(text):
    """Convert ABC text → list of patches (each = list of PATCH_SIZE ints).

    Patches are created by splitting the text into 16-char chunks.
    BOS patch is prepended, EOS patch is appended.
    """
    ids = [ord(c) for c in text if ord(c) < VOCAB_SIZE]
    patches = []
    # BOS patch: 15 × BOS + EOS (same as NotaGen)
    patches.append([BOS_ID] * 15 + [EOS_ID])
    # Content patches
    for i in range(0, len(ids), PATCH_SIZE):
        chunk = ids[i:i + PATCH_SIZE]
        if len(chunk) < PATCH_SIZE:
            chunk = chunk + [PAD_ID] * (PATCH_SIZE - len(chunk))
        patches.append(chunk)
    # EOS patch: BOS + 15 × EOS
    patches.append([BOS_ID] + [EOS_ID] * 15)
    return patches


class PatchABCDataset(torch.utils.data.Dataset):
    """Patch-level dataset for PatchGPT with key augmentation."""

    def __init__(self, entries, max_patches, is_train=True):
        self.entries = entries
        self.max_patches = max_patches
        self.is_train = is_train

    def __len__(self):
        return len(self.entries)

    def _load_abc(self, entry, key):
        folder = os.path.dirname(entry["path"])
        name = os.path.basename(entry["path"])
        path = os.path.join(folder, key, name + "_" + key + ".abc")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _random_key(self, ori_key):
        """NotaGen-style triangular key augmentation ±3 semitones."""
        ori_idx = Key2index.get(ori_key, 0)
        offsets = list(range(-3, 4))
        weights = [1, 2, 3, 4, 3, 2, 1]
        des_idx = (ori_idx + random.choices(offsets, weights=weights)[0]) % 12
        if des_idx == 1:
            return "Db" if random.random() < 0.8 else "C#"
        elif des_idx == 11:
            return "B" if random.random() < 0.8 else "Cb"
        elif des_idx == 6:
            return "F#" if random.random() < 0.5 else "Gb"
        else:
            return Index2Key.get(des_idx, "C")

    def __getitem__(self, idx):
        entry = self.entries[idx]
        key = self._random_key(entry["key"]) if self.is_train else entry["key"]

        try:
            text = self._load_abc(entry, key)
        except FileNotFoundError:
            text = self._load_abc(entry, entry["key"])

        patches = text_to_patches(text)

        # Truncate or random crop to max_patches
        if len(patches) > self.max_patches:
            if self.is_train:
                start = random.randint(0, len(patches) - self.max_patches)
                patches = patches[start:start + self.max_patches]
            else:
                patches = patches[:self.max_patches]

        # Pad to max_patches
        while len(patches) < self.max_patches:
            patches.append([PAD_ID] * PATCH_SIZE)

        return torch.tensor(patches, dtype=torch.long)  # (max_patches, PATCH_SIZE)


# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    # ── Data ──────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(REPO / "test_and_trial_7" / "NotaGen" / "finetune"))

    train_jsonl = DATA / "abc_weighted_train.jsonl"
    eval_jsonl = DATA / "abc_weighted_eval.jsonl"

    with open(train_jsonl) as f:
        train_entries = [json.loads(l) for l in f]
    with open(eval_jsonl) as f:
        eval_entries = [json.loads(l) for l in f]

    if args.smoke:
        train_entries = train_entries[:10]
        eval_entries = eval_entries[:3]
        args.epochs = 2

    print(f"[data] train={len(train_entries)}  eval={len(eval_entries)}")

    # WeightedRandomSampler: oversample repertoire 5x
    _t99 = re.compile(r"guzheng_(train|test)_\d+$")
    weights = [1.0 if _t99.search(os.path.basename(e["path"])) else 5.0
               for e in train_entries]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(train_entries), replacement=True)

    train_set = PatchABCDataset(train_entries, args.max_patches, is_train=True)
    eval_set = PatchABCDataset(eval_entries, args.max_patches, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=0, pin_memory=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────
    model = PatchGPT(
        d_model=CFG["d_model"],
        n_heads=CFG["n_heads"],
        patch_layers=CFG["patch_layers"],
        char_layers=CFG["char_layers"],
        d_ff=CFG["d_ff"],
        max_patches=args.max_patches,
        dropout=CFG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PatchGPT-{args.config} params: {n_params/1e6:.2f}M")
    print(f"        patch decoder: d={CFG['d_model']}, layers={CFG['patch_layers']}, "
          f"heads={CFG['n_heads']}, ff={CFG['d_ff']}")
    print(f"        char  decoder: d={CFG['d_model']}, layers={CFG['char_layers']}, "
          f"heads={CFG['n_heads']}, ff={CFG['d_ff']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    # ── Resume ────────────────────────────────────────────────────────
    EXP = f"patchgpt_{args.config}"
    if args.smoke:
        EXP += "_smoke"
    ckpt_path = CKPT_DIR / f"{EXP}_latest.pth"
    best_path = CKPT_DIR / f"{EXP}_best.pth"
    log_path = LOGS_DIR / f"{EXP}.log"

    start_epoch = 0
    best_eval = float("inf")

    if ckpt_path.exists():
        print(f"[resume] loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_eval = ckpt.get("best_eval", float("inf"))
        print(f"[resume] epoch {start_epoch}, best_eval={best_eval:.4f}")

    # ── LR schedule ───────────────────────────────────────────────────
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)
    global_step = start_epoch * len(train_loader)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"[start] epochs {start_epoch+1}..{args.epochs}, "
          f"{len(train_loader)} batches/epoch, batch_size={CFG['batch_size']}")
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {time.asctime()} device={device} config={args.config} "
                   f"params={n_params/1e6:.2f}M ===\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        optimizer.zero_grad(set_to_none=True)

        for i, patches in enumerate(train_loader):
            patches = patches.to(device)  # (B, max_patches, PATCH_SIZE)

            loss = model(patches)
            loss = loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                # LR schedule
                lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item() * args.grad_accum
            train_n += 1
            global_step += 1

            if (i + 1) % 10 == 0:
                avg = train_loss_sum / train_n
                print(f"  [train] {i+1}/{len(train_loader)} loss={avg:.4f} lr={lr:.2e}")

        train_loss = train_loss_sum / max(train_n, 1)

        # ── Eval ──────────────────────────────────────────────────────
        model.eval()
        eval_loss_sum = 0.0
        eval_n = 0
        with torch.no_grad():
            for patches in eval_loader:
                patches = patches.to(device)
                loss = model(patches)
                eval_loss_sum += loss.item()
                eval_n += 1
        eval_loss = eval_loss_sum / max(eval_n, 1)

        elapsed = time.time() - t0
        is_best = eval_loss < best_eval
        if is_best:
            best_eval = eval_loss

        # ── Save ──────────────────────────────────────────────────────
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_eval": best_eval,
            "eval_loss": eval_loss,
            "train_loss": train_loss,
            "config": args.config,
            "d_model": CFG["d_model"],
            "patch_layers": CFG["patch_layers"],
            "char_layers": CFG["char_layers"],
            "d_ff": CFG["d_ff"],
            "n_heads": CFG["n_heads"],
            "max_patches": args.max_patches,
        }
        torch.save(payload, ckpt_path)
        if is_best:
            torch.save(payload, best_path)

        msg = (f"epoch {epoch:3d}  train={train_loss:.4f}  eval={eval_loss:.4f}  "
               f"best={best_eval:.4f}{' *' if is_best else ''}  {elapsed:.1f}s")
        print(msg)
        with open(log_path, "a") as logf:
            logf.write(msg + "\n")

        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

    print(f"[done] best eval: {best_eval:.4f}")


if __name__ == "__main__":
    main()

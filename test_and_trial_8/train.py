"""Character-level GPT-2 baseline for guzheng ABC generation.

A flat (non-hierarchical) transformer trained from scratch on the same
ABC data that NotaGen uses. This isolates the contribution of NotaGen's
pre-training and patch-level hierarchy.

Architecture: standard GPT-2 (decoder-only transformer)
Input: raw ABC text as character sequences (vocab = 128 ASCII)
~3.5M parameters to match the MIDI decoder-only transformer baseline.

Run:
    python test_and_trial_8/train.py
    python test_and_trial_8/train.py --smoke
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

# ---- paths ----
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "test_and_trial_7" / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ---- args ----
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="quick 2-epoch test")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--context_len", type=int, default=512,
                help="training context length in characters")
ap.add_argument("--stride", type=int, default=256,
                help="sliding window stride for chunking")
args = ap.parse_args()

# ---- reproducibility ----
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---- device ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")


# ==============================================================================
# Model: Character-level GPT-2
# ==============================================================================

class CharGPT(nn.Module):
    """Decoder-only transformer for character-level ABC generation.

    ~3.45M params with default config to match the MIDI transformer baseline.
    """
    def __init__(self, vocab_size=128, d_model=256, n_heads=4, n_layers=6,
                 d_ff=256, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        """
        x: [batch, seq_len] of token ids (0..127)
        targets: [batch, seq_len] for cross-entropy loss
        """
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))

        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        logits = self.head(h)  # [B, T, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=4096, temperature=1.0,
                 top_k=50, top_p=0.95, window=512):
        """Auto-regressive character generation with sliding window for speed.

        Uses a fixed window of the last `window` tokens as context instead of
        the full sequence, making each step O(window) instead of O(seq_len).
        """
        self.eval()
        dev = next(self.parameters()).device
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            # Use only last `window` tokens for speed
            context = tokens[-window:]
            x = torch.tensor([context], device=dev)

            B, T = x.shape
            pos = torch.arange(T, device=dev).unsqueeze(0)
            h = self.token_emb(x) + self.pos_emb(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=dev)
            h = self.transformer(h, mask=mask, is_causal=True)
            logits = self.head(self.ln_f(h[0, -1:]))  # only last position
            logits = logits[0] / temperature

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float('-inf')

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float('-inf')
                logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1).item()

            if tok == 2:  # EOS
                break
            tokens.append(tok)
        return tokens


# ==============================================================================
# Dataset
# ==============================================================================

# Key augmentation: same as NotaGen — triangular probability ±3 semitones
from abctoolkit.transpose import Key2index
Index2Key = {idx: k for k, idx in Key2index.items() if idx not in [1, 11]}


class ABCCharDataset(torch.utils.data.Dataset):
    """Character-level dataset with sliding window chunking and key augmentation."""

    def __init__(self, entries, context_len, stride, is_train=True):
        self.entries = entries
        self.context_len = context_len
        self.stride = stride
        self.is_train = is_train
        # Pre-build chunks: (entry_idx, key) pairs
        # Actual chunking happens in __getitem__ since key is random
        self.indices = list(range(len(entries)))

    def __len__(self):
        # Each entry can produce multiple chunks, but for simplicity
        # we sample one random chunk per entry per epoch
        return len(self.entries)

    def _load_abc(self, entry, key):
        # Path format: abc_augmented/<KEY>/<name>_<KEY>.abc
        # entry["path"] = ".../abc_augmented/<name>"
        folder = os.path.dirname(entry["path"])  # .../abc_augmented
        name = os.path.basename(entry["path"])     # piece name
        path = os.path.join(folder, key, name + "_" + key + ".abc")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _random_key(self, ori_key):
        """NotaGen-style key augmentation: triangular ±3 semitones."""
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
            # Fallback to original key
            text = self._load_abc(entry, entry["key"])

        # Encode: BOS + ascii chars + EOS
        ids = [1] + [ord(c) for c in text if ord(c) < 128] + [2]

        # Random chunk for training, first chunk for eval
        if len(ids) <= self.context_len + 1:
            chunk = ids
        elif self.is_train:
            start = random.randint(0, max(0, len(ids) - self.context_len - 1))
            chunk = ids[start:start + self.context_len + 1]
        else:
            chunk = ids[:self.context_len + 1]

        # Pad if needed
        if len(chunk) < self.context_len + 1:
            chunk = chunk + [0] * (self.context_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ==============================================================================
# Training
# ==============================================================================

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-6):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    # ---- data ----
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

    # WeightedRandomSampler: oversample repertoire 5x (same as NotaGen weighted)
    _t99 = re.compile(r"guzheng_(train|test)_\d+$")
    weights = [1.0 if _t99.search(os.path.basename(e["path"])) else 5.0
               for e in train_entries]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(train_entries), replacement=True)

    train_set = ABCCharDataset(train_entries, args.context_len, args.stride, is_train=True)
    eval_set = ABCCharDataset(eval_entries, args.context_len, args.stride, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)

    # ---- model ----
    model = CharGPT(
        vocab_size=128,
        d_model=256,
        n_heads=4,
        n_layers=6,
        d_ff=256,
        max_seq_len=4096,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CharGPT params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=0.01)

    # ---- resume ----
    EXP = "chargpt_smoke" if args.smoke else "chargpt_weighted"
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

    # ---- LR schedule ----
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)

    global_step = start_epoch * len(train_loader)

    # ---- training ----
    print(f"[start] epochs {start_epoch+1}..{args.epochs}, {len(train_loader)} batches/epoch")
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {time.asctime()} device={device} ===\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # LR schedule
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_n += 1
            global_step += 1

            if (i + 1) % 10 == 0:
                print(f"  [train] {i+1}/{len(train_loader)} loss={train_loss_sum/train_n:.4f} lr={lr:.2e}")

        train_loss = train_loss_sum / max(train_n, 1)

        # Eval
        model.eval()
        eval_loss_sum = 0.0
        eval_n = 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                eval_loss_sum += loss.item()
                eval_n += 1
        eval_loss = eval_loss_sum / max(eval_n, 1)

        elapsed = time.time() - t0
        is_best = eval_loss < best_eval
        if is_best:
            best_eval = eval_loss

        # Save
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_eval": best_eval,
            "eval_loss": eval_loss,
            "train_loss": train_loss,
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

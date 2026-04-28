"""Fine-tune NotaGen on the bundled guzheng dataset.

Reads training data from ../data/dataset.jsonl (one JSON line per piece, with all
15 key augmentations inlined). Repertoire pieces are oversampled 5x relative to
guzheng_tech99 pieces via WeightedRandomSampler (using the per-row `weight` field).

Runs on Apple MPS / CUDA / CPU. fp32 only (bf16 unstable on MPS).
Saves a checkpoint every epoch so a crash doesn't lose hours; resumes if one exists.

Run:
    python notagen/train.py --size small
    python notagen/train.py --size medium --epochs 30
    python notagen/train.py --smoke              # 2-epoch smoke test on 20 pieces
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
NOTAGEN = ROOT / "NotaGen"
DATA_FILE = REPO / "data" / "dataset.jsonl"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["small", "medium", "large"], default="small")
ap.add_argument("--smoke", action="store_true",
                help="2-epoch smoke test on the first 20 train pieces")
ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args()

# Architectures match NotaGen's pretrained checkpoints exactly.
SIZE_CONFIGS = {
    "small":  dict(PATCH_NUM_LAYERS=12, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=768,
                   pretrained_wpe=2048, ckpt="notagen_small_pretrain.pth"),
    "medium": dict(PATCH_NUM_LAYERS=16, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=1024,
                   pretrained_wpe=2048, ckpt="notagen_medium_pretrain.pth"),
    "large":  dict(PATCH_NUM_LAYERS=20, CHAR_NUM_LAYERS=6, HIDDEN_SIZE=1280,
                   pretrained_wpe=1024, ckpt="notagen_large_pretrain.pth"),
}
sc = SIZE_CONFIGS[args.size]

sys.path.insert(0, str(NOTAGEN / "finetune"))
import config
config.PATCH_STREAM = True
config.PATCH_SIZE = 16
config.PATCH_LENGTH = 1024
config.CHAR_NUM_LAYERS = sc["CHAR_NUM_LAYERS"]
config.PATCH_NUM_LAYERS = sc["PATCH_NUM_LAYERS"]
config.HIDDEN_SIZE = sc["HIDDEN_SIZE"]
config.BATCH_SIZE = 1
config.LEARNING_RATE = 1e-5
config.NUM_EPOCHS = args.epochs
config.ACCUMULATION_STEPS = 1
config.PATCH_SAMPLING_BATCH_SIZE = 0
config.LOAD_FROM_CHECKPOINT = False
config.WANDB_LOGGING = False
config.PRETRAINED_PATH = str(CKPT_DIR / sc["ckpt"])
config.EXP_TAG = f"guzheng_{args.size}"

from utils import Patchilizer, NotaGenLMHeadModel
from transformers import GPT2Config, get_constant_schedule_with_warmup
from abctoolkit.transpose import Key2index

Index2Key = {idx: k for k, idx in Key2index.items() if idx not in [1, 11]}

if args.smoke:
    config.NUM_EPOCHS = 2
    SUBSET = 20
    EXP = f"smoke_{args.size}"
else:
    SUBSET = None
    EXP = f"guzheng_{args.size}"

CKPT_PATH = CKPT_DIR / f"notagen_{EXP}_latest.pth"
BEST_PATH = CKPT_DIR / f"notagen_{EXP}_best.pth"
LOG_PATH = LOGS_DIR / f"notagen_{EXP}.log"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def load_dataset(path):
    """Read dataset.jsonl. Returns (train_rows, eval_rows). Each row has:
    name, original_key, source, split, weight, abc: {key: text}."""
    train, eval_ = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            (train if row["split"] == "train" else eval_).append(row)
    return train, eval_


class GuzhengDataset(torch.utils.data.Dataset):
    """Picks a random key per sample (NotaGen-style triangular ±3 semitone window)."""
    def __init__(self, rows, patchilizer):
        self.rows = rows
        self.patchilizer = patchilizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ori_idx = Key2index.get(row["original_key"], 0)
        offsets = list(range(-3, 4))
        weights = [1, 2, 3, 4, 3, 2, 1]
        des_idx = (ori_idx + random.choices(offsets, weights=weights)[0]) % 12
        if des_idx == 1:
            des_key = "Db" if random.random() < 0.8 else "C#"
        elif des_idx == 11:
            des_key = "B" if random.random() < 0.8 else "Cb"
        elif des_idx == 6:
            des_key = "F#" if random.random() < 0.5 else "Gb"
        else:
            des_key = Index2Key[des_idx]

        text = row["abc"][des_key]
        tokens = self.patchilizer.encode_train(text)
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor([1] * len(tokens), dtype=torch.long))


def collate(batch):
    patches, masks = zip(*batch)
    patches = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return patches.to(device), masks.to(device)


patchilizer = Patchilizer()

patch_cfg = GPT2Config(num_hidden_layers=config.PATCH_NUM_LAYERS,
                       max_length=config.PATCH_LENGTH,
                       max_position_embeddings=config.PATCH_LENGTH,
                       n_embd=config.HIDDEN_SIZE,
                       num_attention_heads=config.HIDDEN_SIZE // 64,
                       vocab_size=1)
char_cfg = GPT2Config(num_hidden_layers=config.CHAR_NUM_LAYERS,
                      max_length=config.PATCH_SIZE + 1,
                      max_position_embeddings=config.PATCH_SIZE + 1,
                      hidden_size=config.HIDDEN_SIZE,
                      num_attention_heads=config.HIDDEN_SIZE // 64,
                      vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_cfg, decoder_config=char_cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] params: {n_params/1e6:.1f} M")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)

start_epoch = 0
best_eval = float("inf")

if CKPT_PATH.exists():
    print(f"[resume] loading {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"]
    best_eval = ckpt.get("best_eval", float("inf"))
    print(f"[resume] starting from epoch {start_epoch+1}, best_eval={best_eval:.4f}")
else:
    print(f"[init] loading pretrained {config.PRETRAINED_PATH}")
    ckpt = torch.load(config.PRETRAINED_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    # Small/medium were pretrained at PATCH_LENGTH=2048 — truncate wpe to ours.
    wpe_key = "patch_level_decoder.base.wpe.weight"
    if wpe_key in sd and sd[wpe_key].shape[0] > config.PATCH_LENGTH:
        print(f"[init] truncating wpe {sd[wpe_key].shape[0]} -> {config.PATCH_LENGTH}")
        sd[wpe_key] = sd[wpe_key][: config.PATCH_LENGTH].clone()
    model.load_state_dict(sd)
    print(f"[init] pretrained loaded (epoch {ckpt['epoch']}, loss {ckpt.get('min_eval_loss')})")

train_rows, eval_rows = load_dataset(DATA_FILE)
if SUBSET is not None:
    train_rows = train_rows[:SUBSET]
    eval_rows = eval_rows[:max(2, SUBSET // 10)]
print(f"[data] train={len(train_rows)}  eval={len(eval_rows)}")

train_set = GuzhengDataset(train_rows, patchilizer)
eval_set = GuzhengDataset(eval_rows, patchilizer)

# Per-row weights from dataset.jsonl (5.0 for repertoire, 1.0 for tech99).
sampler_weights = [r["weight"] for r in train_rows]
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sampler_weights, num_samples=len(train_rows), replacement=True)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=config.BATCH_SIZE, collate_fn=collate, sampler=sampler)
eval_loader = torch.utils.data.DataLoader(
    eval_set, batch_size=config.BATCH_SIZE, collate_fn=collate, shuffle=False)
n_rep = sum(1 for w in sampler_weights if w > 1.0)
print(f"[sampler] rep={n_rep}@{max(sampler_weights):g}x, tech99={len(sampler_weights)-n_rep}@1x")


def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n = 0
    t0 = time.time()
    for i, (patches, masks) in enumerate(loader):
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss = model(patches, masks).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                loss = model(patches, masks).loss
        total += loss.item()
        n += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{'train' if train else 'eval '}] {i+1}/{len(loader)}  "
                  f"loss={total/n:.4f}  {elapsed/(i+1):.2f}s/step")
    return total / max(n, 1)


def save_ckpt(path, epoch, eval_loss, best=False):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_eval": best_eval,
        "eval_loss": eval_loss,
    }
    torch.save(payload, path)
    if best:
        torch.save(payload, BEST_PATH)


print(f"[start] epochs {start_epoch+1}..{config.NUM_EPOCHS}")
with open(LOG_PATH, "a") as logf:
    logf.write(f"\n=== run started {time.asctime()} device={device} "
               f"epochs {start_epoch+1}..{config.NUM_EPOCHS} ===\n")

for epoch in range(start_epoch + 1, config.NUM_EPOCHS + 1):
    print(f"---------- epoch {epoch}/{config.NUM_EPOCHS} ----------")
    t0 = time.time()
    train_loss = run_epoch(train_loader, train=True)
    eval_loss = run_epoch(eval_loader, train=False)
    elapsed = time.time() - t0

    is_best = eval_loss < best_eval
    if is_best:
        best_eval = eval_loss
    save_ckpt(CKPT_PATH, epoch, eval_loss, best=is_best)

    msg = (f"epoch {epoch:3d}  train={train_loss:.4f}  eval={eval_loss:.4f}  "
           f"best={best_eval:.4f}{' *' if is_best else ''}  {elapsed:.1f}s")
    print(msg)
    with open(LOG_PATH, "a") as logf:
        logf.write(msg + "\n")
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

print(f"[done] best eval loss: {best_eval:.4f}")

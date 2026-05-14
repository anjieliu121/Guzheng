"""Fine-tune NotaGen-small on guzheng ABC, MPS-compatible.

NotaGen's stock train-gen.py is hardcoded for CUDA + DDP + fp16 autocast.
This script:
  - runs on Apple MPS (or CPU fallback)
  - fp32 only (bf16 unstable on MPS as of torch 2.5)
  - no distributed
  - saves a checkpoint EVERY epoch (not only on best) so a crash doesn't lose hours
  - resumes from the latest epoch checkpoint if one exists
  - logs to a plain text file

Run:
  python test_and_trial_7/02_finetune.py
or smoke test:
  python test_and_trial_7/02_finetune.py --smoke
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

# ---- paths ----
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
NOTAGEN = ROOT / "NotaGen"
DATA = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ---- args ----
ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["small", "medium", "large"], default="small")
ap.add_argument("--smoke", action="store_true",
                help="run only on the first 20 train files for 2 epochs")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--exp", type=str, default=None,
                help="experiment tag; also picks data/abc_<exp>_{train,eval}.jsonl. "
                     "default uses abc_augmented_{train,eval}.jsonl and tag 'guzheng'")
args = ap.parse_args()

# Architecture specs match NotaGen's pretrained checkpoints exactly.
# Small/medium were pretrained at PATCH_LENGTH=2048; large at 1024.
# We use 1024 for all (truncate wpe for small/medium) — our max patch count is 984.
SIZE_CONFIGS = {
    "small":  dict(PATCH_NUM_LAYERS=12, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=768,
                   pretrained_wpe=2048, ckpt="notagen_small_pretrain.pth"),
    "medium": dict(PATCH_NUM_LAYERS=16, CHAR_NUM_LAYERS=3, HIDDEN_SIZE=1024,
                   pretrained_wpe=2048, ckpt="notagen_medium_pretrain.pth"),
    "large":  dict(PATCH_NUM_LAYERS=20, CHAR_NUM_LAYERS=6, HIDDEN_SIZE=1280,
                   pretrained_wpe=1024, ckpt="notagen_large_pretrain.pth"),
}
sc = SIZE_CONFIGS[args.size]

# ---- inject NotaGen config BEFORE importing utils ----
sys.path.insert(0, str(NOTAGEN / "finetune"))
import config
config.PATCH_STREAM = True
config.PATCH_SIZE = 16
config.PATCH_LENGTH = 1024            # max patch count in our data is 984
config.CHAR_NUM_LAYERS = sc["CHAR_NUM_LAYERS"]
config.PATCH_NUM_LAYERS = sc["PATCH_NUM_LAYERS"]
config.HIDDEN_SIZE = sc["HIDDEN_SIZE"]
config.BATCH_SIZE = 1
config.LEARNING_RATE = 1e-5
config.NUM_EPOCHS = 30
config.ACCUMULATION_STEPS = 1
config.PATCH_SAMPLING_BATCH_SIZE = 0
config.LOAD_FROM_CHECKPOINT = False
config.WANDB_LOGGING = False
config.PRETRAINED_PATH = str(CKPT_DIR / sc["ckpt"])
_data_tag = args.exp if args.exp else "augmented"
config.DATA_TRAIN_INDEX_PATH = str(DATA / f"abc_{_data_tag}_train.jsonl")
config.DATA_EVAL_INDEX_PATH = str(DATA / f"abc_{_data_tag}_eval.jsonl")
config.EXP_TAG = f"guzheng_{args.size}"

from utils import Patchilizer, NotaGenLMHeadModel  # noqa: E402
from transformers import GPT2Config, get_constant_schedule_with_warmup  # noqa: E402
from abctoolkit.transpose import Key2index, Key2Mode  # noqa: E402

Index2Key = {idx: k for k, idx in Key2index.items() if idx not in [1, 11]}
Mode2Key = {mode: k for k, modes in Key2Mode.items() for mode in modes}


_exp_suffix = f"_{args.exp}" if args.exp else ""
if args.smoke:
    config.NUM_EPOCHS = 2
    SUBSET = 20
    EXP = f"smoke_{args.size}{_exp_suffix}"
else:
    config.NUM_EPOCHS = args.epochs
    SUBSET = None
    EXP = f"guzheng_{args.size}{_exp_suffix}"

CKPT_PATH = CKPT_DIR / f"notagen_{EXP}_latest.pth"
BEST_PATH = CKPT_DIR / f"notagen_{EXP}_best.pth"
LOG_PATH = LOGS_DIR / f"notagen_{EXP}.log"

# ---- device ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ---- repro ----
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---- data ----
class GuzhengDataset(torch.utils.data.Dataset):
    """Picks a random key per sample (NotaGen's key augmentation)."""
    def __init__(self, files, patchilizer):
        self.files = files
        self.patchilizer = patchilizer
        self.keys = list(Key2index.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        entry = self.files[idx]
        ori_key = entry["key"]
        ori_idx = Key2index.get(ori_key, 0)
        # 7-step nearby key window with triangular probability (matches NotaGen)
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

        folder = os.path.dirname(entry["path"])
        name = os.path.basename(entry["path"])
        path = os.path.join(folder, des_key, name + "_" + des_key + ".abc")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = self.patchilizer.encode_train(text)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor([1] * len(tokens), dtype=torch.long)


def collate(batch):
    patches, masks = zip(*batch)
    patches = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return patches.to(device), masks.to(device)


# ---- model ----
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

# ---- load weights ----
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

# ---- data loaders ----
with open(config.DATA_TRAIN_INDEX_PATH) as f:
    train_files = [json.loads(l) for l in f]
with open(config.DATA_EVAL_INDEX_PATH) as f:
    eval_files = [json.loads(l) for l in f]

if SUBSET is not None:
    train_files = train_files[:SUBSET]
    eval_files = eval_files[:max(2, SUBSET // 10)]

print(f"[data] train={len(train_files)}  eval={len(eval_files)}")

train_set = GuzhengDataset(train_files, patchilizer)
eval_set = GuzhengDataset(eval_files, patchilizer)

# For --exp weighted: oversample repertoire with WeightedRandomSampler
# (keeps epoch length the same, so memory footprint matches baseline).
import re as _re
_t99 = _re.compile(r"guzheng_(train|test)_\d+$")
if args.exp == "weighted":
    weights = [1.0 if _t99.search(e["path"].split("/")[-1]) else 5.0
               for e in train_files]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(train_files), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE,
                                               collate_fn=collate, sampler=sampler)
    n_rep = sum(1 for w in weights if w > 1.0)
    print(f"[weighted] sampler weights: rep={n_rep}@5x, t99={len(weights)-n_rep}@1x")
else:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE,
                                               collate_fn=collate, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config.BATCH_SIZE,
                                          collate_fn=collate, shuffle=False)


# ---- train / eval loops ----
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


# ---- main loop ----
print(f"[start] epochs {start_epoch+1}..{config.NUM_EPOCHS}")
with open(LOG_PATH, "a") as logf:
    logf.write(f"\n=== run started {time.asctime()} device={device} epochs {start_epoch+1}..{config.NUM_EPOCHS} ===\n")

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

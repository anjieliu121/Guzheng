#!/usr/bin/env python3
"""
Step 4: Train decoder-only transformer on guzheng MIDI.

Key features:
- Label smoothing (0.1) to prevent overconfidence
- Higher dropout (0.15) and weight decay (0.05)
- Early stopping with patience=30
- Logs train/val loss per epoch for plotting
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import Dataset, DataLoader

from config import TokenizerConfig, ModelConfig, TrainConfig, trial_root
from tokenizer import MidiTokenizer, scale_from_midi_filename


class GuzhengDataset(Dataset):
    """Chunks note streams, prefixing each chunk with BOS + KEY."""

    def __init__(self, sequences, context_length, stride, tok_cfg):
        self.seq_len = context_length + 1
        self.pad_token = tok_cfg.pad_token
        self.bos_token = tok_cfg.bos_token
        self.eos_token = tok_cfg.eos_token
        self.cfg = tok_cfg
        self.prefix_len = 2
        self.body_len = self.seq_len - self.prefix_len
        self.chunks = []

        for seq in sequences:
            if len(seq) < self.prefix_len + 1:
                continue
            if seq[0] != self.bos_token or not tok_cfg.is_key_token_id(seq[1]):
                continue

            key_tok = seq[1]
            core = seq[self.prefix_len:-1] if seq[-1] == self.eos_token else seq[self.prefix_len:]

            if len(core) <= self.body_len:
                body = core + [self.pad_token] * (self.body_len - len(core))
                self.chunks.append([self.bos_token, key_tok] + body[:self.body_len])
            else:
                for start in range(0, len(core) - self.body_len + 1, stride):
                    self.chunks.append(
                        [self.bos_token, key_tok] + core[start:start + self.body_len]
                    )
                tail_start = len(core) - self.body_len
                if tail_start % stride != 0:
                    self.chunks.append(
                        [self.bos_token, key_tok] + core[tail_start:]
                    )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        if len(chunk) < self.seq_len:
            chunk = chunk + [self.pad_token] * (self.seq_len - len(chunk))
        t = torch.tensor(chunk, dtype=torch.long)
        return t[:-1], t[1:]


def load_midi_dir(midi_dir, tokenizer):
    """Load and tokenize all MIDI files in a directory."""
    if not os.path.isdir(midi_dir):
        return []
    files = sorted(f for f in os.listdir(midi_dir) if f.endswith(".mid"))
    seqs = []
    for f in files:
        try:
            tokens = tokenizer.encode_midi(os.path.join(midi_dir, f))
            if len(tokens) > 4:
                seqs.append(tokens)
        except Exception as e:
            print(f"  Warning: skipping {f}: {e}")
    return seqs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, configs):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {k: asdict(v) for k, v in configs.items()},
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Guzheng Transformer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr

    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(train_cfg.seed)

    os.makedirs(os.path.join(train_cfg.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(train_cfg.output_dir, "logs"), exist_ok=True)

    # Load data
    tokenizer = MidiTokenizer(tok_cfg)
    print(f"\nLoading training data from: {train_cfg.midi_dir}")
    train_seqs = load_midi_dir(train_cfg.midi_dir, tokenizer)
    print(f"  {len(train_seqs)} sequences, {sum(len(s) for s in train_seqs)} total tokens")

    print(f"Loading validation data from: {train_cfg.val_dir}")
    val_seqs = load_midi_dir(train_cfg.val_dir, tokenizer)
    print(f"  {len(val_seqs)} sequences, {sum(len(s) for s in val_seqs)} total tokens")

    if not train_seqs:
        print("ERROR: No training data found. Run steps 01-03 first!")
        return

    train_ds = GuzhengDataset(train_seqs, train_cfg.context_length, train_cfg.stride, tok_cfg)
    val_ds = GuzhengDataset(val_seqs, train_cfg.context_length, train_cfg.stride, tok_cfg) if val_seqs else None

    print(f"Train chunks: {len(train_ds)}")
    if val_ds:
        print(f"Val chunks: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0,
    ) if val_ds else None

    # Model
    from model import GuzhengTransformer
    model = GuzhengTransformer(
        vocab_size=tok_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_token=tok_cfg.pad_token,
        label_smoothing=train_cfg.label_smoothing,
    ).to(device)
    print(f"\nModel parameters: {model.param_count():,}")
    print(f"Vocab size: {tok_cfg.vocab_size}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay, betas=(0.9, 0.98),
    )

    total_steps = train_cfg.num_epochs * max(1, len(train_loader))
    warmup = train_cfg.warmup_steps
    min_lr = 1e-6

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr / train_cfg.learning_rate,
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}")

    configs = {"tokenizer": tok_cfg, "model": model_cfg, "train": train_cfg}

    # Training loop with early stopping
    best_val = float("inf")
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {train_cfg.num_epochs} epochs")
    print(f"Early stopping patience: {train_cfg.early_stopping_patience}")
    print(f"Label smoothing: {train_cfg.label_smoothing}")
    print(f"Weight decay: {train_cfg.weight_decay}")
    print(f"Dropout: {model_cfg.dropout}")
    print(f"Batches/epoch: {len(train_loader)}")
    print()

    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)
            _, loss = model(inp, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(1, n_batches)

        # Validation
        avg_val = float("inf")
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    _, loss = model(inp, tgt)
                    val_loss_sum += loss.item()
                    val_n += 1
            avg_val = val_loss_sum / max(1, val_n)

        ppl = math.exp(min(avg_val if val_loader else avg_train, 20))
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train, 4),
            "val_loss": round(avg_val, 4) if val_loader else None,
            "ppl": round(ppl, 1),
            "lr": lr,
        })

        if epoch % 5 == 0 or avg_val < best_val:
            print(
                f"Epoch {epoch:3d} | train={avg_train:.4f} | val={avg_val:.4f} | "
                f"ppl={ppl:.1f} | lr={lr:.2e} | {elapsed:.1f}s"
            )

        # Early stopping check
        if val_loader:
            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                save_checkpoint(
                    os.path.join(train_cfg.output_dir, "checkpoints", "best_model.pt"),
                    model, optimizer, scheduler, epoch, avg_val, configs,
                )
                if epoch % 5 == 0:
                    print(f"  -> best model saved (val_loss={avg_val:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg.early_stopping_patience})")
                    break
        else:
            # No val set: save best by train loss
            if avg_train < best_val:
                best_val = avg_train
                save_checkpoint(
                    os.path.join(train_cfg.output_dir, "checkpoints", "best_model.pt"),
                    model, optimizer, scheduler, epoch, avg_train, configs,
                )

        # Periodic checkpoint
        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                os.path.join(train_cfg.output_dir, "checkpoints", f"epoch_{epoch:04d}.pt"),
                model, optimizer, scheduler, epoch,
                avg_val if val_loader else avg_train, configs,
            )

    # Save final checkpoint
    final_loss = avg_val if val_loader else avg_train
    save_checkpoint(
        os.path.join(train_cfg.output_dir, "checkpoints", "final_model.pt"),
        model, optimizer, scheduler, epoch, final_loss, configs,
    )

    # Save training history
    history_path = os.path.join(train_cfg.output_dir, "logs", "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete at epoch {epoch}.")
    print(f"Best val loss: {best_val:.4f}")
    print(f"History saved to: {history_path}")
    print(f"Best model: checkpoints/best_model.pt")


if __name__ == "__main__":
    main()

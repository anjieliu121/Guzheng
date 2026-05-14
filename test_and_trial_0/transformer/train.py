#!/usr/bin/env python3
"""Train a decoder-only transformer on tokenised guzheng MIDI."""

import argparse
import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from config import TokenizerConfig, ModelConfig, TrainConfig
from dataset import create_datasets
from model import GuzhengTransformer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, configs):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler._step_count if hasattr(scheduler, "_step_count") else 0,
            "val_loss": val_loss,
            "config": {k: asdict(v) for k, v in configs.items()},
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Guzheng Transformer")
    parser.add_argument("--midi_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--split_csv",
        default=None,
        help="CSV with columns file_base_name,split (overrides auto-detect)",
    )
    parser.add_argument(
        "--no_official_split",
        action="store_true",
        help="Random piece-held-out split instead of train_test_split.csv",
    )
    args = parser.parse_args()

    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.midi_dir:
        train_cfg.midi_dir = args.midi_dir
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.split_csv:
        train_cfg.split_csv = args.split_csv
    if args.no_official_split:
        train_cfg.use_official_split = False
    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.context_length:
        train_cfg.context_length = args.context_length

    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(train_cfg.seed)

    os.makedirs(os.path.join(train_cfg.output_dir, "checkpoints"), exist_ok=True)

    train_ds, val_ds, tokenizer = create_datasets(train_cfg, tok_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0,
    )

    model = GuzhengTransformer(
        vocab_size=tok_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_token=tok_cfg.pad_token,
    ).to(device)
    print(f"Model parameters: {model.param_count():,}")
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
    best_val = float("inf")
    print(f"\nTraining for {train_cfg.num_epochs} epochs, {len(train_loader)} batches/epoch\n")

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

            if batch_idx % train_cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch:3d} | batch {batch_idx:4d}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )

        avg_train = epoch_loss / max(1, n_batches)

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
        ppl = math.exp(min(avg_val, 20))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | train_loss {avg_train:.4f} | val_loss {avg_val:.4f} | "
            f"ppl {ppl:.1f} | {elapsed:.1f}s"
        )

        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(
                os.path.join(train_cfg.output_dir, "checkpoints", "best_model.pt"),
                model, optimizer, scheduler, epoch, avg_val, configs,
            )
            print(f"  -> saved best model (val_loss={avg_val:.4f})")

        if epoch % train_cfg.save_every == 0 or epoch == train_cfg.num_epochs - 1:
            save_checkpoint(
                os.path.join(train_cfg.output_dir, "checkpoints", f"epoch_{epoch:04d}.pt"),
                model, optimizer, scheduler, epoch, avg_val, configs,
            )

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

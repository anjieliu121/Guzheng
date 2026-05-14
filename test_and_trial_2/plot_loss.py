#!/usr/bin/env python3
"""Plot training loss curve from MIDI-RWKV LoRA training."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/anjie/Documents/MyGuzheng/Guzheng"
LOSS_FILE = f"{ROOT}/outputs/midi_rwkv_lora/loss_data.json"
OUT_PATH = f"{ROOT}/outputs/evaluation/loss_curve.png"

def main():
    losses = []
    with open(LOSS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                losses.append(d["loss"])

    steps = np.arange(1, len(losses) + 1)
    epochs = steps / 64  # 64 steps per epoch

    # Smooth with moving average
    window = min(10, len(losses) // 3)
    if window > 1:
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_x = epochs[window-1:]
    else:
        smooth = losses
        smooth_x = epochs

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, alpha=0.3, color='steelblue', label='Raw loss')
    ax.plot(smooth_x, smooth, color='steelblue', linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'MIDI-RWKV LoRA Training Loss ({len(losses)} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark epoch boundaries
    for e in range(1, int(epochs[-1]) + 1):
        ax.axvline(x=e, color='gray', linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Loss curve saved to {OUT_PATH}")
    print(f"Steps: {len(losses)}, Epochs: {epochs[-1]:.1f}")
    print(f"Initial loss: {losses[0]:.3f}, Latest loss: {losses[-1]:.3f}")
    print(f"Min loss: {min(losses):.3f} (step {np.argmin(losses)+1})")

if __name__ == "__main__":
    main()

"""Plot training and validation loss curves for test_and_trial_2."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(os.path.join(LOGS_DIR, "training_history.json")) as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
ppl = [h["ppl"] for h in history]

best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
best_val = min(history, key=lambda h: h["val_loss"])["val_loss"]

# ── Figure 1: Train vs Val Loss ──────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epochs, train_loss, "o-", color="#2196F3", linewidth=2, markersize=4,
         label="Train loss", alpha=0.9)
ax1.plot(epochs, val_loss, "s-", color="#F44336", linewidth=2, markersize=4,
         label="Val loss", alpha=0.9)

# Mark best epoch
ax1.axvline(best_epoch, color="#4CAF50", linestyle="--", alpha=0.7, linewidth=1.5)
ax1.plot(best_epoch, best_val, "*", color="#4CAF50", markersize=18, zorder=5,
         label=f"Best val = {best_val:.4f} (epoch {best_epoch})")

# Shade overfitting region
ax1.axvspan(best_epoch, max(epochs), color="#FFEB3B", alpha=0.1, label="Overfitting region")

ax1.set_xlabel("Epoch", fontsize=13)
ax1.set_ylabel("Loss", fontsize=13)
ax1.set_title("Training & Validation Loss — test_and_trial_2", fontsize=15, fontweight="bold")
ax1.legend(loc="upper right", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.set_xlim(-1, max(epochs) + 1)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "loss_curves.png"), dpi=150)
print(f"Saved: {os.path.join(PLOTS_DIR, 'loss_curves.png')}")

# ── Figure 2: Train-Val Gap ──────────────────────────────────────────
gap = [t - v for t, v in zip(train_loss, val_loss)]

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(epochs, gap, color=["#4CAF50" if g >= 0 else "#F44336" for g in gap], alpha=0.7)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axvline(best_epoch, color="#4CAF50", linestyle="--", alpha=0.7, linewidth=1.5,
            label=f"Best epoch ({best_epoch})")
ax2.set_xlabel("Epoch", fontsize=13)
ax2.set_ylabel("Train − Val Loss", fontsize=13)
ax2.set_title("Generalization Gap (Train − Val)", fontsize=15, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

fig2.tight_layout()
fig2.savefig(os.path.join(PLOTS_DIR, "generalization_gap.png"), dpi=150)
print(f"Saved: {os.path.join(PLOTS_DIR, 'generalization_gap.png')}")

# ── Figure 3: Perplexity ─────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(epochs, ppl, "D-", color="#9C27B0", linewidth=2, markersize=5, alpha=0.9)
ax3.axvline(best_epoch, color="#4CAF50", linestyle="--", alpha=0.7, linewidth=1.5,
            label=f"Best epoch ({best_epoch})")
best_ppl = min(history, key=lambda h: h["val_loss"])["ppl"]
ax3.plot(best_epoch, best_ppl, "*", color="#4CAF50", markersize=18, zorder=5,
         label=f"Best ppl = {best_ppl}")

ax3.set_xlabel("Epoch", fontsize=13)
ax3.set_ylabel("Perplexity", fontsize=13)
ax3.set_title("Validation Perplexity — test_and_trial_2", fontsize=15, fontweight="bold")
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))

fig3.tight_layout()
fig3.savefig(os.path.join(PLOTS_DIR, "perplexity.png"), dpi=150)
print(f"Saved: {os.path.join(PLOTS_DIR, 'perplexity.png')}")

plt.show()
print("Done.")

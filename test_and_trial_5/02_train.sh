#!/bin/bash
# MIDI-RWKV State-Tuning for Trial 5: Full MIDI_transposed Dataset
#
# State tuning on ~472 training files from MIDI_transposed/ with cosine LR decay.
#
# Prerequisites:
#   1. Run 01_prepare_data.py first to prepare data/train/
#   2. conda activate midi_rwkv
#
# Usage:
#   cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_5
#   bash 02_train.sh

set -e

TRIAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$TRIAL_ROOT")"
RWKV_PEFT="$REPO_ROOT/test_and_trial_0/midi-rwkv/RWKV-PEFT"
RWKV_DATA="$RWKV_PEFT/data"

export PROJECT_ROOT="$REPO_ROOT/test_and_trial_0/midi-rwkv"
export TOKENIZERS_PARALLELISM=false

# ── Verify prerequisites ────────────────────────────────────────────────────
if [ ! -d "$TRIAL_ROOT/data/train" ]; then
    echo "ERROR: data/train/ not found. Run 01_prepare_data.py first!"
    exit 1
fi

TRAIN_COUNT=$(ls "$TRIAL_ROOT/data/train/"*.mid 2>/dev/null | wc -l | tr -d ' ')
echo "Training files available: $TRAIN_COUNT"
if [ "$TRAIN_COUNT" -lt 10 ]; then
    echo "ERROR: Too few training files ($TRAIN_COUNT). Check data preparation."
    exit 1
fi

# ── Prepare RWKV-PEFT data directory ────────────────────────────────────────
if [ -d "$RWKV_DATA" ] && [ -L "$RWKV_DATA" ]; then
    echo "Removing existing symlink at $RWKV_DATA"
    rm "$RWKV_DATA"
elif [ -d "$RWKV_DATA" ]; then
    BACKUP="$RWKV_DATA.bak_trial5_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing data to $BACKUP"
    mv "$RWKV_DATA" "$BACKUP"
fi

echo "Symlinking $TRIAL_ROOT/data/train -> $RWKV_DATA"
ln -s "$TRIAL_ROOT/data/train" "$RWKV_DATA"

# Remove old preprocessed cache
PREPROC="$RWKV_DATA/preprocessed"
if [ -d "$PREPROC" ]; then
    echo "Removing old preprocessed cache: $PREPROC"
    rm -rf "$PREPROC"
fi

# ── Training configuration ──────────────────────────────────────────────────
load_model="$PROJECT_ROOT/midi_rwkv.pth"
proj_dir="$TRIAL_ROOT/checkpoints"

n_layer=12
n_embd=384

micro_bsz=1
epoch_save=2
ctx_len=2048
train_epochs=8
lr_init=2e-2
lr_final=1e-3

mkdir -p "$proj_dir"
mkdir -p "$TRIAL_ROOT/logs"

echo ""
echo "========================================"
echo "MIDI-RWKV State-Tuning - Trial 5"
echo "========================================"
echo "Base model: $load_model"
echo "Output dir: $proj_dir"
echo "Data dir:   $RWKV_DATA"
echo "Train files: $TRAIN_COUNT"
echo "LR: $lr_init -> $lr_final (cosine)"
echo "Epochs: $train_epochs"
echo "Context length: $ctx_len"
echo "Batch size: $micro_bsz"
echo "========================================"
echo ""

# ── Run training ────────────────────────────────────────────────────────────
cd "$RWKV_PEFT"
PYTHON="/opt/miniconda3/envs/midi_rwkv/bin/python3"

$PYTHON train.py --load_model "$load_model" \
  --proj_dir "$proj_dir" \
  --n_layer $n_layer \
  --n_embd $n_embd \
  --vocab_size 16000 \
  --ctx_len $ctx_len \
  --micro_bsz $micro_bsz \
  --epoch_count $train_epochs \
  --epoch_begin 0 \
  --epoch_save $epoch_save \
  --lr_init $lr_init \
  --lr_final $lr_final \
  --warmup_steps 20 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-8 \
  --accelerator mps \
  --devices 1 \
  --precision bf16 \
  --strategy auto \
  --grad_cp 0 \
  --my_testing "x070" \
  --train_type "state" \
  --op torch \
  2>&1 | tee "$TRIAL_ROOT/logs/training_output.log"

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $proj_dir"
echo "Training log: $TRIAL_ROOT/logs/training_output.log"
echo ""

# ── Restore original data directory ─────────────────────────────────────────
if [ -L "$RWKV_DATA" ]; then
    rm "$RWKV_DATA"
    echo "Removed symlink at $RWKV_DATA"
fi
LATEST_BACKUP=$(ls -td "$RWKV_DATA.bak_trial5_"* 2>/dev/null | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    mv "$LATEST_BACKUP" "$RWKV_DATA"
    echo "Restored original data from $LATEST_BACKUP"
fi

echo ""
echo "Next steps:"
echo "  bash run_post_training.sh"

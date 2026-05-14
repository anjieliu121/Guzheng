#!/bin/bash
# Run after training completes: generate, evaluate, and check overfitting.
#
# Usage:
#   conda activate midi_rwkv
#   cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_3
#   bash run_post_training.sh

set -e

TRIAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$TRIAL_ROOT")"
RWKV_PEFT="$REPO_ROOT/archive/midi-rwkv/RWKV-PEFT"
PYTHON="/opt/miniconda3/envs/midi_rwkv/bin/python3"

echo "========================================"
echo "POST-TRAINING PIPELINE"
echo "========================================"

# Check for checkpoints
CKPTS=$(ls "$TRIAL_ROOT/checkpoints/"rwkv-*.pth 2>/dev/null | wc -l | tr -d ' ')
echo "Found $CKPTS checkpoints"

if [ "$CKPTS" -eq 0 ]; then
    echo "ERROR: No checkpoints found in $TRIAL_ROOT/checkpoints/"
    exit 1
fi

ls -la "$TRIAL_ROOT/checkpoints/"rwkv-*.pth

# Step 1: Generate from all checkpoints
echo ""
echo "========================================"
echo "STEP 1: GENERATING FROM ALL CHECKPOINTS"
echo "========================================"
cd "$RWKV_PEFT"
$PYTHON "$TRIAL_ROOT/03_generate.py" \
    --all_checkpoints \
    --n_samples 10 \
    --unconstrained_too \
    2>&1 | tee "$TRIAL_ROOT/logs/generation_output.log"

# Step 2: Evaluate
echo ""
echo "========================================"
echo "STEP 2: EVALUATION"
echo "========================================"
cd "$TRIAL_ROOT"
$PYTHON "$TRIAL_ROOT/04_evaluate.py" \
    2>&1 | tee "$TRIAL_ROOT/logs/evaluation_output.log"

# Step 3: Overfitting check
echo ""
echo "========================================"
echo "STEP 3: OVERFITTING CHECK"
echo "========================================"
$PYTHON "$TRIAL_ROOT/05_overfitting_check.py" \
    2>&1 | tee "$TRIAL_ROOT/logs/overfitting_output.log"

echo ""
echo "========================================"
echo "ALL DONE"
echo "========================================"
echo "Results in: $TRIAL_ROOT/evaluation/"
echo "Generated MIDI in: $TRIAL_ROOT/generated/"
echo "Logs in: $TRIAL_ROOT/logs/"

#!/bin/bash
# Run after training completes: generate, post-process, evaluate, and check overfitting.
#
# Usage:
#   conda activate midi_rwkv
#   cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_5
#   bash run_post_training.sh

set -e

TRIAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$TRIAL_ROOT")"
RWKV_PEFT="$REPO_ROOT/test_and_trial_0/midi-rwkv/RWKV-PEFT"
PYTHON="/opt/miniconda3/envs/midi_rwkv/bin/python3"

echo "========================================"
echo "POST-TRAINING PIPELINE (Trial 5)"
echo "========================================"

# Check for checkpoints
CKPTS=$(ls "$TRIAL_ROOT/checkpoints/"rwkv-*.pth 2>/dev/null | wc -l | tr -d ' ')
echo "Found $CKPTS checkpoints"

if [ "$CKPTS" -eq 0 ]; then
    echo "ERROR: No checkpoints found in $TRIAL_ROOT/checkpoints/"
    exit 1
fi

ls -la "$TRIAL_ROOT/checkpoints/"rwkv-*.pth
mkdir -p "$TRIAL_ROOT/logs"

# Step 1: Generate from all checkpoints (val/test/synthetic prompts, EOS blocking)
echo ""
echo "========================================"
echo "STEP 1: GENERATING FROM ALL CHECKPOINTS"
echo "========================================"
cd "$RWKV_PEFT"
$PYTHON "$TRIAL_ROOT/03_generate.py" \
    --all_checkpoints \
    2>&1 | tee "$TRIAL_ROOT/logs/generation_output.log"

# Step 2: Post-process (single-track merge, pentatonic snap, range clamp, polyphony limit)
echo ""
echo "========================================"
echo "STEP 2: POST-PROCESSING"
echo "========================================"
cd "$TRIAL_ROOT"
$PYTHON "$TRIAL_ROOT/04_postprocess.py" \
    2>&1 | tee "$TRIAL_ROOT/logs/postprocess_output.log"

# Step 3: Evaluate
echo ""
echo "========================================"
echo "STEP 3: EVALUATION"
echo "========================================"
$PYTHON "$TRIAL_ROOT/05_evaluate.py" \
    2>&1 | tee "$TRIAL_ROOT/logs/evaluation_output.log"

# Step 4: Overfitting check
echo ""
echo "========================================"
echo "STEP 4: OVERFITTING CHECK"
echo "========================================"
$PYTHON "$TRIAL_ROOT/06_overfitting_check.py" \
    2>&1 | tee "$TRIAL_ROOT/logs/overfitting_output.log"

echo ""
echo "========================================"
echo "ALL DONE"
echo "========================================"
echo "Results in: $TRIAL_ROOT/evaluation/"
echo "Generated MIDI in: $TRIAL_ROOT/generated/"
echo "Logs in: $TRIAL_ROOT/logs/"

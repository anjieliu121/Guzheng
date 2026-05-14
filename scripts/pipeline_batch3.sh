#!/bin/bash
# Full post-processing pipeline for a new generation batch.
#
# Steps:
#   1. Transpose MIDI to D pentatonic (using transpose_to_D.py if key is known,
#      otherwise assume pitches already fall in the pentatonic set)
#   2. Shift scale-degree so D is the tonic (auto_d_center.py)
#      Also quantize to 16th-note grid.
#   3. Apply keyswitches (postprocess_guzheng_keyswitches.py)
#
# Usage: ./pipeline_batch3.sh <raw_midi_dir> <out_name>

set -e

RAW_DIR="$1"
OUT_NAME="$2"

if [ -z "$RAW_DIR" ] || [ -z "$OUT_NAME" ]; then
  echo "Usage: $0 <raw_midi_dir> <out_name>"
  echo "  e.g.  $0 test_and_trial_7/generated/20260416-153400  batch3"
  exit 1
fi

REPO=/Users/anjie/Documents/MyGuzheng/Guzheng
GEN=$REPO/test_and_trial_7/generated

D_DIR=$GEN/medium_${OUT_NAME}_D
KS_DIR=$GEN/medium_${OUT_NAME}_D_ks

echo "Step 1: D-center + quantize → $D_DIR"
python3 $REPO/scripts/auto_d_center.py \
  --input_dir "$RAW_DIR" \
  --output_dir "$D_DIR"

echo ""
echo "Step 2: Apply keyswitches → $KS_DIR"
python3 $REPO/scripts/postprocess_guzheng_keyswitches.py \
  --input_dir "$D_DIR" \
  --output_dir "$KS_DIR"

echo ""
echo "Done."
echo "  D-centered: $D_DIR"
echo "  With KS:    $KS_DIR"

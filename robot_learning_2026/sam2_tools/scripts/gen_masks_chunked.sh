#!/usr/bin/env bash
set -euo pipefail

CORRECTION=true

VIDEO_PATH="scripts/wrist_camera.mp4"
OUTPUT_MASK_DIR="masks/wrist_camera"

SCRIPT="scripts/generate_cloth_masks_sam2_chunked.py"
CHECKPOINT="checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"

CHUNK_SIZE=540

COMMON_ARGS=(
  --input-video "$VIDEO_PATH"
  --output-mask-dir "$OUTPUT_MASK_DIR"
  --checkpoint "$CHECKPOINT"
  --model-cfg "$MODEL_CFG"
  --chunk-size "$CHUNK_SIZE"
  --save-overlays
  --overlay-every 30
)

PLANNED_PROMPTS=(
  --correction 0:75,358,1
  --correction 0:240,208,1
  --correction 0:350,149,1
  --correction 0:423,44,0
  --correction 0:415,366,0
  --correction 0:81,559,0
  --correction 0:311,46,0

  --correction 540:38,509,1
  --correction 540:189,334,1
  --correction 540:344,181,1
  --correction 540:414,205,0
  --correction 540:423,379,0
  --correction 540:94,605,0

  --correction 1080:251,149,1
  --correction 1080:440,151,1
  --correction 1080:445,507,1
  --correction 1080:428,383,0
  --correction 1080:420,9,0
  --correction 1080:95,114,0
  --correction 1080:151,528,0

  --correction 1620:58,578,1
  --correction 1620:91,205,1
  --correction 1620:346,520,1
  --correction 1620:467,388,0
  --correction 1620:466,221,0
  --correction 1620:356,101,0
  --correction 1620:70,59,0
  --correction 1620:174,163,1

  --correction 2160:195,98,1
  --correction 2160:409,181,1
  --correction 2160:456,488,1
  --correction 2160:458,399,0
  --correction 2160:9,198,0
  --correction 2160:183,479,0

  --correction 2700:58,252,1
  --correction 2700:168,218,1
  --correction 2700:262,221,1
  --correction 2700:321,567,1
  --correction 2700:57,578,0
  --correction 2700:459,533,0
  --correction 2700:107,105,0
  --correction 2700:400,44,0
  --correction 2700:470,198,0
  --correction 2700:450,392,0

  --correction 3240:49,474,1 \
  --correction 3240:198,286,1 \
  --correction 3240:315,137,1 \
  --correction 3240:304,47,0 \
  --correction 3240:458,87,0 \
  --correction 3240:450,392,0 \
  --correction 3240:249,587,0 \

  --correction 3780:84,561,1 \
  --correction 3780:181,243,1 \
  --correction 3780:302,212,1 \
  --correction 3780:451,384,0 \
  --correction 3780:461,219,0 \
  --correction 3780:117,122,0 \
  --correction 3780:419,33,0 \
)

if [ "$CORRECTION" = false ]; then
  # Full initial run: use all planned frame-specific prompts.
  # Do NOT use --only-correction-chunks here.
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "${PLANNED_PROMPTS[@]}" \
    --overwrite
else
  # Correction pass: only rerun chunks containing these correction frames.
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --correction 1080:251,149,1 \
    --correction 1080:440,151,1 \
    --correction 1080:445,507,1 \
    --correction 1080:428,383,0 \
    --correction 1080:420,9,0 \
    --correction 1080:95,114,0 \
    --correction 1080:151,528,0 \
    --correction 1500:190,359,1 \
    --correction 1500:358,439,1 \
    --correction 1500:461,87,1 \
    --correction 1500:453,205,0 \
    --correction 1500:458,390,0 \
    --correction 1500:101,538,0 \
    --correction 1500:116,143,0 \
    --only-correction-chunks
fi
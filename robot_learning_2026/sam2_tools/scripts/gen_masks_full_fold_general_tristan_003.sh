#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_general_tristan/videos/observation.images.front/chunk-000/file-003.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-003"

SCRIPT="scripts/generate_cloth_masks_sam2_chunked.py"
CHECKPOINT="sam2/checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"

CHUNK_SIZE=540

# ---------------------------------------------------------

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
  --correction 20:22,302,1
  --correction 20:226,486,0
  --correction 20:428,251,0
  --correction 20:435,379,0
  --correction 20:263,72,0
  --correction 70:68,239,1
  --correction 70:116,371,1
  --correction 70:355,487,0
  --correction 70:428,382,0
  --correction 70:424,148,0
  --correction 70:295,86,0
  --correction 540:23,548,1
  --correction 540:111,340,0
  --correction 540:431,379,0
  --correction 540:435,253,0
  --correction 540:365,139,0
  --correction 560:12,387,1
  --correction 560:87,460,1
  --correction 560:274,519,0
  --correction 560:259,212,0
  --correction 560:447,242,0
  --correction 560:425,363,0
  --correction 560:355,107,0
  --correction 1081:10,226,1
  --correction 1081:147,419,0
  --correction 1081:453,379,0
  --correction 1081:416,239,0
  --correction 1081:314,70,0
  --correction 1120:74,374,1
  --correction 1120:172,470,1
  --correction 1120:267,234,0
  --correction 1120:449,266,0
  --correction 1120:442,371,0
  --correction 1120:355,516,0
  --correction 1120:402,90,0
  --correction 1620:56,273,1
  --correction 1620:144,575,1
  --correction 1620:236,147,0
  --correction 1620:386,546,0
  --correction 1620:418,387,0
  --correction 1620:436,261,0
  --correction 2160:139,141,1
  --correction 2160:243,325,1
  --correction 2160:73,485,1
  --correction 2160:353,515,0
  --correction 2160:448,393,0
  --correction 2160:417,171,0
  --correction 2160:294,73,0
  --correction 2700:189,246,1
  --correction 2700:96,561,1
  --correction 2700:422,583,0
  --correction 3240:370,290,1
  --correction 3240:211,248,1
  --correction 3240:75,140,1
  --correction 3240:199,578,1
  --correction 3240:409,522,1
  --correction 3240:410,596,0
  --correction 3240:462,402,0
  --correction 3240:454,234,0
  --correction 3240:400,86,0
  --correction 3780:112,342,1
  --correction 3780:270,467,1
  --correction 3780:333,373,1
  --correction 3780:370,90,1
  --correction 3780:254,103,1
  --correction 3780:447,206,0
  --correction 3780:462,382,0
  --correction 3780:61,112,0
  --correction 3780:72,549,0
  --correction 4320:269,227,1
  --correction 4320:449,270,1
  --correction 4320:438,376,0
  --correction 4320:306,545,0
  --correction 4320:310,83,0
  --correction 4860:66,492,1
  --correction 4860:237,561,1
  --correction 4860:417,452,1
  --correction 4860:350,308,1
  --correction 4860:407,262,0
  --correction 4860:435,383,0
  --correction 4860:143,275,0
  --correction 4860:141,98,0
  --correction 5400:196,403,1
  --correction 5400:252,536,1
  --correction 5400:293,292,1
  --correction 5400:17,338,1
  --correction 5400:79,214,0
  --correction 5400:247,54,0
  --correction 5400:441,251,0
  --correction 5400:445,385,0
  --correction 5400:414,555,0
  --correction 5940:34,441,1
  --correction 5940:196,588,1
  --correction 5940:153,464,1
  --correction 5940:56,572,1
  --correction 5940:34,559,1
  --correction 5940:188,564,1
  --correction 5940:173,439,1
  --correction 5940:253,308,0
  --correction 5940:323,141,0
  --correction 5940:441,264,0
  --correction 5940:443,380,0



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
    --correction 879:40,439,1 \
    --correction 879:96,265,0 \
    --correction 879:435,266,0 \
    --correction 879:425,377,0 \
    --correction 879:385,103,0 \
    --correction 879:325,545,0 \
    --correction 1170:312,353,1 \
    --correction 1170:338,249,1 \
    --correction 1170:318,300,1 \
    --correction 1170:425,260,0 \
    --correction 1170:419,361,0 \
    --correction 1170:205,554,0 \
    --correction 1170:193,128,0 \
    --correction 1170:447,72,0 \
    --correction 1520:145,295,1 \
    --correction 1520:364,289,1 \
    --correction 1520:97,208,1 \
    --correction 1520:365,119,1 \
    --correction 1520:64,80,0 \ 
    --correction 1520:450,59,0 \
    --correction 1520:73,543,0 \
    --correction 1520:455,394,0 \
    --correction 1520:466,228,0 \
    --only-correction-chunks
fi
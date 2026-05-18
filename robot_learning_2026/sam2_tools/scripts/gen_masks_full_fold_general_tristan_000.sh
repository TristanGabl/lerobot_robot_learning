#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="tmp_videos/file-000.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-000"

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
  --correction 10:53,523,1
  --correction 10:179,475,1
  --correction 10:209,292,0
  --correction 10:437,257,0
  --correction 10:442,372,0
  --correction 10:361,95,0

  --correction 540:149,596,1
  --correction 540:143,301,1
  --correction 540:329,297,1
  --correction 540:434,224,0
  --correction 540:450,390,0
  --correction 540:371,77,0
  --correction 540:411,513,0

  --correction 1080:180,396,1
  --correction 1080:334,258,1
  --correction 1080:304,89,1
  --correction 1080:419,97,0
  --correction 1080:436,368,0
  --correction 1080:153,574,0
  --correction 1080:75,143,0

  --correction 1620:85,293,1
  --correction 1620:350,573,1
  --correction 1620:299,242,1
  --correction 1620:311,62,1
  --correction 1620:415,93,0
  --correction 1620:430,368,0
  --correction 1620:73,552,0
  --correction 1620:93,93,0

  --correction 2160:129,106,1
  --correction 2160:151,309,1
  --correction 2160:328,257,1
  --correction 2160:429,99,1
  --correction 2160:468,214,0
  --correction 2160:465,391,0
  --correction 2160:133,548,0
  --correction 2160:39,108,0

  --correction 2700:141,216,1
  --correction 2700:186,313,1
  --correction 2700:334,285,1
  --correction 2700:411,118,1
  --correction 2700:424,539,1
  --correction 2700:451,405,0
  --correction 2700:457,232,0
  --correction 2700:110,539,0
  --correction 2700:63,147,0

  --correction 3240:150,184,1
  --correction 3240:192,331,1
  --correction 3240:328,249,1
  --correction 3240:392,119,1
  --correction 3240:457,215,0
  --correction 3240:470,389,0
  --correction 3240:183,552,0
  --correction 3240:78,104,0

  --correction 3780:164,311,1
  --correction 3780:156,138,1
  --correction 3780:293,243,1
  --correction 3780:386,126,1
  --correction 3780:451,212,0
  --correction 3780:440,375,0
  --correction 3780:438,495,1
  --correction 3780:109,543,0
  --correction 3780:61,188,0

  --correction 4320:66,137,1
  --correction 4320:325,306,1
  --correction 4320:331,589,1
  --correction 4320:415,514,0
  --correction 4320:455,405,0
  --correction 4320:466,229,0
  --correction 4320:410,58,0

  --correction 4860:95,129,1
  --correction 4860:326,503,1
  --correction 4860:232,377,1
  --correction 4860:469,217,0
  --correction 4860:442,384,0
  --correction 4860:407,520,0
  --correction 4860:359,72,0

  --correction 5400:107,239,1
  --correction 5400:321,274,1
  --correction 5400:406,520,1
  --correction 5400:451,232,0
  --correction 5400:441,379,0
  --correction 5400:451,471,0
  --correction 5400:417,95,0
  --correction 5400:55,86,1
  --correction 5400:31,31,0

  --correction 5940:100,395,1
  --correction 5940:165,104,1
  --correction 5940:413,52,0
  --correction 5940:426,378,0
  --correction 5940:279,485,0

  --correction 6480:24,474,1
  --correction 6480:5,453,1
  --correction 6480:429,265,0
  --correction 6480:424,373,0
  --correction 6480:231,471,0
  --correction 6480:320,135,0
  --correction 6480:171,54,0


  --correction 7020:39,491,1
  --correction 7020:75,596,1
  --correction 7020:441,248,0
  --correction 7020:418,377,0
  --correction 7020:80,277,0
  --correction 7020:306,548,0
  --correction 7020:419,80,0

  --correction 7565:15,478,1
  --correction 7565:115,299,0
  --correction 7565:283,584,0
  --correction 7565:449,252,0
  --correction 7565:443,382,0
  --correction 7565:339,89,0

  --correction 8100:50,454,1
  --correction 8100:102,577,1
  --correction 8100:56,378,1
  --correction 8100:214,246,0
  --correction 8100:366,540,0
  --correction 8100:425,387,0
  --correction 8100:443,263,0
  --correction 8100:383,85,0

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
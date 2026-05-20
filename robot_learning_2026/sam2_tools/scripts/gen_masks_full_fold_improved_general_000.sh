#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH=/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/videos/observation.images.front/chunk-000/file-000.mp4
OUTPUT_MASK_DIR="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/masks/observation.images.front/chunk-000/file-000/"

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
  --correction 50:59,892,1
  --correction 50:179,895,1
  --correction 50:140,988,1
  --correction 50:114,1115,1
  --correction 50:59,1222,1
  --correction 50:481,1146,0
  --correction 50:197,636,0
  --correction 50:643,545,0
  --correction 50:651,750,0
  --correction 50:628,319,0
  --correction 270:65,440,1
  --correction 270:459,279,1
  --correction 270:415,480,1
  --correction 270:365,820,1
  --correction 270:379,979,1
  --correction 270:401,1118,1
  --correction 270:140,682,1
  --correction 270:101,1112,0
  --correction 270:666,969,0
  --correction 270:654,728,0
  --correction 270:674,550,0
  --correction 270:272,145,0
  --correction 810:100,577,1
  --correction 810:284,369,1
  --correction 810:581,77,1
  --correction 810:593,468,1
  --correction 810:398,896,1
  --correction 810:649,909,1
  --correction 810:656,731,0
  --correction 810:668,205,0
  --correction 810:153,158,0
  --correction 810:158,1097,0
  --correction 1350:627,95,1
  --correction 1350:484,182,1
  --correction 1350:522,358,1
  --correction 1350:529,547,1
  --correction 1350:570,726,1
  --correction 1350:651,968,1
  --correction 1350:684,316,1
  --correction 1350:397,579,1
  --correction 1350:382,740,1
  --correction 1350:327,924,0
  --correction 1350:252,229,0
  --correction 1350:694,487,0
  --correction 1350:707,787,0
  --correction 1620:484,463,1
  --correction 1620:258,965,1
  --correction 1620:210,1238,1
  --correction 1620:610,570,1
  --correction 1620:578,1020,1
  --correction 1620:654,340,0
  --correction 1620:251,752,0
  --correction 1620:211,358,0
  --correction 1620:634,91,0
  --correction 1890:560,498,1
  --correction 1890:168,293,1
  --correction 1890:117,451,1
  --correction 1890:316,662,1
  --correction 1890:312,951,1
  --correction 1890:100,1146,1
  --correction 1890:100,799,1
  --correction 1890:442,1044,1
  --correction 1890:599,1085,0
  --correction 1890:681,752,0
  --correction 1890:681,557,0
  --correction 1890:324,186,0
  --correction 1890:564,71,0
  --correction 2160:287,342,1
  --correction 2160:432,504,1
  --correction 2160:447,551,1
  --correction 2160:481,630,1
  --correction 2160:674,626,1
  --correction 2160:512,688,1
  --correction 2160:601,720,0
  --correction 2160:681,500,0
  --correction 2160:561,118,0
  --correction 2160:267,83,0
  --correction 2160:106,750,0
  --correction 2160:403,1085,0
  --correction 2160:356,460,1
  --correction 2160:366,556,1
  --correction 2160:415,583,1
  --correction 2430:613,503,1
  --correction 2430:277,208,1
  --correction 2430:101,1026,1
  --correction 2430:684,761,0
  --correction 2430:692,314,0
  --correction 2430:640,975,0
  --correction 2430:636,115,0
  --correction 2700:258,606,1
  --correction 2700:360,668,1
  --correction 2700:622,574,1
  --correction 2700:541,290,1
  --correction 2700:604,294,1
  --correction 2700:672,378,0
  --correction 2700:686,743,0
  --correction 2700:494,904,0
  --correction 2700:135,164,0
  --correction 2700:648,72,0
  --correction 2700:531,121,1
  --correction 2970:391,533,1
  --correction 2970:216,542,1
  --correction 2970:65,734,1
  --correction 2970:60,916,1
  --correction 2970:95,259,1
  --correction 3240:60,428,1
  --correction 3240:340,241,1
  --correction 3240:595,355,1
  --correction 3240:298,424,1
  --correction 3240:657,725,0
  --correction 3240:438,810,0
  --correction 3240:101,161,0
  --correction 3240:449,370,1

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
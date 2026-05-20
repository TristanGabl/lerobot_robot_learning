#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH=/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/videos/observation.images.front/chunk-000/file-011.mp4
OUTPUT_MASK_DIR="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/masks/observation.images.front/chunk-000/file-011/"

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
  --correction 25:45,741,1
  --correction 25:74,387,0
  --correction 25:318,176,0
  --correction 25:340,842,0
  --correction 25:688,714,0
  --correction 25:683,535,0
  --correction 25:636,223,0
  --correction 50:117,418,1
  --correction 50:322,592,1
  --correction 50:120,679,1
  --correction 50:414,784,1
  --correction 50:130,899,1
  --correction 50:392,1102,0
  --correction 50:409,509,0
  --correction 50:456,238,0
  --correction 50:695,529,0
  --correction 50:662,761,0
  --correction 50:141,156,0
  --correction 270:130,598,1
  --correction 270:62,592,1
  --correction 270:561,232,1
  --correction 270:283,501,1
  --correction 270:572,597,1
  --correction 270:619,892,1
  --correction 270:662,732,0
  --correction 270:649,431,0
  --correction 270:185,209,0
  --correction 270:159,963,0
  --correction 270:680,162,0
  --correction 810:197,808,1
  --correction 810:467,942,1
  --correction 810:318,577,1
  --correction 810:549,662,1
  --correction 810:465,770,1
  --correction 810:561,221,1
  --correction 810:407,334,1
  --correction 810:691,486,0
  --correction 810:716,734,0
  --correction 810:273,1138,0
  --correction 810:129,392,0
  --correction 810:681,68,0
  --correction 1350:147,562,1
  --correction 1350:397,358,1
  --correction 1350:566,411,1
  --correction 1350:397,647,1
  --correction 1350:570,872,1
  --correction 1350:377,915,1
  --correction 1350:668,734,0
  --correction 1350:695,557,0
  --correction 1350:665,649,1
  --correction 1350:666,148,0
  --correction 1350:266,173,0
  --correction 1350:519,203,1
  --correction 1350:208,1058,0
  --correction 1350:208,734,1
  --correction 1350:98,630,1
  --correction 1620:70,437,1
  --correction 1620:136,583,1
  --correction 1620:410,773,1
  --correction 1620:114,805,1
  --correction 1620:232,758,1
  --correction 1620:135,1093,1
  --correction 1620:184,1194,1
  --correction 1620:520,1181,0
  --correction 1620:430,507,0
  --correction 1620:619,398,0
  --correction 1620:651,583,0
  --correction 1620:622,738,0
  --correction 1620:272,215,0
  --correction 1620:540,82,0
  --correction 1890:505,487,1
  --correction 1890:621,153,1
  --correction 1890:684,285,1
  --correction 1890:643,443,1
  --correction 1890:622,703,0
  --correction 1890:484,901,0
  --correction 1890:229,266,0
  --correction 1890:648,25,0
  --correction 1890:106,1023,0
  --correction 2160:155,811,1
  --correction 2160:401,550,1
  --correction 2160:200,544,1
  --correction 2160:48,643,1
  --correction 2160:48,135,1
  --correction 2160:135,305,1
  --correction 2160:639,278,0
  --correction 2160:649,80,0
  --correction 2160:686,709,0
  --correction 2160:380,1026,0
  --correction 2430:415,332,1
  --correction 2430:453,256,1
  --correction 2430:605,372,1
  --correction 2430:494,486,1
  --correction 2430:461,608,1
  --correction 2430:522,681,1
  --correction 2430:675,708,1
  --correction 2430:610,567,0
  --correction 2430:453,924,0
  --correction 2430:587,142,0
  --correction 2430:214,147,0
  --correction 2430:592,442,1
  --correction 2700:106,988,1
  --correction 2700:211,411,1
  --correction 2700:138,121,1
  --correction 2700:581,266,1
  --correction 2700:653,384,1
  --correction 2700:672,749,0
  --correction 2700:560,872,0
  --correction 2700:686,48,0
  --correction 2970:168,1214,1
  --correction 2970:470,1165,1
  --correction 2970:622,1064,1
  --correction 2970:537,840,1
  --correction 2970:523,687,1
  --correction 2970:648,621,1
  --correction 2970:713,504,0
  --correction 2970:716,741,0
  --correction 2970:182,792,0
  --correction 2970:130,264,0
  --correction 2970:595,141,0
  --correction 3240:257,287,1
  --correction 3240:569,644,1
  --correction 3240:616,902,1
  --correction 3240:179,1003,1
  --correction 3240:120,650,1
  --correction 3240:312,758,1
  --correction 3240:683,501,0
  --correction 3240:709,754,0
  --correction 3240:184,407,1
  --correction 3240:53,176,1
  --correction 3240:535,1059,1
  --correction 3510:66,1103,1
  --correction 3510:500,1134,1
  --correction 3510:468,801,1
  --correction 3510:417,681,1
  --correction 3510:627,636,1
  --correction 3510:683,509,0
  --correction 3510:124,638,0
  --correction 3510:334,124,0
  --correction 3510:622,50,0

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
#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_general_tristan/videos/observation.images.front/chunk-000/file-004.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-004"

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
--correction 10:27,513,1
--correction 10:15,520,1
--correction 10:113,347,0
--correction 10:461,244,0
--correction 10:425,375,0
--correction 10:287,65,0
--correction 75:65,397,1
--correction 75:145,494,1
--correction 75:204,351,1
--correction 75:252,122,0
--correction 75:349,551,0
--correction 75:434,254,0
--correction 75:430,371,0
--correction 546:9,625,1
--correction 546:161,409,0
--correction 546:437,253,0
--correction 546:455,378,0
--correction 546:277,110,0
--correction 579:81,259,1
--correction 579:70,414,1
--correction 579:150,585,1
--correction 579:369,550,0
--correction 579:413,383,0
--correction 579:435,273,0
--correction 579:279,163,0
--correction 579:453,60,0
--correction 1080:70,295,1
--correction 1080:81,462,1
--correction 1080:297,202,0
--correction 1080:432,249,0
--correction 1080:436,388,0
--correction 1620:448,301,1
--correction 1620:81,422,1
--correction 1620:412,529,0
--correction 1620:431,600,0
--correction 1620:476,423,0
--correction 1620:474,191,0
--correction 1620:437,68,0
--correction 2160:366,302,1
--correction 2160:319,560,1
--correction 2160:165,345,1
--correction 2160:71,491,1
--correction 2160:350,74,0
--correction 2160:475,223,0
--correction 2160:432,388,0
--correction 2160:450,561,0
--correction 2160:406,465,0
--correction 2700:180,354,1
--correction 2700:248,415,1
--correction 2700:302,322,1
--correction 2700:334,110,1
--correction 2700:421,312,1
--correction 2700:437,565,1
--correction 2700:439,112,1
--correction 2700:453,215,0
--correction 2700:459,384,0
--correction 2700:105,540,0
--correction 2700:146,112,0
--correction 3240:127,287,1
--correction 3240:267,269,1
--correction 3240:383,335,1
--correction 3240:445,503,1
--correction 3240:435,95,1
--correction 3240:460,207,0
--correction 3240:473,402,0
--correction 3240:63,114,0
--correction 3240:105,531,0
--correction 3780:72,446,1
--correction 3780:355,521,1
--correction 3780:310,213,1
--correction 3780:446,383,0
--correction 3780:463,239,0
--correction 3780:272,84,0
--correction 3780:30,554,0
--correction 4320:45,309,1
--correction 4320:261,97,1
--correction 4320:214,348,1
--correction 4320:336,238,1
--correction 4320:455,255,0
--correction 4320:437,378,0
--correction 4320:401,531,0
--correction 4320:125,63,0
--correction 4860:361,289,1
--correction 4860:422,224,1
--correction 4860:470,110,1
--correction 4860:433,16,0
--correction 4860:442,398,0
--correction 4860:236,516,0
--correction 4860:129,186,0
--correction 5400:393,305,1
--correction 5400:362,303,1
--correction 5400:352,408,1
--correction 5400:338,221,1
--correction 5400:273,295,1
--correction 5400:169,184,0
--correction 5400:437,53,0
--correction 5400:447,240,0
--correction 5400:449,396,0
--correction 5940:335,335,1
--correction 5940:163,521,1
--correction 5940:401,577,1
--correction 5940:450,396,0
--correction 5940:437,235,0
--correction 5940:199,137,0
--correction 5940:152,274,0
--correction 6480:82,465,1
--correction 6480:334,263,1
--correction 6480:331,566,1
--correction 6480:197,229,0
--correction 6480:394,83,0
--correction 6480:436,382,0
--correction 7057:11,525,1
--correction 7057:112,369,0
--correction 7057:465,384,0
--correction 7057:454,226,0
--correction 7057:273,75,0
--correction 7083:74,392,1
--correction 7083:104,278,1
--correction 7083:267,551,0
--correction 7083:219,157,0
--correction 7083:448,200,0
--correction 7083:411,371,0
--correction 7560:59,388,1
--correction 7560:31,488,1
--correction 7560:307,567,0
--correction 7560:454,400,0
--correction 7560:449,238,0
--correction 7560:227,156,0
--correction 8130:14,369,1
--correction 8130:162,468,0
--correction 8130:305,95,0
--correction 8130:438,274,0
--correction 8130:444,373,0
--correction 8155:49,373,1
--correction 8155:122,288,1
--correction 8155:278,210,0
--correction 8155:415,118,0
--correction 8155:454,240,0
--correction 8155:453,389,0
--correction 8155:349,527,0
--correction 8656:15,381,1
--correction 8656:208,280,0
--correction 8656:407,360,0
--correction 8656:425,264,0
--correction 8656:230,76,0
--correction 8674:86,304,1
--correction 8674:36,436,1
--correction 8674:304,509,0
--correction 8674:409,360,0
--correction 8674:448,255,0
--correction 8674:349,98,0
--correction 9180:96,486,1
--correction 9180:33,567,1
--correction 9180:299,576,0
--correction 9180:450,378,0
--correction 9180:458,259,0
--correction 9180:234,190,0
--correction 9720:29,530,1
--correction 9720:238,336,0
--correction 9720:452,367,0
--correction 9720:459,248,0
--correction 9720:298,133,0
--correction 10260:98,286,1
--correction 10260:302,139,1
--correction 10260:406,369,0
--correction 10260:292,473,0
--correction 10260:419,128,0

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
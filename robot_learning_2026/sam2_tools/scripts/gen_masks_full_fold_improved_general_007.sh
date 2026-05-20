#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH=/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/videos/observation.images.front/chunk-000/file-007.mp4
OUTPUT_MASK_DIR="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general/masks/observation.images.front/chunk-000/file-007/"

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
  --correction 50:57,869,1
  --correction 50:165,925,1
  --correction 50:47,1053,1
  --correction 50:89,1190,1
  --correction 50:375,1074,0
  --correction 50:283,529,0
  --correction 50:678,522,0
  --correction 50:688,755,0
  --correction 135:143,1015,1
  --correction 135:222,367,1
  --correction 135:162,174,1
  --correction 135:551,196,1
  --correction 135:502,410,1
  --correction 135:662,728,0
  --correction 135:551,901,0
  --correction 135:637,95,0
  --correction 270:176,795,1
  --correction 270:292,468,1
  --correction 270:398,828,1
  --correction 270:586,950,1
  --correction 270:488,603,1
  --correction 270:435,281,1
  --correction 270:511,153,1
  --correction 270:616,112,1
  --correction 270:692,475,0
  --correction 270:706,764,0
  --correction 270:645,646,1
  --correction 270:697,933,1
  --correction 270:348,1188,0
  --correction 270:175,466,0
  --correction 270:114,132,0
  --correction 405:388,395,1
  --correction 405:502,469,1
  --correction 405:334,548,1
  --correction 405:395,676,1
  --correction 405:649,641,1
  --correction 405:691,494,0
  --correction 405:628,466,1
  --correction 405:700,785,0
  --correction 405:500,977,0
  --correction 405:138,272,0
  --correction 675:456,82,1
  --correction 675:123,229,1
  --correction 675:544,424,1
  --correction 675:164,629,1
  --correction 675:126,1053,1
  --correction 675:538,1058,0
  --correction 675:675,743,0
  --correction 675:683,229,0
  --correction 675:684,132,0
  --correction 810:258,723,1
  --correction 810:388,449,1
  --correction 810:490,589,1
  --correction 810:436,711,1
  --correction 810:579,866,1
  --correction 810:569,972,1
  --correction 810:683,750,0
  --correction 810:691,620,1
  --correction 810:680,399,0
  --correction 810:491,202,1
  --correction 810:633,194,1
  --correction 810:112,252,0
  --correction 810:200,1150,0
  --correction 945:377,398,1
  --correction 945:388,522,1
  --correction 945:436,633,1
  --correction 945:316,764,1
  --correction 945:389,837,1
  --correction 945:607,671,1
  --correction 945:473,506,1
  --correction 945:552,480,1
  --correction 945:663,519,0
  --correction 945:716,807,0
  --correction 945:564,1062,0
  --correction 945:114,411,0
  --correction 1215:106,1035,1
  --correction 1215:140,565,1
  --correction 1215:153,123,1
  --correction 1215:497,185,1
  --correction 1215:348,582,1
  --correction 1215:657,360,1
  --correction 1215:665,734,0
  --correction 1215:578,896,0
  --correction 1215:672,88,0
  --correction 1350:188,626,1
  --correction 1350:586,167,1
  --correction 1350:369,714,1
  --correction 1350:671,737,0
  --correction 1350:649,410,0
  --correction 1350:686,567,1
  --correction 1350:575,1048,1
  --correction 1350:409,1121,0
  --correction 1350:223,316,0
  --correction 1485:362,329,1
  --correction 1485:512,408,1
  --correction 1485:447,489,1
  --correction 1485:586,527,1
  --correction 1485:485,627,1
  --correction 1485:573,740,1
  --correction 1485:707,795,0
  --correction 1485:707,451,0
  --correction 1485:596,191,0
  --correction 1485:77,348,0
  --correction 1485:280,1015,0

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
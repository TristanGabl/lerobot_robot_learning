#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_general_tristan/videos/observation.images.front/chunk-000/file-001.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-001"

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
--correction 0:47,383,1
--correction 0:26,561,1
--correction 0:191,541,0
--correction 0:433,371,0
--correction 0:450,259,0
--correction 0:246,185,0
--correction 0:285,100,0
 
--correction 540:248,279,1
--correction 540:82,531,1
--correction 540:33,383,1
--correction 540:106,226,0
--correction 540:360,481,0
--correction 540:420,380,0
--correction 540:430,265,0
--correction 540:417,134,0

--correction 1080:33,497,1
--correction 1080:121,554,1
--correction 1080:364,311,1
--correction 1080:430,250,0
--correction 1080:434,383,0
--correction 1080:145,303,0
--correction 1080:234,93,0

--correction 1620:318,198,1
--correction 1620:429,262,1
--correction 1620:448,126,1
--correction 1620:440,47,0
--correction 1620:437,373,0
--correction 1620:323,471,0
--correction 1620:99,113,0

--correction 2160:127,175,1
--correction 2160:88,417,1
--correction 2160:226,549,1
--correction 2160:442,248,0
--correction 2160:430,374,0
--correction 2160:419,491,0
--correction 2160:314,77,0

--correction 2700:192,233,1
--correction 2700:262,339,1
--correction 2700:361,316,1
--correction 2700:415,484,1
--correction 2700:433,92,1
--correction 2700:459,211,0
--correction 2700:464,387,0
--correction 2700:202,536,0
--correction 2700:78,171,0

--correction 3240:115,257,1
--correction 3240:208,332,1
--correction 3240:312,375,1
--correction 3240:428,96,1
--correction 3240:433,568,1
--correction 3240:457,397,0
--correction 3240:446,216,0
--correction 3240:65,81,0
--correction 3240:113,538,0

--correction 3780:87,117,1
--correction 3780:192,345,1
--correction 3780:347,318,1
--correction 3780:319,583,1
--correction 3780:403,474,0
--correction 3780:454,395,0
--correction 3780:459,235,0
--correction 3780:441,92,0

--correction 4320:94,82,1
--correction 4320:244,218,1
--correction 4320:223,443,1
--correction 4320:400,225,1
--correction 4320:423,371,0
--correction 4320:344,495,0
--correction 4320:430,105,0
--correction 4320:414,11,0

--correction 4860:43,292,1
--correction 4860:72,428,1
--correction 4860:51,230,1
--correction 4860:264,226,0
--correction 4860:375,96,0
--correction 4860:442,235,0
--correction 4860:433,374,0

--correction 5400:223,319,1
--correction 5400:262,455,1
--correction 5400:27,200,1
--correction 5400:334,201,0
--correction 5400:446,250,0
--correction 5400:435,378,0
--correction 5400:380,530,0
--correction 5400:290,35,0

--correction 5940:170,490,1
--correction 5940:421,523,1
--correction 5940:380,304,1
--correction 5940:432,249,0
--correction 5940:439,389,0
--correction 5940:33,123,0
--correction 5940:117,381,0
--correction 5940:222,134,0

--correction 6480:379,313,1
--correction 6480:432,234,0
--correction 6480:449,401,0
--correction 6480:251,470,0

--correction 7020:100,245,1
--correction 7020:354,109,1
--correction 7020:241,268,1
--correction 7020:425,254,0
--correction 7020:426,370,0
--correction 7020:317,484,0
--correction 7020:112,96,0
--correction 7020:452,62,0

--correction 7560:452,62,1
--correction 7560:180,297,1
--correction 7560:228,451,1
--correction 7560:395,485,1
--correction 7560:441,381,0
--correction 7560:465,224,0
--correction 7560:191,110,0
--correction 7560:73,523,0

--correction 8100:85,283,1
--correction 8100:208,434,1
--correction 8100:407,219,1
--correction 8100:257,79,1
--correction 8100:401,89,0
--correction 8100:441,380,0
--correction 8100:192,569,0
--correction 8100:54,76,0

--correction 8640:199,225,1
--correction 8640:229,300,1
--correction 8640:405,109,1
--correction 8640:295,251,1
--correction 8640:399,460,1
--correction 8640:441,384,0
--correction 8640:455,230,0
--correction 8640:114,132,0
--correction 8640:154,526,0

--correction 9180:232,604,1
--correction 9180:152,306,1
--correction 9180:349,258,1
--correction 9180:391,88,0
--correction 9180:466,203,0
--correction 9180:446,381,0
--correction 9180:383,510,0

--correction 9720:83,291,1
--correction 9720:144,515,1
--correction 9720:459,391,0
--correction 9720:456,238,0
--correction 9720:392,87,0
--correction 9720:358,532,0

--correction 10260:90,254,1
--correction 10260:103,607,1
--correction 10260:387,301,1
--correction 10260:414,93,0
--correction 10260:456,204,0
--correction 10260:458,403,0
--correction 10260:386,550,0

--correction 10800:182,73,1
--correction 10800:375,233,1
--correction 10800:164,554,1
--correction 10800:424,378,0
--correction 10800:333,514,0
--correction 10800:399,57,0

--correction 11340:49,199,1
--correction 11340:36,51,1
--correction 11340:104,397,1
--correction 11340:209,141,0
--correction 11340:280,482,0
--correction 11340:408,366,0
--correction 11340:431,249,0
--correction 11340:448,149,0

--correction 11902:6,515,1
--correction 11902:196,412,0
--correction 11902:442,250,0
--correction 11902:446,375,0
--correction 11902:231,154,0

--correction 11941:31,458,1
--correction 11941:51,506,1
--correction 11941:215,563,0
--correction 11941:180,328,0
--correction 11941:286,178,0
--correction 11941:435,261,0
--correction 11941:434,378,0

--correction 12420:253,330,1
--correction 12420:85,542,0
--correction 12420:363,610,0
--correction 12420:55,511,1
--correction 12420:336,580,1
--correction 12420:258,310,1
--correction 12420:134,384,1
--correction 12420:102,345,0
--correction 12420:271,211,0
--correction 12420:434,217,0
--correction 12420:424,376,0
--correction 12420:246,91,0

--correction 12960:101,545,1
--correction 12960:421,516,1
--correction 12960:380,313,1
--correction 12960:445,252,0
--correction 12960:424,373,0
--correction 12960:96,338,0
--correction 12960:233,97,0
--correction 12960:451,123,0

--correction 13500:429,288,1
--correction 13500:110,458,1
--correction 13500:408,497,1
--correction 13500:240,581,1
--correction 13500:443,380,0
--correction 13500:453,236,0
--correction 13500:402,157,0
--correction 13500:38,56,0
--correction 13500:91,314,0

--correction 14040:97,538,1
--correction 14040:295,571,1
--correction 14040:425,466,1
--correction 14040:390,314,1
--correction 14040:427,246,0
--correction 14040:444,392,0
--correction 14040:286,164,0
--correction 14040:164,312,0
--correction 14040:80,104,0

--correction 14580:396,322,1
--correction 14580:136,541,1
--correction 14580:365,592,1
--correction 14580:408,463,1
--correction 14580:447,397,0
--correction 14580:449,232,0
--correction 14580:132,320,0
--correction 14580:75,108,0
--correction 14580:300,115,0

--correction 15120:386,331,1
--correction 15120:226,460,1
--correction 15120:415,465,1
--correction 15120:237,609,1
--correction 15120:444,256,0
--correction 15120:446,397,0
--correction 15120:131,142,0
--correction 15120:147,337,0
--correction 15120:421,137,0

--correction 15660:427,296,1
--correction 15660:144,447,1
--correction 15660:362,529,1
--correction 15660:440,386,0
--correction 15660:456,231,0
--correction 15660:370,112,0
--correction 15660:73,118,0
--correction 15660:50,373,0

--correction 16200:210,296,1
--correction 16200:148,354,1
--correction 16200:221,341,1
--correction 16200:328,434,1
--correction 16200:352,311,1
--correction 16200:420,371,0
--correction 16200:443,243,0
--correction 16200:390,146,0
--correction 16200:123,172,0
--correction 16200:62,555,0

--correction 16740:112,336,1
--correction 16740:79,445,1
--correction 16740:336,434,1
--correction 16740:441,325,1
--correction 16740:453,401,0
--correction 16740:462,227,0
--correction 16740:317,225,0
--correction 16740:440,575,0
--correction 16740:48,596,0
--correction 16740:165,45,0
 
--correction 17280:417,324,1
--correction 17280:182,449,1
--correction 17280:416,479,1
--correction 17280:324,575,1
--correction 17280:454,391,0
--correction 17280:467,217,0
--correction 17280:141,325,0
--correction 17280:226,87,0
--correction 17280:433,96,0

--correction 17820:351,302,1
--correction 17820:370,434,1
--correction 17820:58,526,1
--correction 17820:62,389,0
--correction 17820:107,79,0
--correction 17820:403,114,0
--correction 17820:460,246,0
--correction 17820:445,387,0


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
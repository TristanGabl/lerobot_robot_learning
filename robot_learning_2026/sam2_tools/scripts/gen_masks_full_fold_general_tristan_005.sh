#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="tmp_videos/file-005.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-005"

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
--correction 7:22,325,1
--correction 7:250,529,0
--correction 7:315,111,0
--correction 7:456,232,0
--correction 7:434,378,0
--correction 126:77,395,1
--correction 126:196,285,1
--correction 126:97,567,1
--correction 126:350,555,0
--correction 126:246,119,0
--correction 126:390,64,0
--correction 126:449,237,0
--correction 126:439,377,0
--correction 540:21,204,1
--correction 540:78,287,1
--correction 540:246,265,1
--correction 540:323,498,0
--correction 540:412,374,0
--correction 540:452,255,0
--correction 540:279,53,0
--correction 1080:202,303,1
--correction 1080:30,391,1
--correction 1080:195,592,1
--correction 1080:406,544,0
--correction 1080:447,390,0
--correction 1080:444,235,0
--correction 1080:119,126,0
--correction 1080:341,74,0
--correction 1629:17,396,1
--correction 1629:257,513,0
--correction 1629:333,183,0
--correction 1629:448,243,0
--correction 1629:454,378,0
--correction 1669:67,366,1
--correction 1669:69,513,1
--correction 1669:322,549,0
--correction 1669:319,182,0
--correction 1669:450,253,0
--correction 1669:453,378,0
--correction 2178:9,446,1
--correction 2178:116,550,0
--correction 2178:251,150,0
--correction 2178:448,263,0
--correction 2178:440,364,0
--correction 2214:62,322,1
--correction 2214:123,237,1
--correction 2214:250,526,0
--correction 2214:274,99,0
--correction 2214:417,103,0
--correction 2214:467,237,0
--correction 2214:439,394,0
--correction 2720:20,409,1
--correction 2720:200,274,0
--correction 2720:290,519,0
--correction 2720:450,384,0
--correction 2720:456,253,0
--correction 2720:376,65,0
--correction 2750:56,466,1
--correction 2750:142,385,1
--correction 2750:135,233,0
--correction 2750:376,132,0
--correction 2750:468,246,0
--correction 2750:451,377,0
--correction 2750:340,591,0
--correction 3240:224,145,1
--correction 3240:82,316,1
--correction 3240:281,506,0
--correction 3240:460,385,0
--correction 3240:406,101,0
--correction 3780:434,259,1
--correction 3780:106,317,1
--correction 3780:170,544,1
--correction 3780:431,583,0
--correction 3780:457,50,0
--correction 3780:478,170,0
--correction 4320:233,53,1
--correction 4320:329,173,1
--correction 4320:319,321,1
--correction 4320:172,558,1
--correction 4320:126,249,1
--correction 4320:451,220,0
--correction 4320:435,382,0
--correction 4320:403,506,0
--correction 4320:430,99,0
--correction 4860:281,90,1
--correction 4860:114,294,1
--correction 4860:354,270,1
--correction 4860:372,478,0
--correction 4860:431,388,0
--correction 4860:444,110,0
--correction 4860:371,61,0
--correction 5400:298,177,1
--correction 5400:70,156,1
--correction 5400:236,373,1
--correction 5400:195,589,1
--correction 5400:452,394,0
--correction 5400:402,482,0
--correction 5400:461,234,0
--correction 5400:411,62,0
--correction 5940:119,409,1
--correction 5940:228,309,1
--correction 5940:318,314,1
--correction 5940:354,123,1
--correction 5940:442,82,1
--correction 5940:425,499,1
--correction 5940:451,398,0
--correction 5940:451,227,0
--correction 5940:144,123,0
--correction 5940:67,560,0
--correction 6480:142,234,1
--correction 6480:138,232,1
--correction 6480:379,224,1
--correction 6480:236,194,1
--correction 6480:416,372,0
--correction 6480:225,478,0
--correction 6480:186,84,0
--correction 7020:103,467,1
--correction 7020:371,493,1
--correction 7020:381,303,1
--correction 7020:444,373,0
--correction 7020:443,390,0
--correction 7020:446,233,0
--correction 7020:340,93,0
--correction 7020:106,361,0
--correction 7020:135,135,0
--correction 7560:12,351,1
--correction 7560:15,352,1
--correction 7560:163,527,1
--correction 7560:251,335,1
--correction 7560:130,432,1
--correction 7560:397,483,0
--correction 7560:438,383,0
--correction 7560:452,246,0
--correction 7560:175,198,0
--correction 7560:221,60,0
--correction 8100:85,373,1
--correction 8100:36,472,1
--correction 8100:241,567,0
--correction 8100:192,183,0
--correction 8100:273,87,0
--correction 8100:442,269,0
--correction 8100:453,387,0
--correction 8640:151,275,1
--correction 8640:374,239,1
--correction 8640:437,393,0
--correction 8640:444,125,0
--correction 8640:391,63,0
--correction 8640:379,596,0
--correction 9180:409,332,1
--correction 9180:217,78,1
--correction 9180:60,392,1
--correction 9180:307,561,1
--correction 9180:474,427,0
--correction 9180:468,222,0
--correction 9180:400,82,0
--correction 9720:366,298,1
--correction 9720:66,380,1
--correction 9720:228,62,1
--correction 9720:349,569,0
--correction 9720:453,392,0
--correction 9720:450,231,0
--correction 9720:376,124,0
--correction 10260:69,186,1
--correction 10260:422,156,1
--correction 10260:131,374,1
--correction 10260:312,531,0
--correction 10260:438,379,0
--correction 10260:456,261,0
--correction 10800:74,303,1
--correction 10800:348,303,1
--correction 10800:438,228,0
--correction 10800:445,395,0
--correction 10800:415,99,0
--correction 10800:378,514,0
--correction 11340:66,469,1
--correction 11340:219,341,1
--correction 11340:375,298,1
--correction 11340:353,74,1
--correction 11340:463,210,0
--correction 11340:459,393,0
--correction 11340:172,585,0
--correction 11340:92,145,0
--correction 11340:459,62,0
--correction 11880:168,344,1
--correction 11880:380,464,1
--correction 11880:448,184,1
--correction 11880:449,386,0
--correction 11880:420,27,0
--correction 11880:148,136,0
--correction 11880:131,560,0
--correction 12420:252,100,1
--correction 12420:69,515,1
--correction 12420:11,497,1
--correction 12420:299,310,1
--correction 12420:450,257,0
--correction 12420:433,394,0
--correction 12420:388,513,0
--correction 12420:419,145,0
--correction 12420:59,120,0
--correction 12960:432,472,1
--correction 12960:304,344,1
--correction 12960:361,291,1
--correction 12960:371,308,1
--correction 12960:363,252,1
--correction 12960:441,236,0
--correction 12960:457,393,0
--correction 12960:253,487,0
--correction 12960:406,74,0
--correction 12960:78,58,0
--correction 13500:33,390,1
--correction 13500:304,244,1
--correction 13500:297,254,1
--correction 13500:332,539,1
--correction 13500:421,376,0
--correction 13500:430,271,0
--correction 13500:428,98,0
--correction 13500:128,146,0
--correction 13500:138,50,0
--correction 14040:62,455,1
--correction 14040:53,535,1
--correction 14040:356,601,0
--correction 14040:177,292,0
--correction 14040:290,145,0
--correction 14040:454,264,0
--correction 14040:446,388,0
--correction 14580:67,264,1
--correction 14580:385,110,1
--correction 14580:372,139,1
--correction 14580:332,479,0
--correction 14580:433,395,0
--correction 14580:463,111,0
--correction 15120:152,85,1
--correction 15120:35,328,1
--correction 15120:224,616,1
--correction 15120:334,352,1
--correction 15120:442,242,0
--correction 15120:463,401,0
--correction 15120:420,115,0
--correction 15120:421,504,0
--correction 15660:74,495,1
--correction 15660:179,358,1
--correction 15660:371,326,1
--correction 15660:457,485,1
--correction 15660:315,89,1
--correction 15660:455,237,0
--correction 15660:397,187,1
--correction 15660:447,230,0
--correction 15660:465,416,0
--correction 15660:119,592,0
--correction 15660:37,170,0
--correction 15660:458,63,0

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
#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH="tmp_videos/file-002.mp4"
OUTPUT_MASK_DIR="masks/full_fold_general_tristan/chunk-000/file-002"

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
--correction 0:39,140,1
--correction 0:4,120,1
--correction 0:150,66,0
--correction 0:256,461,0
--correction 0:409,364,0
--correction 0:430,267,0
--correction 0:418,118,0

--correction 564:14,482,1
--correction 564:254,392,0
--correction 564:465,248,0
--correction 564:424,387,0
--correction 564:168,170,0

--correction 578:38,385,1
--correction 578:120,454,1
--correction 578:288,516,0
--correction 578:245,302,0
--correction 578:451,243,0
--correction 578:432,375,0
--correction 578:317,126,0

--correction 1080:41,454,1
--correction 1080:73,581,1
--correction 1080:12,546,1
--correction 1080:233,339,0
--correction 1080:440,258,0
--correction 1080:433,376,0
--correction 1080:374,191,0

--correction 1620:37,227,1
--correction 1620:120,436,1
--correction 1620:104,116,1
--correction 1620:216,110,0
--correction 1620:398,71,0
--correction 1620:428,371,0

--correction 2160:94,121,1
--correction 2160:142,404,1
--correction 2160:255,88,1
--correction 2160:386,155,0
--correction 2160:414,376,0
--correction 2160:454,586,0

--correction 2700:77,299,1
--correction 2700:301,478,1
--correction 2700:402,262,1
--correction 2700:451,160,0
--correction 2700:430,381,0
--correction 2700:391,600,0
--correction 2700:404,59,0

--correction 3240:138,130,1
--correction 3240:379,215,1
--correction 3240:376,519,0
--correction 3240:437,378,0
--correction 3240:443,60,0

--correction 3780:68,137,1
--correction 3780:115,400,1
--correction 3780:194,106,1
--correction 3780:395,140,0
--correction 3780:416,371,0
--correction 3780:436,554,0

--correction 4320:31,405,1
--correction 4320:94,501,1
--correction 4320:170,272,0
--correction 4320:376,177,0
--correction 4320:455,235,0
--correction 4320:429,368,0

--correction 4860:132,165,1
--correction 4860:113,435,0
--correction 4860:133,424,1
--correction 4860:405,135,0
--correction 4860:401,11,0
--correction 4860:436,385,0
--correction 4860:201,566,0

--correction 5400:112,312,1
--correction 5400:143,477,1
--correction 5400:241,198,0
--correction 5400:459,242,0
--correction 5400:431,388,0
--correction 5400:413,570,0

--correction 5940:78,351,1
--correction 5940:133,469,1
--correction 5940:160,184,1
--correction 5940:298,99,0
--correction 5940:298,572,0
--correction 5940:423,381,0
--correction 5940:444,229,0

--correction 6480:126,154,1
--correction 6480:219,442,1
--correction 6480:300,532,0
--correction 6480:403,363,0
--correction 6480:452,237,0
--correction 6480:352,117,0

--correction 7020:117,282,1
--correction 7020:213,519,1
--correction 7020:70,494,1
--correction 7020:369,528,0
--correction 7020:309,137,0
--correction 7020:438,254,0
--correction 7020:437,398,0

--correction 7560:228,425,1
--correction 7560:226,157,1
--correction 7560:452,93,0
--correction 7560:427,375,0
--correction 7560:322,559,0

--correction 8100:109,226,1
--correction 8100:321,253,1
--correction 8100:421,382,0
--correction 8100:376,500,0
--correction 8100:441,89,0

--correction 8640:260,191,1
--correction 8640:251,513,1
--correction 8640:376,516,0
--correction 8640:418,382,0
--correction 8640:447,72,0
--correction 8640:441,160,0

--correction 9180:137,164,1
--correction 9180:311,226,1
--correction 9180:451,407,0
--correction 9180:340,513,0
--correction 9180:450,55,0

--correction 9720:191,154,1
--correction 9720:301,220,1
--correction 9720:408,369,0
--correction 9720:302,505,0
--correction 9720:429,94,0

--correction 10260:233,195,1
--correction 10260:165,491,1
--correction 10260:387,538,0
--correction 10260:451,376,0
--correction 10260:456,137,0
--correction 10260:432,43,0

--correction 10800:412,97,1
--correction 10800:199,344,1
--correction 10800:387,321,1
--correction 10800:439,515,1
--correction 10800:460,410,0
--correction 10800:462,235,0
--correction 10800:80,137,0
--correction 10800:111,524,0

--correction 11340:165,308,1
--correction 11340:333,302,1
--correction 11340:414,120,1
--correction 11340:425,545,1
--correction 11340:455,314,1
--correction 11340:455,227,0
--correction 11340:456,396,0

--correction 11880:178,160,1
--correction 11880:208,97,1
--correction 11880:138,263,1
--correction 11880:221,312,1
--correction 11880:138,451,1
--correction 11880:310,546,0
--correction 11880:435,392,0
--correction 11880:450,251,0
--correction 11880:415,96,0

--correction 12420:329,309,1
--correction 12420:420,252,0
--correction 12420:432,376,0
--correction 12420:161,481,0
--correction 12420:128,171,0

--correction 12427:351,310,1
--correction 12427:371,422,1
--correction 12427:367,258,1
--correction 12427:424,250,0
--correction 12427:443,381,0
--correction 12427:138,187,0
--correction 12427:190,502,0
--correction 12427:423,67,0

--correction 12960:291,541,1
--correction 12960:405,322,1
--correction 12960:43,537,1
--correction 12960:443,377,0
--correction 12960:438,233,0
--correction 12960:128,336,0
--correction 12960:188,117,0

--correction 13507:8,461,1
--correction 13507:208,523,0
--correction 13507:408,370,0
--correction 13507:443,256,0
--correction 13507:236,192,0

--correction 13518:43,503,1
--correction 13518:117,469,1
--correction 13518:151,270,0
--correction 13518:352,529,0
--correction 13518:445,389,0
--correction 13518:454,256,0
--correction 13518:292,160,0

--correction 14040:8,243,1
--correction 14040:134,241,0
--correction 14040:436,260,0
--correction 14040:413,375,0
--correction 14040:340,111,0

--correction 14098:35,238,1
--correction 14098:131,277,1
--correction 14098:287,444,0
--correction 14098:276,121,0
--correction 14098:428,82,0
--correction 14098:450,247,0
--correction 14098:450,385,0

--correction 14580:8,327,1
--correction 14580:146,237,0
--correction 14580:429,262,0
--correction 14580:417,378,0
--correction 14580:271,514,0
--correction 14580:373,78,0

--correction 14615:26,383,1
--correction 14615:122,477,1
--correction 14615:102,285,1
--correction 14615:224,225,0
--correction 14615:349,507,0
--correction 14615:426,379,0
--correction 14615:453,262,0
--correction 14615:395,88,0

--correction 15120:18,153,1
--correction 15120:112,341,0
--correction 15120:443,185,0
--correction 15120:432,372,0
--correction 15120:305,59,0

--correction 15137:124,383,1
--correction 15137:16,475,1
--correction 15137:270,572,0
--correction 15137:224,225,0
--correction 15137:339,104,0
--correction 15137:392,362,0

--correction 15660:29,396,1
--correction 15660:142,517,1
--correction 15660:128,327,1
--correction 15660:222,245,0
--correction 15660:309,543,0
--correction 15660:402,361,0
--correction 15660:448,255,0
--correction 15660:414,148,0

--correction 16200:90,364,1
--correction 16200:205,603,1
--correction 16200:376,580,0
--correction 16200:418,363,0
--correction 16200:444,95,0
--correction 16200:391,29,0
--correction 16200:464,216,0

--correction 16740:69,244,1
--correction 16740:47,70,1
--correction 16740:236,277,1
--correction 16740:360,300,1
--correction 16740:369,547,1
--correction 16740:443,390,0
--correction 16740:452,229,0
--correction 16740:424,118,0
--correction 16740:433,516,0

--correction 17280:146,315,1
--correction 17280:257,378,1
--correction 17280:354,291,1
--correction 17280:412,97,1
--correction 17280:458,236,0
--correction 17280:451,397,0
--correction 17280:174,567,0
--correction 17280:98,135,0

--correction 17820:286,191,1
--correction 17820:452,126,1
--correction 17820:450,257,1
--correction 17820:425,368,0
--correction 17820:192,497,0

--correction 18360:327,312,1
--correction 18360:424,260,0
--correction 18360:434,377,0
--correction 18360:219,478,0
--correction 18360:118,161,0

--correction 18376:305,195,1
--correction 18376:330,313,1
--correction 18376:327,420,1
--correction 18376:233,306,1
--correction 18376:443,361,0
--correction 18376:450,226,0
--correction 18376:181,86,0
--correction 18376:228,562,0

--correction 18900:33,567,1
--correction 18900:61,603,1
--correction 18900:284,377,1
--correction 18900:318,528,1
--correction 18900:470,284,1
--correction 18900:448,390,0
--correction 18900:457,226,0
--correction 18900:428,105,0
--correction 18900:165,79,0
--correction 18900:61,280,0



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
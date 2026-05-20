#!/usr/bin/env bash
set -euo pipefail


# --------------------- CONFIGURATION ---------------------

CORRECTION=false

VIDEO_PATH=/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general_plus_addons/videos/observation.images.front/chunk-000/file-013.mp4
OUTPUT_MASK_DIR="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_improved_general_plus_addons/masks/observation.images.front/chunk-000/file-013/"

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
  --correction 50:166,886,1
  --correction 50:296,772,1
  --correction 50:276,556,1
  --correction 50:350,365,1
  --correction 50:366,639,1
  --correction 50:510,543,1
  --correction 50:670,523,0
  --correction 50:670,742,0
  --correction 50:563,943,0
  --correction 50:147,315,0
  --correction 270:63,803,1
  --correction 270:436,605,1
  --correction 270:535,802,1
  --correction 270:326,832,1
  --correction 270:593,1043,1
  --correction 270:152,1155,1
  --correction 270:664,739,0
  --correction 270:687,819,0
  --correction 270:110,548,0
  --correction 270:323,256,0
  --correction 540:390,560,1
  --correction 540:466,606,1
  --correction 540:336,651,1
  --correction 540:524,795,1
  --correction 540:653,842,1
  --correction 540:584,693,0
  --correction 540:696,417,0
  --correction 540:546,164,0
  --correction 540:180,189,0
  --correction 810:363,408,1
  --correction 810:576,247,1
  --correction 810:676,153,1
  --correction 810:489,359,1
  --correction 810:643,394,1
  --correction 810:660,722,0
  --correction 810:292,917,0
  --correction 810:210,212,0
  --correction 1080:266,634,1
  --correction 1080:426,356,1
  --correction 1080:393,474,1
  --correction 1080:533,436,1
  --correction 1080:410,559,1
  --correction 1080:583,467,1
  --correction 1080:678,530,0
  --correction 1080:635,703,0
  --correction 1080:370,868,0
  --correction 1080:196,296,0
  --correction 1080:558,227,0
  --correction 1350:223,914,1
  --correction 1350:256,494,1
  --correction 1350:253,279,1
  --correction 1350:355,295,1
  --correction 1350:581,425,1
  --correction 1350:336,560,1
  --correction 1350:429,954,1
  --correction 1350:569,688,0
  --correction 1350:653,533,0
  --correction 1350:526,172,0
  --correction 1350:87,416,0
  --correction 1350:238,221,1
  --correction 1620:32,582,1
  --correction 1620:87,731,1
  --correction 1620:273,938,1
  --correction 1620:204,756,1
  --correction 1620:192,680,1
  --correction 1620:435,645,1
  --correction 1620:360,737,1
  --correction 1620:530,882,0
  --correction 1620:293,477,0
  --correction 1620:615,537,0
  --correction 1620:630,683,0
  --correction 1620:609,275,0
  --correction 1890:129,1100,1
  --correction 1890:520,634,1
  --correction 1890:643,637,1
  --correction 1890:393,961,1
  --correction 1890:618,949,1
  --correction 1890:367,1173,1
  --correction 1890:658,504,0
  --correction 1890:200,565,0
  --correction 1890:449,130,0
  --correction 2160:286,865,1
  --correction 2160:449,851,1
  --correction 2160:321,943,1
  --correction 2160:606,1023,1
  --correction 2160:535,838,1
  --correction 2160:616,602,1
  --correction 2160:656,726,0
  --correction 2160:680,493,0
  --correction 2160:209,660,0
  --correction 2160:458,256,0
  --correction 2430:269,442,1
  --correction 2430:478,390,1
  --correction 2430:453,233,1
  --correction 2430:529,39,1
  --correction 2430:696,301,1
  --correction 2430:598,189,1
  --correction 2430:244,126,0
  --correction 2430:263,749,0
  --correction 2430:624,697,0
  --correction 2700:158,640,1
  --correction 2700:321,617,1
  --correction 2700:375,437,1
  --correction 2700:455,562,1
  --correction 2700:658,530,0
  --correction 2700:647,706,0
  --correction 2700:403,920,0
  --correction 2700:150,305,0
  --correction 2700:475,268,0
  --correction 2970:407,480,1
  --correction 2970:518,914,1
  --correction 2970:646,983,1
  --correction 2970:593,679,0
  --correction 2970:644,562,0
  --correction 2970:569,290,0
  --correction 2970:170,620,0
  --correction 3240:55,685,1
  --correction 3240:380,586,1
  --correction 3240:223,711,1
  --correction 3240:350,817,1
  --correction 3240:86,997,1
  --correction 3240:173,1138,1
  --correction 3240:510,997,0
  --correction 3240:227,436,0
  --correction 3240:615,291,0
  --correction 3240:630,725,0
  --correction 3240:403,145,0
  --correction 3510:112,931,1
  --correction 3510:589,1120,1
  --correction 3510:207,1097,1
  --correction 3510:464,854,1
  --correction 3510:655,574,1
  --correction 3510:596,688,0
  --correction 3510:676,473,0
  --correction 3510:267,660,0
  --correction 3510:373,182,0
  --correction 3780:632,619,1
  --correction 3780:675,393,0
  --correction 3780:635,302,1
  --correction 3780:633,113,1
  --correction 3780:687,178,1
  --correction 3780:352,261,0
  --correction 3780:258,966,0
  --correction 3780:678,766,0
  --correction 4050:273,554,1
  --correction 4050:427,459,1
  --correction 4050:384,259,1
  --correction 4050:433,344,1
  --correction 4050:656,374,1
  --correction 4050:512,388,1
  --correction 4050:655,548,0
  --correction 4050:624,691,0
  --correction 4050:373,842,0
  --correction 4050:144,288,0
  --correction 4050:470,145,0
  --correction 4320:230,425,1
  --correction 4320:407,530,1
  --correction 4320:264,665,1
  --correction 4320:475,740,1
  --correction 4320:93,1114,1
  --correction 4320:301,940,1
  --correction 4320:504,1031,0
  --correction 4320:689,731,0
  --correction 4320:675,537,0
  --correction 4320:393,284,0
  --correction 4320:83,301,0
  --correction 4590:243,354,1
  --correction 4590:287,347,1
  --correction 4590:330,485,1
  --correction 4590:420,511,1
  --correction 4590:413,608,1
  --correction 4590:523,568,1
  --correction 4590:450,660,1
  --correction 4590:609,671,0
  --correction 4590:672,534,0
  --correction 4590:450,228,0
  --correction 4590:384,1009,0
  --correction 4860:486,651,1
  --correction 4860:306,842,1
  --correction 4860:660,894,1
  --correction 4860:466,945,1
  --correction 4860:106,1114,1
  --correction 4860:569,1069,1
  --correction 4860:689,740,0
  --correction 4860:680,497,0
  --correction 4860:106,630,0
  --correction 4860:306,165,0
  --correction 5130:375,554,1
  --correction 5130:586,487,1
  --correction 5130:518,436,1
  --correction 5130:496,334,1
  --correction 5130:546,178,1
  --correction 5130:612,69,1
  --correction 5130:652,192,1
  --correction 5130:641,342,1
  --correction 5130:669,734,0
  --correction 5130:404,988,0
  --correction 5130:187,276,0
  --correction 5400:343,416,1
  --correction 5400:486,408,1
  --correction 5400:466,210,1
  --correction 5400:526,119,1
  --correction 5400:583,258,1
  --correction 5400:501,327,1
  --correction 5400:692,311,1
  --correction 5400:675,545,0
  --correction 5400:644,669,0
  --correction 5400:249,837,0
  --correction 5400:198,275,0
  --correction 5400:641,95,0
  --correction 5670:116,1098,1
  --correction 5670:343,845,1
  --correction 5670:186,626,1
  --correction 5670:356,516,1
  --correction 5670:224,396,1
  --correction 5670:469,528,1
  --correction 5670:433,671,1
  --correction 5670:633,689,0
  --correction 5670:612,875,0
  --correction 5670:701,504,0
  --correction 5670:123,98,0
  --correction 5940:590,1143,1
  --correction 5940:504,751,1
  --correction 5940:630,702,0
  --correction 5940:643,530,0
  --correction 5940:432,608,1
  --correction 5940:143,931,0
  --correction 5940:329,305,0
  --correction 5940:513,106,0
  --correction 6210:84,905,1
  --correction 6210:526,637,1
  --correction 6210:392,1067,1
  --correction 6210:703,439,0
  --correction 6210:187,444,0
  --correction 6210:409,136,0
  --correction 6480:363,682,1
  --correction 6480:380,782,1
  --correction 6480:600,858,1
  --correction 6480:543,653,1
  --correction 6480:669,579,1
  --correction 6480:690,467,0
  --correction 6480:233,539,0
  --correction 6480:436,176,0
  --correction 6750:213,551,1
  --correction 6750:404,559,1
  --correction 6750:553,454,1
  --correction 6750:490,387,1
  --correction 6750:590,268,1
  --correction 6750:700,491,0
  --correction 6750:643,708,0
  --correction 6750:452,977,0
  --correction 6750:132,282,0
  --correction 6750:683,95,0
  --correction 6750:581,227,1
  --correction 6750:149,597,1
  --correction 7020:64,1009,1
  --correction 7020:440,1004,1
  --correction 7020:280,579,1
  --correction 7020:136,279,1
  --correction 7020:380,413,1
  --correction 7020:552,467,1
  --correction 7020:680,531,0
  --correction 7020:675,746,0
  --correction 7020:700,1236,0
  --correction 7020:472,175,0
  --correction 7290:135,889,1
  --correction 7290:563,895,1
  --correction 7290:250,1123,1
  --correction 7290:624,1177,1
  --correction 7290:344,749,1
  --correction 7290:521,562,1
  --correction 7290:623,711,0
  --correction 7290:141,563,0
  --correction 7290:636,112,0
  --correction 7290:224,184,0
  --correction 7560:96,980,1
  --correction 7560:472,1100,1
  --correction 7560:507,688,1
  --correction 7560:650,653,1
  --correction 7560:706,588,1
  --correction 7560:656,522,0
  --correction 7560:712,765,0
  --correction 7560:190,694,0
  --correction 7560:240,221,0
  --correction 7560:378,99,0
  --correction 7560:387,855,1

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
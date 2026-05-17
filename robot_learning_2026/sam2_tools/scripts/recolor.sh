python scripts/recolor_video_from_masks.py \
  --input-video scripts/test_long.mp4 \
  --mask-dir masks/full_fold_general_tristan/chunk-000 \
  --output-video scripts/long_test_red.mp4 \
  --target-hue 0 \
  --target-sat 190 \
  --save-overlays \
  --overwrite

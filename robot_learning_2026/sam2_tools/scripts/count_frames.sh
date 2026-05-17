#!/usr/bin/env bash

ffprobe -v error \
  -select_streams v:0 \
  -count_frames \
  -show_entries stream=nb_read_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  "/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_general_tristan/videos/observation.images.front/chunk-000/file-001.mp4"

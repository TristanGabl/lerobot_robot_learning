#!/bin/bash

# This script runs a very short training session with grayscale enabled and sample saving turned on.
# It uses the dummy data available in the repo for testing.

lerobot-train \
  --dataset.repo_id="full_fold_tristan_300eps" \
  --dataset.root="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_tristan_300eps" \
  --policy.type="diffusion" \
  --policy.repo_id="local/testing_masks_diffusion" \
  --output_dir="outputs/train/grayscale_test" \
  --job_name=grayscale_test \
  --policy.push_to_hub=false \
  --policy.device=cpu \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=50 \
  --policy.use_grayscale=false \
  --policy.save_train_samples=true \
  --policy.save_train_samples_prob=0.5 \
  --policy.use_group_norm=false \
  --policy.use_separate_rgb_encoder_per_camera=false
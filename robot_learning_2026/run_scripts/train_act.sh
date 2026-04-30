# Train a diffusion policy on the collected dataset

param(
    [string]$Dataset = "grab-task",
    [string]$Root = "robot_learning_2026/50_corner_grab",
    [string]$OutputDir = "outputs/train/act_grab10k",
    [string]$JobName = "act",
    [int]$BatchSize = 4,
    [int]$Steps = 10000,
    [string]$Policy = "act"
)

lerobot-train \
  --dataset.repo_id="local/$Dataset" \
  --dataset.root=$Root \
  --policy.type=$Policy \
  --policy.repo_id="local/$JobName" \
  --output_dir=$OutputDir \
  --job_name=$JobName \
  --policy.device=cuda \
  --wandb.enable=true \
  --batch_size=$BatchSize \
  --steps=$Steps \
  --policy.push_to_hub=true

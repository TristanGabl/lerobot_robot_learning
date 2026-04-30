# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="local/grab-task" \
  --dataset.root="/home/ubuntu/lerobot_robot_learning/robot_learning_2026/50_corner_grab" \
  --policy.type="diffusion" \
  --policy.repo_id="diffusion_grab" \
  --output_dir=outputs/train/diffusion_grab_$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_grab \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=5000 \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=false

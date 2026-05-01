# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="slochmann/50_corner_grab" \
  --policy.type="multi_task_dit" \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.horizon=32 \
  --policy.objective=diffusion \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.use_amp=true \
  --batch_size=200 \
  --steps=30000 \
  --policy.repo_id="slochmann/DiT_grab" \
  --output_dir=outputs/train/DiT_grab_$(date +%Y%m%d_%H%M%S) \
  --job_name=DiT_grab \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --save_freq=10000 \
  --eval_freq=5000 \
  --policy.push_to_hub=true

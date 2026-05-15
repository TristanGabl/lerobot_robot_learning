# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/full_fold_tristan" \
  --policy.repo_id="DerBoroter/full_fold_tristan_dit" \
  --policy.type="multi_task_dit" \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.drop_n_last_frames=31 \
  --policy.objective=diffusion \
  --policy.image_resize_shape=[320,240] \
  --policy.image_crop_shape=null \
  --policy.use_amp=true \
  --batch_size=64 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=5000 \
  --output_dir=outputs/train/DiT_grab_$(date +%Y%m%d_%H%M%S) \
  --job_name=DiT_grab \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --num_workers=8

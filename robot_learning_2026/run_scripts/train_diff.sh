# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/50_fold" \
  --policy.type="diffusion" \
  --policy.repo_id="DerBoroter/diffusion_fold" \
  --output_dir=outputs/train/diffusion_fold_$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_fold \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=5000 \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=true

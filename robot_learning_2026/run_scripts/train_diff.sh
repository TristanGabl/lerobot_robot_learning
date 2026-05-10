# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/single_fold_2cm" \
  --policy.type="diffusion" \
  --policy.repo_id="DerBoroter/diffusion_fold_2cm" \
  --output_dir=outputs/train/diffusion_fold_2cm_$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_fold \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=25000 \
  --eval_freq=10000000 \
  --policy.resize_shape=[320,240] \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=true

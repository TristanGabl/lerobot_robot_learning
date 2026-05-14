# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/single_fold_2cm" \
  --policy.type="diffusion" \
  --policy.repo_id="DerBoroter/diffusion_fold_2cm_imgtransf" \
  --output_dir=outputs/train/diffusion_fold_2cm_imgtransf$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_fold_2cm_imgtransf \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=64 \
  --steps=50000 \
  --save_freq=25000 \
  --eval_freq=10000000 \
  --policy.resize_shape=[320,240] \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=true \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.drop_n_last_frames=31 \
  --policy.use_group_norm=false \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs='{"identity":{"type":"Identity","weight":0.8,"kwargs":{}},"color_jitter":{"type":"ColorJitter","weight":0.2,"kwargs":{"brightness":[0.7,1.3],"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":[-0.03,0.03]}}}'
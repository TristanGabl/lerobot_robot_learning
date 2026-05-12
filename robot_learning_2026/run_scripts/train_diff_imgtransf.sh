# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/single_fold_2cm" \
  --policy.type="diffusion" \
  --policy.repo_id="DerBoroter/diffusion_fold_2cm" \
  --output_dir=outputs/train/diffusion_fold_2cm_$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_fold_imgtransf \
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
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs='{"brightness":{"type":"ColorJitter","weight":1.0,"kwargs":{"brightness":[0.7,1.3]}},"contrast":{"type":"ColorJitter","weight":1.0,"kwargs":{"contrast":[0.8,1.2]}},"saturation":{"type":"ColorJitter","weight":1.0,"kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","weight":1.0,"kwargs":{"hue":[-0.03,0.03]}}}'

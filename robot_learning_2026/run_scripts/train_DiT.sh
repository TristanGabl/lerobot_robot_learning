# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/full_fold_tristan_300eps" \
  --policy.repo_id="DerBoroter/full_fold_tristan_300eps_dit" \
  --output_dir=outputs/train/full_fold_tristan_300eps_$(date +%Y%m%d_%H%M%S) \
  --policy.type="multi_task_dit" \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.horizon=48 \
  --policy.hidden_dim=768 \
  --policy.num_heads=12 \
  --policy.num_layers=7 \
  --policy.timestep_embed_dim=384 \
  --policy.drop_n_last_frames=31 \
  --policy.objective=diffusion \
  --policy.image_resize_shape=[341,256] \
  --policy.image_crop_shape=[256,256] \
  --policy.vision_num_keypoints=64 \
  --policy.image_crop_shape=null \
  --policy.use_amp=true \
  --batch_size=64 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=5000 \
  --job_name=DiT_grab \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.private=true \
  --num_workers=8 \
  '--dataset.image_transforms.tfs={"identity":{"type":"Identity","weight":0.8,"kwargs":{}},"color_jitter":{"type":"ColorJitter","weight":0.2,"kwargs":{"brightness":[0.5,1.5],"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":[-0.03,0.03]}}}' \
  --dataset.transforms_refresh=100000

# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/full_fold_improved" \
  --policy.type="diffusion" \
  --policy.repo_id="DerBoroter/full_fold_improved_dino" \
  --output_dir=outputs/train/full_fold_improved_$(date +%Y%m%d_%H%M%S) \
  --job_name=diffusion_fold \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=1 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_freq=10000000 \
  --policy.resize_shape=[320,192] \
  --policy.crop_shape=null \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=true \
  --policy.private=true \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.drop_n_last_frames=31 \
  --policy.pretrained_backbone_weights="dinov3_vits16" \
  --policy.backbone_lr_factor=0.1 \
  --policy.use_group_norm=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.random_order=false \
  '--dataset.image_transforms.tfs={"identity":{"type":"Identity","weight":0.8,"kwargs":{}},"color_jitter":{"type":"ColorJitter","weight":0.2,"kwargs":{"brightness":[0.5,1.5],"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":[-0.03,0.03]}}}'
  

# ResNet18_Weights.IMAGENET1K_V1 for resnet18
# dinov3_vits16 for dinov3 backbone 
# git submodule update --init --recursive

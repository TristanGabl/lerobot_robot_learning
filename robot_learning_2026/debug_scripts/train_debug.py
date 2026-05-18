import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parents[2])


sys.argv = [
    "debug_train.py",
    "--dataset.repo_id=DerBoroter/full_fold_improved",
    "--policy.type=diffusion",
    "--policy.repo_id=DerBoroter/debug",
    "--output_dir=outputs/train/debug",
    "--job_name=diffusion_fold",
    "--policy.device=cuda",
    "--wandb.enable=true",
    "--wandb.project=derRoboter",
    "--batch_size=1",
    "--steps=10",
    "--save_freq=10000",
    "--eval_freq=10000000",
    "--policy.resize_shape=[320,192]",
    "--policy.crop_shape=null",
    "--policy.use_amp=true",
    "--num_workers=0",
    "--log_freq=1",
    "--save_checkpoint=false",
    "--policy.push_to_hub=false",
    "--policy.horizon=64",
    "--policy.n_action_steps=32",
    "--policy.drop_n_last_frames=31",
    "--policy.pretrained_backbone_weights=dinov3_vits16",
    "--policy.backbone_lr_factor=0.1",
    "--policy.use_group_norm=false",
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.max_num_transforms=1",
    "--dataset.image_transforms.random_order=false",
    '--dataset.image_transforms.tfs={"identity":{"type":"Identity","weight":0.8,"kwargs":{}},"color_jitter":{"type":"ColorJitter","weight":0.2,"kwargs":{"brightness":[0.5,1.5],"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":[-0.03,0.03]}}}',
]

from lerobot.scripts.lerobot_train import main

main()

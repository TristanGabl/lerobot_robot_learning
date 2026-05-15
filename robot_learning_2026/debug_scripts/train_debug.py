import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parents[2])


sys.argv = [
    "debug_train.py",
    "--dataset.repo_id=DerBoroter/full_fold_tristan_300eps",
    "--policy.type=diffusion",
    "--policy.repo_id=DerBoroter/diffusion_fold_2cm",
    "--output_dir=outputs/train/diffusion_fold_2cm_debug",
    "--job_name=diffusion_fold",
    "--policy.device=cuda",
    "--batch_size=3",
    "--steps=10",
    "--save_freq=10",
    "--eval_freq=0",
    "--policy.resize_shape=[320,240]",
    "--policy.use_amp=true",
    "--num_workers=0",
    "--log_freq=1",
    "--save_checkpoint=false",
    "--policy.push_to_hub=false",
    "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
    "--policy.backbone_lr_factor=0.1",
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.max_num_transforms=1",
    "--dataset.image_transforms.random_order=false",
    '--dataset.image_transforms.tfs={"identity":{"type":"Identity","weight":0.5,"kwargs":{}},"color_jitter":{"type":"ColorJitter","weight":0.2,"kwargs":{"brightness":[0.7,1.3],"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":[-0.03,0.03]}}}',
    "--dataset.transforms_refresh=2"
]

from lerobot.scripts.lerobot_train import main

main()

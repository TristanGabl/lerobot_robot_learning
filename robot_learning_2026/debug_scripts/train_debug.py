import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parents[2])


sys.argv = [
    "debug_train.py",
    "--dataset.repo_id=DerBoroter/single_fold_2cm",
    "--policy.type=diffusion",
    "--policy.repo_id=DerBoroter/diffusion_fold_2cm",
    "--output_dir=outputs/train/diffusion_fold_2cm_debug",
    "--job_name=diffusion_fold",
    "--policy.device=cuda",
    "--batch_size=2",
    "--steps=3",
    "--save_freq=3",
    "--eval_freq=0",
    "--policy.resize_shape=[320,240]",
    "--policy.use_amp=true",
    "--num_workers=0",
    "--log_freq=1",
    "--save_checkpoint=false",
    "--policy.push_to_hub=false",
    "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
    "--policy.vision_backbone=dinov3_vits16"
]

from lerobot.scripts.lerobot_train import main

main()

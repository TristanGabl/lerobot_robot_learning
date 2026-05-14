import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parents[2])

    
sys.argv = [
    "debug_train.py",
    "--dataset.repo_id=DerBoroter/full_fold_tristan",
    "--policy.type=multi_task_dit",
    "--policy.repo_id=DerBoroter/agnostic_dit",
    "--output_dir=outputs/train/agnostic_dit",
    "--job_name=dit_fold",
    "--policy.device=cpu",
    "--batch_size=2",
    "--steps=3",
    "--save_freq=3",
    "--eval_freq=0",
    "--policy.use_amp=true",
    "--num_workers=0",
    "--log_freq=1",
    "--save_checkpoint=false",
    "--policy.push_to_hub=false",
    "--policy.image_resize_shape=[320,240]", 
    "--policy.image_crop_shape=null"
]

from lerobot.scripts.lerobot_train import main

main()

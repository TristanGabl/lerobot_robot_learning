import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parents[2])

sys.argv = [
    "debug_train.py",
    "--dataset.repo_id=DerBoroter/50_corner_grab",
    "--policy.type=multi_task_dit",
    "--policy.horizon=32",
    "--policy.n_action_steps=24",
    "--policy.objective=diffusion",
    "--policy.noise_scheduler_type=DDPM",
    "--policy.num_train_timesteps=100",
    "--policy.image_resize_shape=[320,240]",
    "--policy.image_crop_shape=null",
    "--policy.use_amp=true",
    "--policy.repo_id=DerBoroter/DiT_grab",
    "--output_dir=outputs/train/DiT_grab_debug",
    "--job_name=DiT_grab",
    "--policy.device=cpu",
    "--batch_size=2",
    "--steps=3",
    "--save_freq=3",
    "--eval_freq=0",
    "--num_workers=0",
    "--log_freq=1",
    "--save_checkpoint=false",
    "--policy.push_to_hub=false",
]

from lerobot.scripts.lerobot_train import main

main()

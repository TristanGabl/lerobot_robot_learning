# Record demonstrations from teleoperation
# Saves trajectory data to robot_learning_2026/dummy_data or specified location

# Set permissions for serial ports
sudo chmod 666 /dev/ttyACM1

lerobot-record \
  --robot.type=so101_follower --robot.port='/dev/ttyACM1' --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=true \
  --dataset.single_task="fold" \
  --dataset.streaming_encoding=false \
  --dataset.encoder_threads=6 \
  --dataset.repo_id=local/eval_fold_diff_$(date +%Y%m%d_%H%M%S) \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=90 \
  --policy.use_amp=true \
  --policy.path="DerBoroter/diffusion_fold_2cm_crop" \
  --policy.device=cuda \
  --policy.n_action_steps=15 \
  --interpolation_multiplier=1 \
  --dataset.fps=30 \
  --policy.noise_scheduler_type="DDIM" \
  --dataset.push_to_hub=false \
  --policy.num_inference_steps=50


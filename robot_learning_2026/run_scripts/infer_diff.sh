# Record demonstrations from teleoperation
# Saves trajectory data to robot_learning_2026/dummy_data or specified location

# Set permissions for serial ports
sudo chmod 666 /dev/ttyACM1

lerobot-record \
  --robot.type=so101_follower --robot.port='/dev/ttyACM1' --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.single_task="grab" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.repo_id=local/eval_grab_diff_$(date +%Y%m%d_%H%M%S) \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=90 \
  --policy.use_amp=true \
  --policy.path="slochmann/diffusion-grab" \
  --policy.device=cuda \
  --policy.n_action_steps=100 \
  --dataset.fps=15 \
  --policy.num_inference_steps=25


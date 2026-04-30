# Record demonstrations from teleoperation
# Saves trajectory data to robot_learning_2026/dummy_data or specified location

# Set permissions for serial ports
sudo chmod 666 /dev/ttyACM0

lerobot-record \
  --robot.type=so101_follower --robot.port='/dev/ttyACM0' --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.single_task="grab" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.repo_id=local/eval_grab_act10k_$(date +%Y%m%d_%H%M%S) \
  --dataset.num_episodes=1 \
  --policy.use_amp=true \
  --policy.path="/home/seb/MSc/lerobot_robot_learning/outputs/train/act_grab10k/checkpoints/010000/pretrained_model" \
  --policy.device=cuda \
  --policy.n_action_steps=100
  
  #--policy.num_inference_steps=10

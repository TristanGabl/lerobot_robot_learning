# DAgger: roll out a trained policy with human-in-the-loop corrections via teleop.

if [[ "$(uname)" == "Darwin" ]]; then
  echo "mac version"
  PORT='/dev/tty.usbmodem5B141126191'
  TELEOP_PORT='/dev/tty.usbmodem5B140317801'
  DEVICE=mps
  CAM=0
else
  sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
  PORT='/dev/ttyACM0'
  TELEOP_PORT='/dev/ttyACM1'
  DEVICE=cuda
  CAM=2
fi

lerobot-rollout \
  --strategy.type=dagger \
  --robot.type=so101_follower --robot.port="$PORT" --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAM, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --task="fold" \
  --policy.path="DerBoroter/diffusion_fold_2cm_150ep" \
  --policy.noise_scheduler_type="DDIM" \
  --policy.resize_shape=[320,240] \
  --device="$DEVICE" \
  --return_to_initial_position=true \
  --strategy.num_episodes=50 \
  --dataset.repo_id="DerBoroter/rollout_fold_2cm_dagger" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.push_to_hub=true \
  --dataset.single_task="single_fold"

# dataset path must start with DerBoroter/rollout_...
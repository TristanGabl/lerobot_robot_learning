# Deploy a trained policy via lerobot-rollout (base strategy = inference, no recording)

if [[ "$(uname)" == "Darwin" ]]; then
  echo "mac version"
  PORT='/dev/tty.usbmodem5B141126191'
  DEVICE=mps
  CAM=0
else
  sudo chmod 666 /dev/ttyACM0
  PORT='/dev/ttyACM0'
  DEVICE=cuda
  CAM=0
fi

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$PORT" --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: "$CAM", width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --task="fold" \
  --policy.path="DerBoroter/full_fold_improved_dit_diffusion" \
  --policy.noise_scheduler_type="DDIM" \
  --device="$DEVICE" \
  --return_to_initial_position=true

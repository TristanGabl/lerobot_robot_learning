if [[ "$(uname)" == "Darwin" ]]; then
  ROBOT_PORT='/dev/tty.usbmodem5B141126191'
  TELEOP_PORT='/dev/tty.usbmodem5B140317801'
  CAM=0
else
  export WINIT_UNIX_BACKEND=x11
  ROBOT_PORT='/dev/ttyACM1'
  TELEOP_PORT='/dev/ttyACM0'
  CAM=2
fi

lerobot-record \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAM, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.repo_id="DerBoroter/full_fold_tristan" \
  --dataset.num_episodes=50 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.push_to_hub=true \
  --dataset.private=true \
  --resume=true \
  --dataset.root=outputs/full_fold_tristan \
  --dataset.single_task="single_fold"
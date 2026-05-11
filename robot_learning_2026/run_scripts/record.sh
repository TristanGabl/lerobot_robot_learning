if [[ "$(uname)" == "Darwin" ]]; then
  echo "mac version"
  lerobot-record \
    --robot.type=so101_follower --robot.port='/dev/tty.usbmodem5B141126191' --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader --teleop.port='/dev/tty.usbmodem5B140317801' --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
    --display_data=false \
    --dataset.repo_id="DerBoroter/100_first_fold_tristan" \
    --dataset.num_episodes=18 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=6 \
    --dataset.push_to_hub=true \
    --resume=true \
    --dataset.root=outputs/100_first_fold_tristan \
    --dataset.single_task="single_fold"

else
WINIT_UNIX_BACKEND=x11 lerobot-record \
  --robot.type=so101_follower --robot.port='/dev/ttyACM1' --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port='/dev/ttyACM0' --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.repo_id="DerBoroter/single_fold_2cm" \
  --dataset.num_episodes=50 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.push_to_hub=true \
  --resume=true \
  --dataset.root=outputs/fold_2cm\
  --dataset.reset_time_s=10 \
  --dataset.single_task="single_fold"
fi
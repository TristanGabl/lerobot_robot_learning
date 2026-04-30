
DISPLAY=:0 lerobot-record \
  --robot.type=so101_follower --robot.port='/dev/ttyACM1' --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port='/dev/ttyACM0' --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.repo_id="slochmann/folding_task" \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.push_to_hub=true \
  --dataset.root=outputs/fold_$(date +%Y%m%d_%H%M%S) \
  --dataset.reset_time_s=0 \
  --dataset.single_task="1fold"

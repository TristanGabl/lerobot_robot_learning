# Record demonstrations from teleoperation
# Saves trajectory data to robot_learning_2026/dummy_data or specified location

param(
    [string]$Dataset = "folding-task",
    [int]$Episodes = 10,
    [string]$Root = "robot_learning_2026/fold3"
)

lerobot-record \
  --robot.type=so101_follower --robot.port='COM9' --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port='COM10' --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=true \
  --dataset.repo_id="slochmann/$Dataset" \
  --dataset.num_episodes=$Episodes \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=6 \
  --dataset.push_to_hub=true \
  --dataset.root=$Root \
  --dataset.reset_time_s=0 \
  --dataset.single_task="$Dataset"

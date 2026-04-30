# Replay recorded episodes from the dataset

param(
    [string]$Dataset = "folding-task",
    [int]$Episode = 9,
    [string]$Root = "robot_learning_2026/fold"
)

lerobot-replay \
  --robot.type=so101_follower --robot.port='COM9' --robot.id=my_awesome_follower_arm \
  --dataset.repo_id="local/$Dataset" \
  --dataset.root=$Root \
  --dataset.episode=$Episode

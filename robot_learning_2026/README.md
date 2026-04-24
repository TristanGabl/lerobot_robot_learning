
controllers:
```
Leader Arm:     /dev/tty.usbmodem5B140317801
Follower arm:   /dev/tty.usbmodem5B141126191
```

calibration:
```
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B141126191 --robot.id=my_awesome_follower_arm

lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B140317801 --teleop.id=my_awesome_leader_arm                           
```

teleoperation (with camera rotation corrected):
```
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B141126191 \
--robot.id=my_awesome_follower_arm \
--teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B140317801 \
--teleop.id=my_awesome_leader_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
--display_data=true
```

record teleoperations
```
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B141126191 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B140317801 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=local/record-test \
  --dataset.num_episodes=1 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --dataset.root=robot_learning_2026/dummy_data \
  --dataset.single_task="folding towel"
```

replay
```
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141126191 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=local/record-test \
    --dataset.root=robot_learning_2026/dummy_data \
    --dataset.episode=0 # choose the episode you want to replay
```



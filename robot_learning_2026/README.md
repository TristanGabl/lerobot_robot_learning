repo install
```
uv python install 3.12
uv venv --python 3.12

# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\activate

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Apple Silicon)
brew install ffmpeg

uv pip install -e .

pip install 'lerobot[feetech]'
```


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

can skip calibration, but have to copy calibration files
```
copy robot_learning_2026/my_awesome_follower_arm.json to /.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json

copy robot_learning_2026/my_awesome_leader_arm.json to /Users/trgabl/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_awesome_leader_arm.json
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



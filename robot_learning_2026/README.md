# Robot Learning - Folding


## 1. Installation

Create a virtual environment and install dependencies:

```bash
# all in one (inludeds dependencies for Asynchronous Inference)
brew install ffmpeg   # macOS (one-time system dep)
uv sync --locked --extra feetech --extra diffusion --extra training --extra async --extra test --extra dev
```

or split up
```bash
uv python install 3.12
uv venv --python 3.12

# Activate environment
source .venv/bin/activate  # Linux/macOS
# Windows: .venv\Scripts\activate

# System Dependencies
sudo apt install ffmpeg    # Ubuntu/Debian
# brew install ffmpeg      # macOS

# Install LeRobot and some additional features (you will probably have to install more)
uv pip install -e .
uv pip install 'lerobot[feetech,diffusion,dataset,training]'
```

## 2. Hardware Setup & Calibration

Identify your arm ports with the following command:
```
lerobot-find-port
```

Typical examples:
- **Linux:** `/dev/ttyACM0` (Leader), `/dev/ttyACM1` (Follower)
- **macOS:** `/dev/tty.usbmodem5B140317801` (Leader), `/dev/tty.usbmodem5B141126191` (Follower)

*(Linux only)* Grant USB port permissions everytime you connect the arms:
```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

### Calibration
If you already have calibration files, you can skip manual calibration:
```bash
# Copy pre-existing configurations
cp robot_learning_2026/my_awesome_follower_arm.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp robot_learning_2026/my_awesome_leader_arm.json ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```
Otherwise, calibrate manually:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=path_to/follower_port --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=path_to/leader_port --teleop.id=my_awesome_leader_arm                           
```

## 3. Data Collection

Before recording, find your camera's index via trial and error:
```bash
lerobot-find-cameras opencv
```

### Test Teleoperation
Ensure the arms and cameras sync correctly (press `Ctrl+C` to exit):
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=path_to/follower_port --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=path_to/leader_port --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=true
```

### Record Demonstrations
> **Note (Linux + Wayland):** The CLI for recording on Wayland is not well supported, you can find a script in `scripts/` that bypasses the CLI with a simple GUI. In the follwing commands simply substitute `lerobot-record` with `python robot_learning_2026/scripts/record_gui.py` to use it.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=path_to/follower_port --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=path_to/leader_port --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=true \
  --dataset.repo_id=local/record-test \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --dataset.root=robot_learning_2026/dummy_data \
  --dataset.single_task="folding towel"
```
This saves trajectories in `.parquet` format with a single `.mp4` video containing all the recorded examples.

### Replay
To verify the recorded behaviour, you can replay the trajectories:
```bash
lerobot-replay \
  --robot.type=so101_follower --robot.port=path_to/follower_port --robot.id=my_awesome_follower_arm \
  --dataset.repo_id=local/record-test \
  --dataset.root=robot_learning_2026/dummy_data \
  --dataset.episode=0
```

## 4. Training and Inference

Train a diffusion policy on the collected dataset:
```bash
lerobot-train \
  --dataset.repo_id=local/record-test \
  --dataset.root=robot_learning_2026/dummy_data \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_full \
  --job_name=diffusion_full \
  --policy.device=cuda \
  --wandb.enable=false \
  --batch_size=4 \
  --steps=2000 \
  --policy.push_to_hub=false
```
*Optional: Add `--dataset.episodes="[0,1,2]"` if you only want to train on a specific subset of recordings.*

### Inference
Once trained, run the model on the robot. (**Warning:** CPU inference is very unstable and slow, use GPU).
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=path_to/follower_port \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --dataset.single_task="folding towel" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.repo_id=local/eval_rl_folding \
  --dataset.num_episodes=1 \
  --policy.path=outputs/train/diffusion_full/checkpoints/last/pretrained_model \
  --policy.num_inference_steps=5
```

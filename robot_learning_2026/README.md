# Robot Learning - Folding


## 1. Installation

run all in one
```bash
sudo chmod -R +xwr  ./robot_learning_2026/run_scripts/
./robot_learning_2026/run_scripts/install.sh
```
Or

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
uv pip install 'lerobot[feetech,diffusion,dataset,training, viz, multi_task_dit]'
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

### Calibration (also in install.sh)
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

* the gripper must have *at least 6cm* distance from the towel when starting

**Folding:**
* the vertices have to match within *2cm* when folding 
* fold from *right bottom corner to left top*, else the camera doesnt see the edge of the towel to match the vertices
> read the email by simon sukup

Before recording, find your camera's index via trial and error:
```bash
lerobot-find-cameras opencv
```

### Test Teleoperation

Ensure the arms and cameras sync correctly (press `Ctrl+C` to exit):
```bash
teleoperate.sh
```

### Record Demonstrations

> **Note (Linux + Wayland):** The CLI for recording on Wayland is not well supported, you can find a script in `scripts/` that bypasses the CLI with a simple GUI. In the follwing commands simply substitute `lerobot-record` with `python robot_learning_2026/scripts/record_gui.py` to use it. If that doesnt work either, install X11 and use the original script!

```bash
record.sh
```

This saves trajectories in `.parquet` format with a single `.mp4` video containing all the recorded examples to https://huggingface.co/DerBoroter

### Replay
To verify the recorded behaviour, you can replay the trajectories:
```bash
replay.sh
```

## 4. Training and Inference
(Change the training parameters in the following commands)

compute instance **stop it again when done, limited credits**:
https://brev.nvidia.com/org/org-3Cfc5RioM40O3xB9vb9LYO3QYfd/billing?openAddCredits=true

*Compute is limited and charged by uptime, depending on your batch and model size you can train multiple policies in parallel. Smaller batchsize (e.g. 8) counterinutitively trains faster than larger here, at least for diffusion*

```bash
train_diff.sh #train diffusion policy, ~1h for 50k steps (20M params)
train_DiT.sh # train DiT policy, hyperparams not tuned
train_act.sh # train action policy (simple, small, fast, not allowed)
```
* checkpoints are pushed to:
https://huggingface.co/DerBoroter
* training metrics pushed to:
https://wandb.ai/derRoboter/derRoboter

*Optional: Add `--dataset.episodes="[0,1,2]"` if you only want to train on a specific subset of recordings.*

>*have not trained this:*
We can use directly DiT for training, which supports both diffusion and flow matching. For example we can train diffusion with:

NOTE: doublecheck + checkout documentation
https://huggingface.co/docs/lerobot/multi_task_dit
  

### Inference
Once trained, run the model on the robot. (**Warning:** CPU inference is very unstable and slow, use GPU).
> *automatically downloads selected policy from hf*
```bash
infer_act.sh #action policy inference
infer_diff.sh #diffusion policy inference
```

### Async Inference

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=100.80.255.111:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141126191 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
    --task="folding towel" \
    --policy_type=diffusion \
    --pretrained_name_or_path=outputs/train/diffusion_full/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

## 5. Develop custom policy
Develop custo policy from scratch (for future reference, now we can just modify the basic diffusion directly)

https://huggingface.co/docs/lerobot/en/bring_your_own_policies 
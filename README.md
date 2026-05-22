# Robot Learning - Folding Group 26

## 0. Installation
```bash
sudo chmod -R +xwr  ./robot_learning_2026/run_scripts/
./robot_learning_2026/run_scripts/install.sh
```

## 1. Rollout

The scripts used for the demo rollout were:
* for the first three tasks:

```
./robot_learning_2026/run_scripts/run_eval.sh
```
* for the bonus: 

```
./robot_learning_2026/run_scripts/run_eval_general.sh
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
./robot_learning_2026/run_scripts/teleoperate.sh
```

### Record Demonstrations

> **Note (Linux + Wayland):** The CLI for recording on Wayland is not well supported, you can find a script in `scripts/` that bypasses the CLI with a simple GUI. In the follwing commands simply substitute `lerobot-record` with `python robot_learning_2026/scripts/record_gui.py` to use it. If that doesnt work either, install X11 and use the original script!

```bash
./robot_learning_2026/run_scripts/record.sh
```

This saves trajectories in `.parquet` format with a single `.mp4` video containing all the recorded examples to https://huggingface.co/DerBoroter

### Replay
To verify the recorded behaviour, you can replay the trajectories:
```bash
./robot_learning_2026/run_scripts/replay.sh
```

## 4. Training
(Change the training parameters in the following commands)
```bash
./robot_learning_2026/run_scripts/train_diff_new.sh #train diffusion policy, tuned config
./robot_learning_2026/run_scripts/train_DiT.sh # train DiT policy, tuned config
./robot_learning_2026/run_scripts/train_act.sh # train action policy (simple, small, fast, not allowed)
```
* checkpoints are pushed to:
https://huggingface.co/DerBoroter
* training metrics pushed to:
https://wandb.ai/derRoboter/derRoboter

*Optional: Add `--dataset.episodes="[0,1,2]"` if you only want to train on a specific subset of recordings.*

NOTE: doublecheck + checkout documentation
https://huggingface.co/docs/lerobot/multi_task_dit
  

# Datasets

- **datasets/full_fold_improved**: 300 episodes of the full base task recorded over 3 locations
- **datasets/full_fold_improved_general_plus_addon**: 200 episodes of the full bonus task with various offsets and rotations + ~75 extra runs starting after the first fold to increase accurcy of second fold

# Models

- **checkpoints/full_fold_improved_dit_diffusion_v2**: model for full base task
- **checkpoints/full_fold_improved_general_dit_diffusion_color_aug_larger_with_addon**: model for full bonus task (naming got ridiculous)
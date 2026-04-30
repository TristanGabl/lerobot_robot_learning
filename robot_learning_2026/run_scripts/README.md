# Local Run Scripts

This folder contains PowerShell scripts for running common LeRobot operations locally. These scripts are **not pushed to git** and are meant for your local development and testing.

## Scripts

### `teleoperate.ps1`
Start teleoperation mode with the leader and follower arms.

```powershell
.\teleoperate.ps1
```

### `record.ps1`
Record demonstrations from teleoperation.

```powershell
# Record 10 episodes of "folding-task" (default)
.\record.ps1

# Record with custom parameters
.\record.ps1 -Dataset "my-task" -Episodes 20 -Root "outputs/my_dataset"
```

### `replay.ps1`
Replay recorded episodes to verify your data.

```powershell
# Replay episode 0 from the 20-corner-grab dataset (default)
.\replay.ps1

# Replay with custom parameters
.\replay.ps1 -Dataset "my-task" -Episode 5 -Root "robot_learning_2026/my_dataset"
```

### `train.ps1`
Train a diffusion policy on your dataset.

```powershell
# Train with default parameters
.\train.ps1

# Train with custom parameters
.\train.ps1 -Dataset "my-task" -Root "robot_learning_2026/my_dataset" `
  -OutputDir "outputs/train/my_policy" -JobName "my_policy" `
  -BatchSize 8 -Steps 5000 -Policy "diffusion"
```

## Requirements

- Python environment activated with LeRobot installed
- Feetech SDK installed: `uv pip install scservo-sdk` or `uv pip install -e ".[feetech]"`
- (for replay) Robot arms connected to COM9 (follower) and COM10 (leader)
- (for recording) Robot arms and camera connected

## Configuration

Edit the port numbers and camera settings directly in the scripts if needed.

- **Follower port:** `COM9`
- **Leader port:** `COM10`
- **Camera index:** `0`
- **Camera resolution:** `480x640`
- **Camera FPS:** `30`
- **Camera rotation:** `-90`

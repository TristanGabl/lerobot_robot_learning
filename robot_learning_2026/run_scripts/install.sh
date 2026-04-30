sudo apt update
sudo apt install ffmpeg   
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv

uv venv --python 3.12 #create virtual environment !!WITH PYTHON=3.12!!
source .venv/bin/activate #activate virtual environment

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install -e .
uv pip install 'lerobot[feetech,diffusion,dataset,training, viz, multi_task_dit, aloha, pusht]'
uv pip install pynput


# Copy pre-existing configurations
rm -r ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
rm -r ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
mkdir ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
mkdir ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
cp robot_learning_2026/my_awesome_follower_arm.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp robot_learning_2026/my_awesome_leader_arm.json ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/

# make scripts executable
sudo chmod -R +xwr  ./robot_learning_2026/run_scripts/
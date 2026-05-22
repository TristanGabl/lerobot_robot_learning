sudo apt update
sudo apt install ffmpeg   
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
bash

uv venv --python 3.12 #create virtual environment !!WITH PYTHON=3.12!!
source .venv/bin/activate #activate virtual environment

uv pip install -e .
uv pip install 'lerobot[feetech,diffusion,dataset,training, viz, multi_task_dit, aloha, pusht, multi_task_dit]'
uv pip install pynput
uv sync --locked --extra feetech --extra diffusion --extra training --extra async --extra test --extra dev
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install torchmetrics
uv pip install opencv-python
# git submodule update --init --recursive

# make scripts executable
sudo chmod -R +xwr  ./robot_learning_2026/run_scripts/
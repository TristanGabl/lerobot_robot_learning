from pathlib import Path
import torch.nn as nn

_REPO_ROOT = Path(__file__).parents[5]  # lerobot_robot_learning/src/lerobot/policies/diffusion/ -> lerobot_robot_learning/

DINOV3_REPO = str(_REPO_ROOT / "robot_learning_2026" / "dinov3")
DINOV3_WEIGHTS = str(_REPO_ROOT / "robot_learning_2026" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

class DINOv3SpatialBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
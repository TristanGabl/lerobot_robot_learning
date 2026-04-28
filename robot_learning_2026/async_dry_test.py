# dry_test.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks.mock_robot import MockRobotConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
import threading, time

client = RobotClient(RobotClientConfig(
    robot=MockRobotConfig(n_motors=6),
    server_address="100.80.255.111:8080",
    policy_type="diffusion",
    pretrained_name_or_path="outputs/train/diffusion_full/checkpoints/last/pretrained_model",
    policy_device="cuda",
    actions_per_chunk=10,
    task="folding towel",
))

client.start()
action_thread = threading.Thread(target=client.receive_actions, args=(True,), daemon=True)
action_thread.start()

time.sleep(5)  # let it run for 5 seconds
client.stop()

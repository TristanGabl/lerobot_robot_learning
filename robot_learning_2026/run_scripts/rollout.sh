# Deploy a trained policy via lerobot-rollout (base strategy = inference, no recording)

sudo chmod 666 /dev/ttyACM1

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port='/dev/ttyACM1' --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}}" \
  --display_data=false \
  --task="fold" \
  --duration=90 \
  --fps=30 \
  --interpolation_multiplier=1 \
  --policy.path="DerBoroter/diffusion_fold_2cm_ResNet18" \
  --policy.noise_scheduler_type="DDIM" \
  --return_to_initial_position=false \
  --policy.resize_shape=[320,240]

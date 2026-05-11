if [[ "$(uname)" == "Darwin" ]]; then
    python -m lerobot.async_inference.robot_client \
        --server_address=100.80.255.111:8080 \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5B141126191 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: -90}}" \
        --task="folding towel" \
        --policy_type=diffusion \
        --pretrained_name_or_path="DerBoroter/diffusion_fold_2cm_crop" \
        --policy_device=cuda \
        --actions_per_chunk=50 \
        --aggregate_fn_name=weighted_average \
        --debug_visualize_queue_size=True \
        --policy.n_action_steps=15 \
        --interpolation_multiplier=1 \
        --policy.num_inference_steps=50 \
        --policy.noise_scheduler_type="DDIM"
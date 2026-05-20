# Train a Multi-Task DiT (flow matching) policy with periodic validation
# on a separate held-out dataset.

uv run python robot_learning_2026/scripts/train_with_validation.py \
  --dataset.repo_id="DerBoroter/full_fold_tristan" \
  --val_dataset.repo_id="DerBoroter/full_fold_val" \
  --val_freq=1000 \
  --val_batches=20 \
  --policy.type="multi_task_dit" \
  --policy.repo_id="DerBoroter/full_fold_dit_multitask" \
  --output_dir=outputs/train/dit_fold_tristan_$(date +%Y%m%d_%H%M%S) \
  --job_name=dit_fold \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=derRoboter \
  --batch_size=64 \
  --steps=50000 \
  --save_freq=25000 \
  --eval_freq=10000000 \
  --policy.use_amp=true \
  --num_workers=8 \
  --policy.push_to_hub=true \
  --policy.objective="flow_matching" \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.image_crop_shape=[224,224] \
  --policy.num_layers=4 \
  --policy.hidden_dim=384 \
  --policy.private=true 

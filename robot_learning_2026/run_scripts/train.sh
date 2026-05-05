# Train a diffusion policy on the collected dataset

lerobot-train \
  --dataset.repo_id="DerBoroter/xyz" \
  --policy.type="diffusion" \
  --policy.repo_id="diffusion_grab" \
  --output_dir=DerBoroter/xyz \
  --job_name=diffusion_grab \
  --policy.device=cuda \
  --wandb.enable=true \
  --batch_size=4 \
  --steps=2000 \
  --policy.push_to_hub=true

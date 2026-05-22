#!/usr/bin/env python3

"""
Visualization script for visualising impact of dataset image augmentations from
LeRobot standard image transformations such as ColorJitter.
Used to tune boundaries for training.
"""

from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------- CONFIGURATION ----------------------------
# HuggingFace repo id
repo_id = "DerBoroter/full_fold_tristan"

# Output directory for visualizations
output_dir = Path(
    "robot_learning_2026/out_viz_transform"
)
#output_dir.mkdir(parents=True, exist_ok=True)

# Number of samples to visualize and number of random augmentations per sample
n_samples = 5
n_aug_per_sample = 6

# Boundaries for each property
boundaries_config = {
    "brightness": {"low": 0.5, "high": 1.5},
    "contrast": {"low": 0.8, "high": 1.2},
    "saturation": {"low": 0.8, "high": 1.2},
    "hue": {"low": -0.03, "high": 0.03},
}
# -----------------------------------------------------------------------



# Try to find a font, fallback to default if not found
try:
    font = ImageFont.truetype("/usr/share/fonts/fonts-go/Go-Mono.ttf", 20)
except Exception:
    font = ImageFont.load_default()

def add_label_to_img(tensor, label):
    # Convert tensor back to PIL
    img_pil = F.to_pil_image(tensor)
    
    # Create a new image with extra space at the bottom for text
    padding = 40
    new_img = Image.new("RGB", (img_pil.width, img_pil.height + padding), (0, 0, 0))
    new_img.paste(img_pil, (0, 0))
    
    # Draw text
    draw = ImageDraw.Draw(new_img)
    # Center text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text(((img_pil.width - text_width) // 2, img_pil.height + 5), label, font=font, fill=(255, 255, 255))
    
    # Convert back to tensor
    return F.to_tensor(new_img)



# Random RGB photometric jitter.
# These are intentionally moderate-to-strong so you can see the range.
random_jitter = v2.ColorJitter(
    brightness=(boundaries_config["brightness"]["low"], boundaries_config["brightness"]["high"]),
    contrast=(boundaries_config["contrast"]["low"], boundaries_config["contrast"]["high"]),
    saturation=(boundaries_config["saturation"]["low"], boundaries_config["saturation"]["high"]),
    hue=(boundaries_config["hue"]["low"], boundaries_config["hue"]["high"]),
)

# Individual "extreme" transforms so you can see the boundaries clearly.
extreme_transforms = {
    "brightness_low": v2.ColorJitter(brightness=(boundaries_config["brightness"]["low"], boundaries_config["brightness"]["low"])),
    "brightness_high": v2.ColorJitter(brightness=(boundaries_config["brightness"]["high"], boundaries_config["brightness"]["high"])),
    "contrast_low": v2.ColorJitter(contrast=(boundaries_config["contrast"]["low"], boundaries_config["contrast"]["low"])),
    "contrast_high": v2.ColorJitter(contrast=(boundaries_config["contrast"]["high"], boundaries_config["contrast"]["high"])),
    "saturation_low": v2.ColorJitter(saturation=(boundaries_config["saturation"]["low"], boundaries_config["saturation"]["low"])),
    "saturation_high": v2.ColorJitter(saturation=(boundaries_config["saturation"]["high"], boundaries_config["saturation"]["high"])),
    "hue_low": v2.ColorJitter(hue=(boundaries_config["hue"]["low"], boundaries_config["hue"]["low"])),
    "hue_high": v2.ColorJitter(hue=(boundaries_config["hue"]["high"], boundaries_config["hue"]["high"])),
}

dataset = LeRobotDataset(repo_id=repo_id)

indices = torch.linspace(0, len(dataset) - 1, steps=n_samples).long().tolist()

for sample_i, idx in enumerate(indices):
    sample = dataset[idx]
    img = sample["observation.images.front"]

    # NOTE: some LeRobot versions may return uint8 [0, 255] -> save_image prefers float [0, 1].
    if img.dtype == torch.uint8:
        img = img.float() / 255.0

    sample_dir = output_dir / f"sample_{sample_i:02d}_idx_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_image(img, sample_dir / "00_original.png")

    # Save random jitter examples
    labeled_original = add_label_to_img(img, "original")
    random_imgs = [labeled_original]
    for aug_i in range(n_aug_per_sample):
        aug = random_jitter(img)
        labeled_aug = add_label_to_img(aug, f"rand_{aug_i}")
        save_image(aug, sample_dir / f"random_aug_{aug_i:02d}.png")
        random_imgs.append(labeled_aug)

    grid = make_grid(random_imgs, nrow=len(random_imgs))
    save_image(grid, sample_dir / "random_aug_grid.png")

    # Save boundary examples
    boundary_imgs = [labeled_original]
    for name, tf in extreme_transforms.items():
        aug = tf(img)
        labeled_aug = add_label_to_img(aug, name)
        save_image(aug, sample_dir / f"{name}.png")
        boundary_imgs.append(labeled_aug)

    boundary_grid = make_grid(boundary_imgs, nrow=len(boundary_imgs))
    save_image(boundary_grid, sample_dir / "boundary_grid.png")

print(f"Saved transform examples to: {output_dir}")
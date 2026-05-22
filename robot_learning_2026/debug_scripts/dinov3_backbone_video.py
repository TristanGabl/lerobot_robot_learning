# this was claude generated

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

DINOV3_REPO = "/Users/trgabl/lerobot_robot_learning/robot_learning_2026/dinov3"
WEIGHTS = f"{DINOV3_REPO}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_PATH = "/Users/trgabl/lerobot_robot_learning/robot_learning_2026/50_corner_grab/videos/observation.images.front/chunk-000/file-001.mp4"
OUT_PATH = "/Users/trgabl/lerobot_robot_learning/robot_learning_2026/debug_scripts/dinov3_features.mp4"

model = torch.hub.load(repo_or_dir=DINOV3_REPO, model="dinov3_vits16", source="local", weights=WEIGHTS)
model.eval()

frames, _, meta = tvio.read_video(VIDEO_PATH, pts_unit="sec", output_format="TCHW")
fps = meta["video_fps"]
frames_f = frames.float() / 255.0  # (T, C, H, W)

_, _, H, W = frames_f.shape
h_p = (H // PATCH_SIZE) * PATCH_SIZE
w_p = (W // PATCH_SIZE) * PATCH_SIZE

# Run inference on all frames
all_tokens = []
with torch.inference_mode():
    for frame in tqdm(frames_f, desc="Extracting features"):
        img_t = TF.normalize(TF.resize(frame, (h_p, w_p)), IMAGENET_MEAN, IMAGENET_STD)
        feats = model.get_intermediate_layers(img_t.unsqueeze(0), n=1, reshape=True, norm=True)
        spatial = feats[0].squeeze(0)  # (C, fh, fw)
        all_tokens.append(spatial.permute(1, 2, 0).reshape(-1, spatial.shape[0]).numpy())

fh, fw = feats[0].shape[2], feats[0].shape[3]

# Fit PCA on first frame so colors are consistent across all frames
pca3 = PCA(n_components=3, whiten=True)
pca3.fit(all_tokens[0])

# Render each frame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

out_frames = []
for i, x in enumerate(tqdm(all_tokens, desc="Rendering")):
    rgb = torch.sigmoid(torch.from_numpy(pca3.transform(x)) * 2.0).numpy()
    rgb_map = rgb.reshape(fh, fw, 3)

    orig = TF.to_pil_image(frames[i])
    iw, ih = orig.size
    rgb_disp = np.array(Image.fromarray((rgb_map * 255).astype(np.uint8)).resize((iw, ih), Image.NEAREST))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(rgb_disp); axes[1].set_title(f"Patch features {iw} x {ih} (RGB)"); axes[1].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    out_frame = torch.from_numpy(np.array(Image.open(buf).convert("RGB")))
    out_frames.append(out_frame)

out_tensor = torch.stack(out_frames)  # (T, H, W, C)
tvio.write_video(OUT_PATH, out_tensor, fps=fps)
print(f"Saved to {OUT_PATH}")

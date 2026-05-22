# SAM2 Mask Generation Framework

This framework generates temporally consistent masks for the cloth during in the episodes datasets using the SAM2 model.

### Motivation
The goal is to achieve color invariance: this cannot be done with the standard ColorJitter, and grayscale early testing did not achieve good enough results. By masking the object, we can recolor only the target item during training.

### How it works
1. **Point Selection**: Pick keypoints on the object in a evenly spaced frames using `scripts/pick_points_batches.sh`.
2. **Mask Propagation**:  Propagate using SAM2 the picked points into masks across the entire video sequence using `scripts/gen_masks_path.sh`.
3. **Post-processing**: Fill gaps in sequences where the object is occluded, out of view or where SAM2 failed to track the cloth using `scripts/fill_missing_empty_masks.py`. This step is necessary to ensure frame-to-frame conversion with the episode videos (each episode frame needs a linked mask). 
4. **Integration**: Augment the LeRobot dataset with the masks by linking the masks to the corresponding video frames using `robot_learning_2026/scripts/build_mask_manifest.py`.
---

## Environment Setup

Create a new environment, download and install SAM2:

```
python3 -m venv .venv_sam2
source .venv_sam2/bin/activate
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Optionally install dependencies to use the notebooks

```
pip install -e ".[notebooks]"
```

Download the sam2 checkpoints (small model is enough for the cloth tracking)

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

Install extra dependencies
```
pip install opencv-python   
```


## Quick Start Workflow

1. **Pick Points**: Extracts frames (prefetching for efficincy) and opens a GUI to select object points.
   ```bash
   bash scripts/pick_points_batches.sh
   ```

2. **Generate Masks**: Run SAM2 propagation on the selected points.
   ```bash
   bash scripts/gen_masks_path.sh
   ```

3. **Fill Gaps**: Create empty masks for frames where the object isn't present or where SAM2 failed to track the cloth.
   ```bash
   python scripts/fill_missing_empty_masks.py --masks-path data/masks
   ```

4. **Verify**: (Optional) Generate a video to check mask quality and consitency.
   ```bash
   bash scripts/recolor.sh
   ```

5. **Build Manifest**: Link masks to the dataset for the training recoloring pipeline.
   ```bash
   python ../scripts/build_mask_manifest.py --repo-id your-dataset-id --masks-dir data/masks
   ``` 


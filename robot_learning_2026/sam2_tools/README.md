# Environment setup

Create a new environment, download and install sam2:

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

Download the sam2 checkpoints

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

Install extra dependencies
```
pip install opencv-python
```


## my_workflow

1) Prefetc all the frames to pick the points for sam2 and start pick points process

    `scripts/pick_points_batches.sh`

- Generate the masks, this will leave holes if a batch doesn't have start frame

    `scripts/gen_masks_path.sh`

- Test empty masks to fill holes left in gen_masks
    `scripts/fill_missing_empty_mask.py --masks-path=path/to/masks --dry-run`

- Fill in empty masks
    `scripts/fill_missing_empty_mask.py --masks-path=path/to/masks`

- Inspect result by generating a new video with recolored cloth based on masks
    `scripts/color.sh`


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
pick_points_batches.sh: prefetches all the frames to pick the points for sam2 and start pick points process
gen_masks_*: generate the masks, this will leave holes if a batch doesn't have start frame
fill_missing_empty_masks.py: fill with empty masks holes left in gen_masks
recolor.sh: test out the masks by generating a new video with recolored cloth based on masks


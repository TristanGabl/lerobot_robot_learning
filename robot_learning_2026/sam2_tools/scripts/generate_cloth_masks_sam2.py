#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_frame_count(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return int(data["streams"][0]["nb_read_frames"])


def extract_frames(video_path: Path, frames_dir: Path, overwrite: bool = True) -> None:
    if frames_dir.exists() and overwrite:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob("*.jpg"))
    if existing and not overwrite:
        print(f"Using existing frames in {frames_dir}")
        return

    run([
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        str(frames_dir / "%06d.jpg"),
    ])


def make_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame cloth masks for a video using SAM2."
    )

    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-mask-dir", required=True)
    parser.add_argument("--workdir", default="sam2_mask_work")

    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_small.pt",
    )

    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="0-based frame index where points/box are defined.",
    )

    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Cloth box on prompt frame, e.g. --box 120 180 520 430",
    )

    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Positive cloth click on prompt frame. Can repeat.",
    )

    parser.add_argument(
        "--neg-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Negative background/robot/table click on prompt frame. Can repeat.",
    )

    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save overlay debug images.",
    )
    parser.add_argument(
        "--overlay-every",
        type=int,
        default=10,
        help="Save one overlay every N frames.",
    )

    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted jpg frames in workdir.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output mask dir if it exists.",
    )

    args = parser.parse_args()

    input_video = Path(args.input_video).expanduser().resolve()
    output_mask_dir = Path(args.output_mask_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    frames_dir = workdir / "frames"
    overlay_dir = output_mask_dir.parent / f"{output_mask_dir.name}_overlays"

    if output_mask_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output mask dir already exists: {output_mask_dir}\n"
                f"Use --overwrite or choose a new path."
            )
        shutil.rmtree(output_mask_dir)

    output_mask_dir.mkdir(parents=True, exist_ok=False)

    if args.save_overlays:
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.box is None and not args.point:
        raise ValueError(
            "Provide either --box X1 Y1 X2 Y2 or at least one --point X Y on the cloth."
        )

    src_frame_count = ffprobe_frame_count(input_video)
    print(f"Input decoded frames: {src_frame_count}")

    extract_frames(input_video, frames_dir, overwrite=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) != src_frame_count:
        print(
            f"Warning: extracted {len(frame_paths)} frames, "
            f"but ffprobe counted {src_frame_count}."
        )

    if not (0 <= args.prompt_frame < len(frame_paths)):
        raise ValueError(
            f"--prompt-frame {args.prompt_frame} outside extracted frame range 0..{len(frame_paths)-1}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        args.checkpoint,
        device=device,
    )

    with torch.inference_mode():
        inference_state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        predictor.reset_state(inference_state)

        points = []
        labels = []

        if args.point:
            for p in args.point:
                points.append(p)
                labels.append(1)

        if args.neg_point:
            for p in args.neg_point:
                points.append(p)
                labels.append(0)

        points_np = np.array(points, dtype=np.float32) if points else None
        labels_np = np.array(labels, dtype=np.int32) if labels else None
        box_np = np.array(args.box, dtype=np.float32) if args.box is not None else None

        obj_id = 1

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=args.prompt_frame,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np,
            box=box_np,
        )

        masks_written = 0

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            # We track one object only.
            mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
            mask = np.squeeze(mask).astype(bool)

            frame_bgr = cv2.imread(str(frame_paths[out_frame_idx]))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_paths[out_frame_idx]}")

            if mask.shape[:2] != frame_bgr.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # 0-based naming is easiest for direct frame indexing later.
            mask_u8 = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(output_mask_dir / f"{out_frame_idx:06d}.png"), mask_u8)
            masks_written += 1

            if args.save_overlays and out_frame_idx % args.overlay_every == 0:
                overlay = make_overlay(frame_bgr, mask)
                cv2.imwrite(str(overlay_dir / f"{out_frame_idx:06d}.jpg"), overlay)

    mask_paths = sorted(output_mask_dir.glob("*.png"))
    print(f"Masks written: {len(mask_paths)}")

    if len(mask_paths) != len(frame_paths):
        missing = len(frame_paths) - len(mask_paths)
        raise RuntimeError(
            f"Mask count mismatch: frames={len(frame_paths)}, masks={len(mask_paths)}, missing={missing}"
        )

    metadata = {
        "input_video": str(input_video),
        "frame_count_ffprobe": src_frame_count,
        "frames_extracted": len(frame_paths),
        "masks_written": len(mask_paths),
        "prompt_frame": args.prompt_frame,
        "points": args.point,
        "neg_points": args.neg_point,
        "box": args.box,
        "model_cfg": args.model_cfg,
        "checkpoint": args.checkpoint,
        "mask_format": "uint8 PNG, 0 background, 255 cloth",
        "frame_indexing": "0-based mask filenames: 000000.png corresponds to first extracted frame",
    }
    (output_mask_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print("\nDone.")
    print(f"Mask dir: {output_mask_dir}")
    if args.save_overlays:
        print(f"Overlay dir: {overlay_dir}")
    print(f"Metadata: {output_mask_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
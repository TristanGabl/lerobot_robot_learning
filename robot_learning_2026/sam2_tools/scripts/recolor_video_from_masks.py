#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_fps(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    rate = data["streams"][0]["avg_frame_rate"]
    num, den = rate.split("/")
    return float(num) / float(den)


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


def save_video_from_frames(frames_out_dir: Path, output_video: Path, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_out_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ])


def hsv_recolor_preserve_shading(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    target_hue: int,
    target_sat: int,
    value_factor: float = 1.0,
    target_val: int | None = None,
    val_blend: float = 0.7,
    alpha_blur: int = 9,
    mask_threshold: int = 127,
) -> np.ndarray:
    """
    Recolor only masked pixels while preserving shading/folds.

    OpenCV HSV uint8 convention:
      H: 0..179
      S: 0..255
      V: 0..255

    For example:
      red    ~= H 0
      yellow ~= H 30
      green  ~= H 60
      blue   ~= H 110
      purple ~= H 145
    """
    mask = mask_u8 > mask_threshold

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    recolored_hsv = hsv.copy()

    recolored_hsv[..., 0][mask] = np.clip(target_hue, 0, 179)
    recolored_hsv[..., 1][mask] = np.clip(target_sat, 0, 255)

    v = recolored_hsv[..., 2].astype(np.float32)

    if target_val is not None:
        target_val = np.clip(target_val, 0, 255)
        v[mask] = (1.0 - val_blend) * v[mask] + val_blend * target_val

    if value_factor != 1.0:
        v[mask] *= value_factor

    recolored_hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)

    recolored_bgr = cv2.cvtColor(recolored_hsv, cv2.COLOR_HSV2BGR)
    alpha = mask.astype(np.float32) * 255.0
    if alpha_blur > 0:
        if alpha_blur % 2 == 0:
            alpha_blur += 1
        alpha = cv2.GaussianBlur(alpha, (alpha_blur, alpha_blur), 0)

    alpha = (alpha / 255.0)[..., None]

    out = recolored_bgr.astype(np.float32) * alpha + frame_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


    
def make_overlay(frame_bgr: np.ndarray, mask_u8: np.ndarray, mask_threshold: int) -> np.ndarray:
    mask = mask_u8 > mask_threshold
    overlay = frame_bgr.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recolor a video using precomputed per-frame cloth masks."
    )

    parser.add_argument("--input-video", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--workdir", default="mask_recolor_work")

    parser.add_argument("--target-hue", type=int, default=60)
    parser.add_argument("--target-sat", type=int, default=190)
    parser.add_argument("--value-factor", type=float, default=1.0)
    parser.add_argument(
        "--target-val",
        type=int,
        default=None,
        help="Optional target HSV value/brightness 0..255. If set, moves masked pixels toward this V.",
)

    parser.add_argument("--alpha-blur", type=int, default=9)
    parser.add_argument("--mask-threshold", type=int, default=127)

    parser.add_argument(
        "--mask-index-offset",
        type=int,
        default=0,
        help=(
            "Mask filename offset. Default 0 means frame index 0 uses 000000.png. "
            "Use 1 if your masks are named 000001.png for the first frame."
        ),
    )

    parser.add_argument(
        "--val-blend",
        type=float,
        default=0.7,
        help="How strongly to blend masked pixels toward --target-val. 0 keeps original V, 1 forces target V.",
    )

    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-every", type=int, default=10)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_video = Path(args.input_video).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()

    frames_dir = workdir / "frames"
    recolored_dir = workdir / "recolored_frames"
    overlay_dir = workdir / "overlays"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir does not exist: {mask_dir}")

    if output_video.exists() and not args.overwrite:
        raise FileExistsError(f"Output video already exists: {output_video}. Use --overwrite.")

    if recolored_dir.exists():
        shutil.rmtree(recolored_dir)
    recolored_dir.mkdir(parents=True, exist_ok=True)

    if args.save_overlays:
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    fps = ffprobe_fps(input_video)
    src_frame_count = ffprobe_frame_count(input_video)

    print(f"Input FPS: {fps}")
    print(f"Input decoded frames: {src_frame_count}")
    print(f"Mask dir: {mask_dir}")

    extract_frames(input_video, frames_dir, overwrite=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) != src_frame_count:
        print(
            f"Warning: extracted {len(frame_paths)} frames, "
            f"but ffprobe counted {src_frame_count}."
        )

    missing_masks: list[str] = []

    for idx, frame_path in enumerate(frame_paths):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        mask_idx = idx + args.mask_index_offset
        mask_path = mask_dir / f"{mask_idx:06d}.png"

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        if mask_u8.shape[:2] != frame_bgr.shape[:2]:
            mask_u8 = cv2.resize(
                mask_u8,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        out = hsv_recolor_preserve_shading(
            frame_bgr,
            mask_u8,
            target_hue=args.target_hue,
            target_sat=args.target_sat,
            value_factor=args.value_factor,
            target_val=args.target_val,
            val_blend=args.val_blend,
            alpha_blur=args.alpha_blur,
            mask_threshold=args.mask_threshold,
        )

        cv2.imwrite(str(recolored_dir / f"{idx + 1:06d}.jpg"), out)

        if args.save_overlays and idx % args.overlay_every == 0:
            overlay = make_overlay(frame_bgr, mask_u8, args.mask_threshold)
            cv2.imwrite(str(overlay_dir / f"{idx:06d}.jpg"), overlay)

    if missing_masks:
        print("\nFirst missing masks:")
        for p in missing_masks[:20]:
            print(f"  {p}")
        raise RuntimeError(
            f"Missing {len(missing_masks)} masks. "
            f"Check --mask-dir and --mask-index-offset."
        )

    out_frame_count = len(sorted(recolored_dir.glob("*.jpg")))
    if out_frame_count != len(frame_paths):
        raise RuntimeError(
            f"Recolored frame count mismatch: frames={len(frame_paths)}, recolored={out_frame_count}"
        )

    save_video_from_frames(recolored_dir, output_video, fps)

    dst_frame_count = ffprobe_frame_count(output_video)
    print(f"Output decoded frames: {dst_frame_count}")

    if src_frame_count != dst_frame_count:
        raise RuntimeError(
            f"Frame count mismatch: input={src_frame_count}, output={dst_frame_count}"
        )

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(recolored_dir, ignore_errors=True)

    print("\nDone.")
    print(f"Recolored video: {output_video}")
    if args.save_overlays:
        print(f"Debug overlays:   {overlay_dir}")


if __name__ == "__main__":
    main()
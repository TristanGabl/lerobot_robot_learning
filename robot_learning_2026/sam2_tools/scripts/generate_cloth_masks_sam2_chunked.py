#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
from collections import defaultdict
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


def link_or_copy_chunk_frames(
    frame_paths: list[Path],
    chunk_start: int,
    chunk_end: int,
    chunk_frames_dir: Path,
) -> list[Path]:
    """
    Create a temporary chunk directory where local frame 0 maps to global chunk_start.
    """
    if chunk_frames_dir.exists():
        shutil.rmtree(chunk_frames_dir)
    chunk_frames_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = frame_paths[chunk_start:chunk_end]

    for local_idx, src_path in enumerate(chunk_paths):
        dst_path = chunk_frames_dir / f"{local_idx:06d}.jpg"
        try:
            dst_path.symlink_to(src_path.resolve())
        except OSError:
            shutil.copy2(src_path, dst_path)

    return sorted(chunk_frames_dir.glob("*.jpg"))


def parse_correction(s: str) -> tuple[int, float, float, int]:
    """
    Parse correction syntax:
      GLOBAL_FRAME:X,Y,LABEL

    Convention:
      430:315,280,1 ->  positive cloth click
      430:210,190,0 ->  negative wrong-object/background click
    """
    try:
        frame_str, rest = s.split(":", 1)
        x_str, y_str, label_str = rest.split(",", 2)

        global_frame = int(frame_str)
        x = float(x_str)
        y = float(y_str)
        label = int(label_str)

        if global_frame < 0:
            raise ValueError("GLOBAL_FRAME must be >= 0")
        if label not in (0, 1):
            raise ValueError("LABEL must be 0 or 1")

        return global_frame, x, y, label

    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid correction '{s}'. Expected GLOBAL_FRAME:X,Y,LABEL, "
            f"for example 430:315,280,1 or 430:210,190,0"
        ) from e


def group_corrections_by_chunk(
    corrections: list[tuple[int, float, float, int]],
    chunk_size: int,
) -> dict[int, list[tuple[int, float, float, int]]]:
    # This
    grouped: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
    for global_frame, x, y, label in corrections:
        chunk_start = (global_frame // chunk_size) * chunk_size
        grouped[chunk_start].append((global_frame, x, y, label))
    return dict(grouped)


def build_initial_prompt_arrays(
    points: list[list[float]] | None,
    neg_points: list[list[float]] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    all_points = []
    all_labels = []

    if points:
        for p in points:
            all_points.append(p)
            all_labels.append(1)

    if neg_points:
        for p in neg_points:
            all_points.append(p)
            all_labels.append(0)

    if not all_points:
        return None, None

    return (
        np.array(all_points, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
    )


def validate_corrections_in_range(
    corrections: list[tuple[int, float, float, int]],
    total_frames: int,
) -> None:
    for global_frame, x, y, label in corrections:
        if not (0 <= global_frame < total_frames):
            raise ValueError(
                f"Correction frame {global_frame} is outside video frame range "
                f"0..{total_frames - 1}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-frame cloth masks for a video using SAM2, chunked to avoid RAM issues, "
            "with optional correction clicks on bad frames."
        )
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
        "--chunk-size",
        type=int,
        default=150,
        help="Frames per SAM2 chunk. 270 = 5 sec at 30 FPS.",
    )

    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help=(
            "0-based GLOBAL frame index used to document where the prompt was chosen. "
            "For chunked processing, the same prompt coordinates are reused on local frame 0 "
            "of every processed chunk."
        ),
    )

    # Not used: too imprecise
    """ parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Cloth box coordinates. Reused on local frame 0 of every processed chunk.",
    ) """

    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Positive cloth click. Can repeat. Reused on local frame 0 of every processed chunk.",
    )

    parser.add_argument(
        "--neg-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Negative background/robot/table click. Can repeat. Reused on local frame 0 of every processed chunk.",
    )

    parser.add_argument(
        "--correction",
        type=parse_correction,
        action="append",
        default=[],
        help=(
            "Correction point in format GLOBAL_FRAME:X,Y,LABEL. "
            "LABEL=1 positive cloth, LABEL=0 negative background/wrong object. "
            "Can be repeated."
        ),
    )

    parser.add_argument(
        "--only-correction-chunks",
        action="store_true",
        help=(
            "Only rerun chunks that contain at least one --correction frame. "
            "Requires existing output masks; corrected masks overwrite those chunk frames."
        ),
    )

    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save overlay debug images.",
    )
    parser.add_argument(
        "--overlay-every",
        type=int,
        default=30,
        help="Save one overlay every N global frames.",
    )

    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted full-video jpg frames in workdir.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output mask dir for a full run. Not used for --only-correction-chunks.",
    )

    args = parser.parse_args()

    input_video = Path(args.input_video).expanduser().resolve()
    output_mask_dir = Path(args.output_mask_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()

    frames_dir = workdir / "frames"
    chunk_frames_dir = workdir / "chunk_frames"
    overlay_dir = output_mask_dir.parent / f"{output_mask_dir.name}_overlays"

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    if args.box is None and not args.point and not args.correction:
        raise ValueError(
            "Provide either --box, at least one --point, or at least one --correction/prompt-at-frame."
        )

    corrections_by_chunk = group_corrections_by_chunk(args.correction, args.chunk_size)

    if args.only_correction_chunks:
        if not args.correction:
            raise ValueError("--only-correction-chunks requires at least one --correction.")
        if not output_mask_dir.exists():
            raise FileNotFoundError(
                f"--only-correction-chunks requires existing masks at {output_mask_dir}"
            )
        output_mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output_mask_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output mask dir already exists: {output_mask_dir}\n"
                    f"Use --overwrite or choose a new path."
                )
            shutil.rmtree(output_mask_dir)
        output_mask_dir.mkdir(parents=True, exist_ok=False)

    if args.save_overlays:
        # For full overwrite runs, clear old overlays.
        # For correction runs, keep existing overlays but overwrite corrected frame overlays.
        if overlay_dir.exists() and not args.only_correction_chunks:
            shutil.rmtree(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    src_frame_count = ffprobe_frame_count(input_video)
    print(f"Input decoded frames: {src_frame_count}")
    print(f"Chunk size: {args.chunk_size}")

    validate_corrections_in_range(args.correction, src_frame_count)

    if args.correction:
        print("\nCorrections requested:")
        for global_frame, x, y, label in args.correction:
            chunk_start = (global_frame // args.chunk_size) * args.chunk_size
            chunk_end = min(chunk_start + args.chunk_size, src_frame_count)
            local_frame = global_frame - chunk_start
            kind = "positive" if label == 1 else "negative"
            print(
                f"  global={global_frame}, local={local_frame}, "
                f"chunk=[{chunk_start},{chunk_end}), point=({x},{y}), {kind}"
            )

    extract_frames(input_video, frames_dir, overwrite=True)

    # ffmpeg image2 starts at 000001.jpg with this pattern, but our list is sorted,
    # and list index 0 corresponds to decoded/global frame 0.
    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if len(frame_paths) != src_frame_count:
        print(
            f"Warning: extracted {len(frame_paths)} frames, "
            f"but ffprobe counted {src_frame_count}."
        )

    if not frame_paths:
        raise RuntimeError(f"No frames extracted to {frames_dir}")

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

    box_np = np.array(args.box, dtype=np.float32) if args.box is not None else None
    points_np, labels_np = build_initial_prompt_arrays(args.point, args.neg_point)

    metadata_chunks = []
    total_masks_written = 0
    processed_chunks = 0
    skipped_chunks = 0

    for chunk_start in range(0, len(frame_paths), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(frame_paths))
        chunk_len = chunk_end - chunk_start
        chunk_corrections = corrections_by_chunk.get(chunk_start, [])

        if args.only_correction_chunks and not chunk_corrections:
            print(f"\nSkipping chunk global frames [{chunk_start}, {chunk_end}) no corrections.")
            skipped_chunks += 1
            continue

        processed_chunks += 1
        print(f"\nProcessing chunk global frames [{chunk_start}, {chunk_end}) ({chunk_len} frames)")

        local_frame_paths = link_or_copy_chunk_frames(
            frame_paths=frame_paths,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_frames_dir=chunk_frames_dir,
        )

        if len(local_frame_paths) != chunk_len:
            raise RuntimeError(
                f"Chunk frame count mismatch: expected {chunk_len}, got {len(local_frame_paths)}"
            )

        with torch.inference_mode():
            inference_state = predictor.init_state(
                video_path=str(chunk_frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            predictor.reset_state(inference_state)

            obj_id = 1

            # Optional generic prompt on local frame 0 of this chunk.
            # Only use this if --point/--neg-point or --box were provided.
            local_prompt_frame = 0

            if points_np is not None or box_np is not None:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_prompt_frame,
                    obj_id=obj_id,
                    points=points_np,
                    labels=labels_np,
                    box=box_np,
                )

            # Optional correction prompts inside this chunk.
            if chunk_corrections:
                print(f"Applying {len(chunk_corrections)} correction point(s) in this chunk.")

                corrections_by_local_frame: dict[int, list[tuple[float, float, int]]] = defaultdict(list)

                for global_frame, x, y, label in chunk_corrections:
                    if not (chunk_start <= global_frame < chunk_end):
                        raise RuntimeError(
                            f"Correction frame {global_frame} not inside chunk "
                            f"[{chunk_start}, {chunk_end})"
                        )

                    local_frame = global_frame - chunk_start
                    corrections_by_local_frame[local_frame].append((x, y, label))

                for local_frame, pts in sorted(corrections_by_local_frame.items()):
                    corr_points_np = np.array(
                        [[x, y] for x, y, label in pts],
                        dtype=np.float32,
                    )
                    corr_labels_np = np.array(
                        [label for x, y, label in pts],
                        dtype=np.int32,
                    )

                    print(
                        f"  local frame {local_frame}, "
                        f"global frame {chunk_start + local_frame}: "
                        f"{len(pts)} correction point(s)"
                    )

                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local_frame,
                        obj_id=obj_id,
                        points=corr_points_np,
                        labels=corr_labels_np,
                        box=None,
                    )

            chunk_masks_written = 0

            for out_local_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                global_idx = chunk_start + out_local_idx

                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
                mask = np.squeeze(mask).astype(bool)

                frame_bgr = cv2.imread(str(frame_paths[global_idx]))
                if frame_bgr is None:
                    raise RuntimeError(f"Could not read frame {frame_paths[global_idx]}")

                if mask.shape[:2] != frame_bgr.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                mask_u8 = mask.astype(np.uint8) * 255

                # Global naming inside this video: 000000.png is first decoded frame.
                mask_path = output_mask_dir / f"{global_idx:06d}.png"
                cv2.imwrite(str(mask_path), mask_u8)

                chunk_masks_written += 1
                total_masks_written += 1

                if args.save_overlays and global_idx % args.overlay_every == 0:
                    overlay = make_overlay(frame_bgr, mask)
                    cv2.imwrite(str(overlay_dir / f"{global_idx:06d}.jpg"), overlay)

            metadata_chunks.append(
                {
                    "chunk_start": chunk_start,
                    "chunk_end_exclusive": chunk_end,
                    "chunk_len": chunk_len,
                    "masks_written": chunk_masks_written,
                    "prompt_local_frame": local_prompt_frame,
                    "prompt_global_frame_used_for_coordinates": args.prompt_frame,
                    "corrections": [
                        {
                            "global_frame": global_frame,
                            "local_frame": global_frame - chunk_start,
                            "x": x,
                            "y": y,
                            "label": label,
                        }
                        for global_frame, x, y, label in chunk_corrections
                    ],
                }
            )

            del inference_state

        shutil.rmtree(chunk_frames_dir, ignore_errors=True)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    mask_paths = sorted(output_mask_dir.glob("*.png"))
    print(f"\nMasks currently in output dir: {len(mask_paths)}")
    print(f"Masks written in this run: {total_masks_written}")
    print(f"Processed chunks: {processed_chunks}")
    print(f"Skipped chunks: {skipped_chunks}")

    # In correction only mode, this validates that you already had a complete set.
    # In full mode, this validates this run generated a complete set.
    if len(mask_paths) != len(frame_paths):
        missing = len(frame_paths) - len(mask_paths)
        raise RuntimeError(
            f"Mask count mismatch: frames={len(frame_paths)}, masks={len(mask_paths)}, missing={missing}. "
            f"If this was a correction-only run, make sure the original full mask set exists first."
        )

    metadata = {
        "input_video": str(input_video),
        "frame_count_ffprobe": src_frame_count,
        "frames_extracted": len(frame_paths),
        "masks_in_output_dir": len(mask_paths),
        "masks_written_this_run": total_masks_written,
        "chunk_size": args.chunk_size,
        "processed_chunks": processed_chunks,
        "skipped_chunks": skipped_chunks,
        "only_correction_chunks": args.only_correction_chunks,
        "chunks": metadata_chunks,
        "prompt_frame": args.prompt_frame,
        "points": args.point,
        "neg_points": args.neg_point,
        "box": args.box,
        "corrections": [
            {
                "global_frame": global_frame,
                "x": x,
                "y": y,
                "label": label,
            }
            for global_frame, x, y, label in args.correction
        ],
        "model_cfg": args.model_cfg,
        "checkpoint": args.checkpoint,
        "mask_format": "uint8 PNG, 0 background, 255 cloth",
        "frame_indexing": "0-based mask filenames: 000000.png corresponds to first decoded frame of this input video",
        "note": (
            "Prompt coordinates are reused on local frame 0 of every processed chunk. "
            "Correction coordinates are GLOBAL frame indices converted to local chunk frame indices. "
            "Inspect overlays around correction frames and chunk boundaries."
        ),
    }

    metadata_path = output_mask_dir / "metadata.json"
    if args.only_correction_chunks and metadata_path.exists():
        # Preserve previous metadata by storing this run as a correction pass.
        try:
            previous = json.loads(metadata_path.read_text())
        except Exception:
            previous = {}

        correction_history = previous.get("correction_history", [])
        correction_history.append(metadata)

        previous["correction_history"] = correction_history
        previous["latest_correction_run"] = metadata
        previous["masks_in_output_dir"] = len(mask_paths)

        metadata_path.write_text(json.dumps(previous, indent=2))
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print("\nDone.")
    print(f"Mask dir: {output_mask_dir}")
    if args.save_overlays:
        print(f"Overlay dir: {overlay_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
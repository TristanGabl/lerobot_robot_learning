#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import bisect


def ffprobe_frame_count(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return int(data["streams"][0]["nb_read_frames"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--masks-root", default="masks")
    parser.add_argument("--camera", default="observation.images.front")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    masks_root = Path(args.masks_root)

    video_root = dataset_root / "videos" / args.camera
    if not video_root.exists():
        raise FileNotFoundError(video_root)

    video_paths = sorted(video_root.glob("chunk-*/*.mp4"))
    if not video_paths:
        raise RuntimeError(f"No mp4 files found under {video_root}")

    manifest = {args.camera: []}
    global_start = 0

    for video_path in video_paths:
        n = ffprobe_frame_count(video_path)
        rel_video = video_path.relative_to(dataset_root)

        # videos/observation.images.front/chunk-000/file-000.mp4
        # masks/observation.images.front/chunk-000/file-000
        chunk_name = video_path.parent.name
        file_stem = video_path.stem

        rel_mask_dir = masks_root / args.camera / chunk_name / file_stem

        manifest[args.camera].append(
            {
                "video_path": str(rel_video),
                "mask_dir": str(rel_mask_dir),
                "global_start": global_start,
                "global_end_exclusive": global_start + n,
                "num_frames": n,
            }
        )

        print(
            f"{rel_video}: frames={n}, "
            f"global=[{global_start}, {global_start + n}) -> {rel_mask_dir}"
        )

        global_start += n

    out_path = Path(args.out).expanduser().resolve() if args.out else dataset_root / masks_root / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nTotal video frames: {global_start}")
    print(f"Wrote manifest: {out_path}")


if __name__ == "__main__":
    main()
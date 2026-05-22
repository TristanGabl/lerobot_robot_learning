#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from huggingface_hub import snapshot_download

# NOTE: deprecated -> grayscale is now done in forward pass in diffusion code


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def require_bin(name: str) -> None:
    try:
        run([name, "-version"])
    except FileNotFoundError:
        raise RuntimeError(
            f"Missing required binary: {name}\n"
            f"Install it, e.g. on Ubuntu:\n"
            f"  sudo apt update && sudo apt install -y ffmpeg"
        )


def copy_non_video_files(src: Path, dst: Path, overwrite: bool = False) -> None:
    """
    Copy the full LeRobot dataset except videos/.
    data/ and meta/ stay byte-identical, so states/actions/timestamps are unchanged.
    """
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {dst}\n"
                f"Delete it first or rerun with --overwrite."
            )
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=False)

    for item in src.iterdir():
        if item.name == "videos":
            continue

        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def get_dataset_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = json.loads(info_path.read_text())

    if "fps" in info:
        return float(info["fps"])

    features = info.get("features", {})
    for feature in features.values():
        if not isinstance(feature, dict):
            continue

        for maybe_video_info in (
            feature.get("info", {}),
            feature.get("video", {}),
            feature,
        ):
            if not isinstance(maybe_video_info, dict):
                continue

            for key in ("video.fps", "fps"):
                if key in maybe_video_info:
                    return float(maybe_video_info[key])

    raise KeyError(f"Could not find dataset fps in {info_path}")


def ffprobe_video_stream(path: Path) -> dict:
    """
    Return ffprobe info for the first video stream.
    Uses -count_frames so nb_read_frames is based on actual decoded frames.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = run(cmd)
    data = json.loads(proc.stdout)

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")

    return streams[0]


def decoded_frame_count(path: Path) -> int:
    stream = ffprobe_video_stream(path)

    # Prefer actual decoded frame count.
    value = stream.get("nb_read_frames")
    if value not in (None, "N/A"):
        return int(value)

    # Fall back to container metadata only if decoded count is unavailable.
    value = stream.get("nb_frames")
    if value not in (None, "N/A"):
        return int(value)

    raise RuntimeError(f"Could not determine frame count for {path}")


def parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    return float(Fraction(rate))


def convert_video_to_grayscale_ffmpeg(
    src_mp4: Path,
    dst_mp4: Path,
    *,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
    force_cfr: bool,
) -> None:
    """
    Convert to grayscale-looking MP4.

    Important:
    - Keeps one decoded input frame -> one encoded output frame.
    - Does not use OpenCV.
    - Outputs yuv420p H.264 by default, which is broadly readable.
    - The video is grayscale in content, but still normal MP4 video for LeRobot loaders.
    """
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)

    vf = "format=gray,format=yuv420p"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src_mp4),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-c:v",
        codec,
    ]

    if codec == "libx264":
        cmd += [
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]

    # For frame-perfect count, passthrough avoids ffmpeg duplicating/dropping frames.
    # If you absolutely need constant frame rate timestamps, use --force-cfr.
    if force_cfr:
        cmd += [
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
        ]
    else:
        cmd += [
            "-fps_mode",
            "passthrough",
        ]

    # Make MP4 friendlier to loaders/players.
    cmd += [
        "-movflags",
        "+faststart",
        str(dst_mp4),
    ]

    proc = run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_mp4}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}"
        )


def convert_all_videos(
    src: Path,
    dst: Path,
    *,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
    force_cfr: bool,
) -> None:
    src_video_root = src / "videos"
    dst_video_root = dst / "videos"

    if not src_video_root.exists():
        raise FileNotFoundError(f"No videos/ folder found in source dataset: {src}")

    mp4s = sorted(src_video_root.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 files found under {src_video_root}")

    print(f"Using dataset FPS: {fps}")
    print(f"Using encoder:     {codec}")
    print(f"Found {len(mp4s)} video shard(s).")

    failures: list[str] = []

    for i, src_mp4 in enumerate(mp4s, start=1):
        rel = src_mp4.relative_to(src_video_root)
        dst_mp4 = dst_video_root / rel

        print(f"\n[{i}/{len(mp4s)}] {rel}")

        src_info_before = ffprobe_video_stream(src_mp4)
        src_count = decoded_frame_count(src_mp4)

        print(
            "  source:",
            f"{src_info_before.get('codec_name')},",
            f"{src_info_before.get('pix_fmt')},",
            f"{src_info_before.get('width')}x{src_info_before.get('height')},",
            f"frames={src_count},",
            f"avg_fps={parse_rate(src_info_before.get('avg_frame_rate'))}",
        )

        convert_video_to_grayscale_ffmpeg(
            src_mp4,
            dst_mp4,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=preset,
            force_cfr=force_cfr,
        )

        dst_info = ffprobe_video_stream(dst_mp4)
        dst_count = decoded_frame_count(dst_mp4)

        print(
            "  output:",
            f"{dst_info.get('codec_name')},",
            f"{dst_info.get('pix_fmt')},",
            f"{dst_info.get('width')}x{dst_info.get('height')},",
            f"frames={dst_count},",
            f"avg_fps={parse_rate(dst_info.get('avg_frame_rate'))}",
        )

        if src_count != dst_count:
            failures.append(f"{rel}: source={src_count}, output={dst_count}")

    if failures:
        raise RuntimeError(
            "Frame count mismatch detected. Do NOT use this dataset.\n"
            + "\n".join(failures)
        )

    print("\nAll converted videos passed decoded frame-count validation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local grayscale copy of a LeRobot dataset from a Hugging Face dataset repo. "
            "Copies data/ and meta/ unchanged, converts videos/ with ffmpeg, and validates frame counts."
        )
    )

    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--codec",
        default="libx264",
        help="Video encoder. Default: libx264. Alternative if unavailable: mpeg4",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality. Lower is better/larger. Default: 18.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset. Try ultrafast for speed, slow for smaller files.",
    )
    parser.add_argument(
        "--force-cfr",
        action="store_true",
        help=(
            "Force constant frame rate at dataset FPS. Usually leave this OFF for strict "
            "one-input-frame to one-output-frame behavior."
        ),
    )

    args = parser.parse_args()

    require_bin("ffmpeg")
    require_bin("ffprobe")

    print(f"Downloading dataset snapshot: {args.repo_id}")

    src_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=[
                "data/**",
                "meta/**",
                "videos/**",
                "README*",
            ],
        )
    )

    out_path = Path(args.out).expanduser().resolve()

    print(f"Source snapshot: {src_path}")
    print(f"Output dataset:   {out_path}")

    copy_non_video_files(src_path, out_path, overwrite=args.overwrite)

    fps = args.fps if args.fps is not None else get_dataset_fps(src_path)

    convert_all_videos(
        src=src_path,
        dst=out_path,
        fps=fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        force_cfr=args.force_cfr,
    )

    print("\nDone.")
    print(f"Grayscale dataset written to: {out_path}")
    print("\nSanity checks:")
    print(f"  ffprobe -v error -count_frames -select_streams v:0 "
          f"-show_entries stream=codec_name,pix_fmt,nb_read_frames "
          f"-of default=nw=1 {out_path}/videos/<camera_name>/chunk-000/file-000.mp4")
    print(f"  ffplay {out_path}/videos/<camera_name>/chunk-000/file-000.mp4")


if __name__ == "__main__":
    main()
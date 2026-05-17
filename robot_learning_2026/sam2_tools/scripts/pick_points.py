#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2


def ffprobe_frame_count(path: Path) -> int | None:
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
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        return None


import tempfile
import subprocess
from pathlib import Path
import cv2


def read_frame(video_path: Path, frame_idx: int):
    """
    Read a specific frame using ffmpeg instead of cv2.VideoCapture.
    This is more robust for AV1 videos.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "frame.png"

        # select exact 0-based frame number
        vf = f"select=eq(n\\,{frame_idx})"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(out_path),
        ]

        subprocess.run(cmd, check=True)

        frame = cv2.imread(str(out_path))
        if frame is None:
            raise RuntimeError(f"ffmpeg extracted frame {frame_idx}, but OpenCV could not read {out_path}")

        return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Click SAM2 positive/negative points on a chosen video frame."
    )
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale only. Printed coordinates are converted back to original image coords.",
    )
    parser.add_argument(
        "--correction-mode",
        action="store_true",
        help="Print --correction FRAME:X,Y,LABEL instead of --point/--neg-point.",
    )
    parser.add_argument(
        "--save-image",
        default=None,
        help="Optional path to save the clicked visualization image.",
    )
    parser.add_argument("--image", default=None, help="Optional already-extracted frame image.")

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if args.frame < 0:
        raise ValueError("--frame must be >= 0")

    if args.image is None and args.video is None:
        raise ValueError("You must provide either --image or --video")

    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        video_name = video_path.name
    else:
        video_path = None
        video_name = Path(args.image).expanduser().resolve().name

    if args.image is not None:
        image_path = Path(args.image).expanduser().resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
    else:
        # Here video_path is guaranteed not None because of the check above.
        frame_count = ffprobe_frame_count(video_path)
        if frame_count is not None and args.frame >= frame_count:
            raise ValueError(f"--frame {args.frame} outside video range 0..{frame_count - 1}")

        frame = read_frame(video_path, args.frame)
    
    original = frame.copy()
    vis = frame.copy()

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    clicked: list[tuple[int, int, int]] = []  # x, y, label

    #win = f"{video_path.name} frame={args.frame} | left=pos right=neg u=undo c=clear q=quit"
    win = f"{video_name} frame={args.frame} | left=pos right=neg u=undo c=clear q=quit"

    def redraw():
        nonlocal vis
        vis = original.copy()

        for x, y, label in clicked:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            text = "P" if label == 1 else "N"
            cv2.circle(vis, (x, y), 7, color, -1)
            cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(
                vis,
                text,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        header = f"frame={args.frame}  left=positive  right=negative  u=undo  c=clear  q=quit"
        cv2.putText(
            vis,
            header,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def print_click(x: int, y: int, label: int):
        if args.correction_mode:
            print(f"--correction {args.frame}:{x},{y},{label}")
        else:
            if label == 1:
                print(f"--point {x} {y}")
            else:
                print(f"--neg-point {x} {y}")

    def click(event, x_display, y_display, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x = int(round(x_display / args.scale))
        y = int(round(y_display / args.scale))

        h, w = original.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # Shift + left click = negative, plain left click = positive
        label = 0 if (flags & cv2.EVENT_FLAG_SHIFTKEY) else 1

        clicked.append((x, y, label))
        print_click(x, y, label)
        redraw()

    redraw()

    #cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, click)

    while True:
        if args.scale != 1.0:
            shown = cv2.resize(
                vis,
                None,
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_AREA if args.scale < 1 else cv2.INTER_LINEAR,
            )
        else:
            shown = vis

        cv2.imshow(win, shown)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("u"):
            if clicked:
                removed = clicked.pop()
                print(f"# undo {removed}")
                redraw()
        elif key == ord("c"):
            clicked.clear()
            print("# cleared")
            redraw()

    if args.save_image:
        out = Path(args.save_image).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"Saved clicked visualization: {out}")

    cv2.destroyAllWindows()

    print("\n# Copy/paste:")
    for x, y, label in clicked:
        if args.correction_mode:
            print(f"  --correction {args.frame}:{x},{y},{label} \\")
        else:
            if label == 1:
                print(f"  --point {x} {y} \\")
            else:
                print(f"  --neg-point {x} {y} \\")


if __name__ == "__main__":
    main()
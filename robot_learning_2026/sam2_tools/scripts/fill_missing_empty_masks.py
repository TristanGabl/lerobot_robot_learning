#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


MASK_RE = re.compile(r"^(\d+)\.png$")


def find_numbered_masks(mask_dir: Path) -> dict[int, Path]:
    masks: dict[int, Path] = {}

    for p in mask_dir.glob("*.png"):
        m = MASK_RE.match(p.name)
        if not m:
            continue
        masks[int(m.group(1))] = p

    return masks


def infer_mask_shape(mask_paths: list[Path]) -> tuple[int, int]:
    for p in mask_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.shape[:2]

    raise RuntimeError("Could not read any existing PNG mask to infer image size.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing numbered PNG masks with empty all-zero masks."
    )
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Expected total frame count. If omitted, fills gaps from min existing index to max existing index.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First expected mask index. Default: 0.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Filename zero padding. Default: 6 -> 000123.png.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be created.",
    )

    args = parser.parse_args()

    mask_dir = Path(args.mask_dir).expanduser().resolve()
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    masks = find_numbered_masks(mask_dir)
    if not masks:
        raise RuntimeError(f"No numbered PNG masks found in {mask_dir}")

    h, w = infer_mask_shape(list(masks.values()))

    if args.num_frames is not None:
        expected_indices = range(args.start_index, args.start_index + args.num_frames)
    else:
        #expected_indices = range(min(masks), max(masks) + 1)
        expected_indices = range(args.start_index, max(masks) + 1)

    missing = [i for i in expected_indices if i not in masks]

    print(f"Mask dir:          {mask_dir}")
    print(f"Existing masks:    {len(masks)}")
    print(f"Mask shape:        {w}x{h}")
    print(f"Missing masks:     {len(missing)}")

    if not missing:
        print("Nothing to fill.")
        return

    print("First missing:")
    for i in missing[:20]:
        print(f"  {i:0{args.digits}d}.png")

    if args.dry_run:
        print("\nDry run only. No files written.")
        return

    empty = np.zeros((h, w), dtype=np.uint8)

    for i in missing:
        out_path = mask_dir / f"{i:0{args.digits}d}.png"
        ok = cv2.imwrite(str(out_path), empty)
        if not ok:
            raise RuntimeError(f"Failed to write {out_path}")

    print(f"\nWrote {len(missing)} empty mask(s).")


if __name__ == "__main__":
    main()
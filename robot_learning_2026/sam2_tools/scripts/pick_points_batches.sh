#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/full_fold_general_tristan/videos/observation.images.front/chunk-000/file-005.mp4"
LOG_FILE="out-005.log"
PREFETCH_FRAMES_DIR="point_frames_005"
CHUNK_SIZE=540
N_CHUNKS=40

mkdir -p "$PREFETCH_FRAMES_DIR"

PREFETCH=true
PICK_POINTS=true

# Optional: clear old log
: > "$LOG_FILE"

# PREFETCH FRAMES
if [ "$PREFETCH" = true ]; then
    echo "Prefetching frames into $PREFETCH_FRAMES_DIR..."

    for i in $(seq 0 $((N_CHUNKS - 1))); do
        frame=$((i * CHUNK_SIZE))
        out="$PREFETCH_FRAMES_DIR/frame_${frame}.png"

        echo "Extracting exact frame $frame -> $out"

        # Remove old file first so we don't accidentally keep a stale PNG
        rm -f "$out"

        ffmpeg -hide_banner -loglevel error -y \
            -i "$VIDEO_PATH" \
            -vf "select=eq(n\,${frame})" \
            -frames:v 1 \
            "$out" || {
                echo "ffmpeg failed at frame $frame. Stopping prefetch."
                break
            }

        if [ ! -s "$out" ]; then
            echo "No PNG produced for frame $frame, probably past end of video. Stopping prefetch."
            rm -f "$out"
            break
        fi
    done
fi

# PICK POINTS FROM PREFETCHED FRAMES
# Sort naturally: frame_5940.png, frame_6480.png, ..., frame_10260.png
if [ "$PICK_POINTS" = true ]; then
    find "$PREFETCH_FRAMES_DIR" -maxdepth 1 -type f -name 'frame_*.png' \
        | sort -V \
        | while read -r image_path; do

            filename="$(basename "$image_path")"
            frame="${filename#frame_}"
            frame="${frame%.png}"

            echo "========================================" | tee -a "$LOG_FILE"
            echo "Picking points for frame $frame / image $image_path" | tee -a "$LOG_FILE"
            echo "========================================" | tee -a "$LOG_FILE"

            python scripts/pick_points.py \
                --video "$VIDEO_PATH" \
                --image "$image_path" \
                --correction-mode \
                --frame "$frame" \
                2>&1 | tee -a "$LOG_FILE"
        done
fi
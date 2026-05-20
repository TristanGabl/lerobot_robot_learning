#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="/home/clompa/spaces/robot_learning/lerobot_robot_learning/robot_learning_2026/sam2_tools/scripts"

failed=()

for i in $(seq -w 001 011); do
    script="$SCRIPT_DIR/gen_masks_full_fold_improved_general_${i}.sh"

    echo "========================================"
    echo "Running $script"
    echo "========================================"

    if "$script"; then
        echo "OK: $script"
    else
        echo "FAILED: $script"
        failed+=("$script")
    fi
done

echo "========================================"
echo "Finished all scripts."
echo "Failed scripts: ${#failed[@]}"
printf '%s\n' "${failed[@]}"
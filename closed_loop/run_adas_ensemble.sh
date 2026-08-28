#!/bin/bash

# Wrapper script to run the CARLA ADAS closed-loop module directly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/weights"

find_weight() {
    pattern="$1"
    find "$WEIGHTS_DIR" -maxdepth 1 -type f \( -iname "$pattern" -o -iname "*${pattern}*" \) 2>/dev/null | head -n 1
}

YOLO_MODEL="${YOLO_MODEL:-$(find_weight 'yolo*.pt')}"
RTDETR_MODEL="${RTDETR_MODEL:-$(find_weight 'rtdetr*.pt')}"
FASTER_MODEL="${FASTER_MODEL:-$(find_weight 'faster*.pth')}"
DEPTH_MODEL="${DEPTH_MODEL:-$REPO_ROOT/DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth}"

if [ -z "$YOLO_MODEL" ] || [ -z "$RTDETR_MODEL" ] || [ -z "$FASTER_MODEL" ]; then
    echo "Missing one or more weights files in $WEIGHTS_DIR"
    echo "Expected: YOLO .pt, RT-DETR .pt, Faster R-CNN .pth"
    echo "You can override them explicitly with:"
    echo "  YOLO_MODEL=/path/to/yolo.pt RTDETR_MODEL=/path/to/rtdetr.pt FASTER_MODEL=/path/to/faster.pth bash closed_loop/run_adas_ensemble.sh"
    exit 1
fi

cd "$SCRIPT_DIR"

python3 main.py \
    --mode ensemble \
    --host localhost \
    --port 2000 \
    --log-level INFO \
    --yolo-model "$YOLO_MODEL" \
    --rtdetr-model "$RTDETR_MODEL" \
    --faster-model "$FASTER_MODEL" \
    --depth-model "$DEPTH_MODEL" \
    --num-classes 9 \
    --conf-threshold 0.40 \
    --cruise-throttle 0.35 \
    --warning-distance 15.0 \
    --brake-distance 7.0 \
    --lane-width 3.5 \
    --max-speed 30.0 \
    "$@"

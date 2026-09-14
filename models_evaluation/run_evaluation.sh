#!/bin/bash

# Wrapper script to run Closed-Loop CARLA Quantitative Evaluation Module

# https://drive.google.com/file/d/11rEdIA_swZF-xADYzX6xYdBILX_N-ip8/view?usp=drive_link
YOLO_MODEL="/home/ubuntu/Downloads/best_yolov8n.pt"

# https://drive.google.com/file/d/1-uQ9iFZrLiHYOb6_WvEk8xmqzhaRte4n/view?usp=drive_link
RTDETR_MODEL="/home/ubuntu/Downloads/rt-detr_best.pt"

# https://drive.google.com/file/d/1BIthy7YeYUof0PahNvXEVxp-DEbSh3Zx/view?usp=drive_link
FASTER_MODEL="/home/ubuntu/Downloads/fastercnn_best.pt"

# https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/blob/main/depth_anything_v2_metric_hypersim_vits.pth
DEPTH_MODEL="DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"

python3 evaluate_scenarios.py \
    --perception_mode ensemble \
    --host localhost \
    --port 2000 \
    --yolo-model "$YOLO_MODEL" \
    --faster-model "$FASTER_MODEL" \
    --rtdetr-model "$RTDETR_MODEL" \
    --depth-model "$DEPTH_MODEL" \
    --num-classes 9 \
    --conf-threshold 0.40 \
    --cruise-throttle 0.35 \
    --warning-distance 15.0 \
    --brake-distance 7.0 \
    --runs 5 \
    --max-speed 30.0 \
    $1

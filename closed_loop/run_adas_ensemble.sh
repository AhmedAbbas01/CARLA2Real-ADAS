#!/bin/bash

# Wrapper script to run the CARLA ADAS closed-loop module directly

YOLO_MODEL="~/Downloads/best_yolov8n.pt"
RTDETR_MODEL="rtdetr.pt"
FASTER_MODEL="faster_rcnn.pth"
DEPTH_MODEL="DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"

python3 main.py \
    --mode ensemble \
    --host localhost \
    --port 2000 \
    --log-level INFO \
    --yolo-model "$YOLO_MODEL" \
    --rtdetr-model "$RTDETR_MODEL" \
    --faster-model "$FASTER_MODEL" \
    --depth-model "$DEPTH_MODEL" \
    --num-classes 8 \
    --conf-threshold 0.40 \
    --cruise-throttle 0.35 \
    --warning-distance 15.0 \
    --brake-distance 7.0 \
    --lane-width 3.5 \
    --max-speed 30.0 \
    --visualize

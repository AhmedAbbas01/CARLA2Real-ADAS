# CARLA Closed-Loop ADAS Module

This module provides a real-time, vision-based Advanced Driver Assistance System (ADAS) for the CARLA simulator. It utilizes object detection and metric depth estimation to autonomously navigate the ego-vehicle, avoid collisions, and perform emergency braking.

## Architecture

The closed-loop system is modularized into several components to separate concerns and allow for robust expandability:

- **`main.py`**: The application entry point. Parses configuration arguments and orchestrates the perception and controller layers.
- **`perception.py`**: Manages the inference layers. Supports running a single YOLO model or an ensemble of YOLO, RT-DETR, and Faster R-CNN, coupled with DepthAnythingV2 for metric depth extraction.
- **`controller.py`**: Interacts directly with the CARLA API. It processes vehicle telemetry, calculates steering targets, evaluates danger levels, and issues final throttle/brake commands.
- **`ensemble.py`**: Executes Weighted Boxes Fusion (WBF) algorithm to reconcile bounding boxes across multiple models and maps median depth distances to predictions.

## Prerequisites

Ensure you have the required weights downloaded and placed in the appropriate directories prior to execution. Depending on your mode (`single` or `ensemble`), you will need:
- YOLOv8 (`.pt`)
- RT-DETR (`.pt`)
- Faster R-CNN (`.pth`)
- DepthAnythingV2 Metric Depth Checkpoints (`.pth`)

> Note: At this moment, the script loads the pretrained DepthAnythingV2 checkpoint. You need to clone [the repo](https://github.com/DepthAnything/Depth-Anything-V2) and download the **metric_depth pretrained models** before running.

## Usage

You can execute the simulation using the provided shell script, which cleanly exposes all necessary configurations for the ADAS system:

```bash
./run_adas_ensemble.sh --visualize
```
> Note: `--visualize` is optional and can be passed to toggle the OpenCV visualizer to monitor detections and real-time telemetry.

### Key Parameters

| Argument | Default | Description |
|---|---|---|
| `--mode` | `ensemble` | Set the perception pipeline (`single` or `ensemble`). |
| `--cruise-throttle` | `0.35` | Safe cruising throttle value when no obstacles are present. |
| `--warning-distance` | `15.0` | Distance threshold (meters) to trigger deceleration. |
| `--brake-distance` | `7.0` | Distance threshold (meters) to trigger emergency braking. |
| `--lane-width` | `3.5` | Physical ego-lane width (meters) used to filter out adjacent-lane objects. |
| `--max-speed` | `30.0` | Maximum safe cruising speed in km/h. |

## Limitations

- No true TTC
    - The current implementation does not yet calculate physical Time-To-Collision.
- Detector errors affect control
    - False negatives can prevent braking. False positives can cause unnecessary braking.

## Next Steps

Possible extensions are:

- Time-To-Collision (TTC)
- Relative velocity estimation
- CARLA ground-truth distance for controlled validation
- More realistic AEB thresholds
- Lane-aware obstacle filtering
- Detection/control logging
- Quantitative AEB experiments such as stopping distance and collision rate

## Expected Behavior

To validate the expected behavior, introduce an obstacle and observe the Ego controller transition:
```bash
# from: SAFE
Throttle=0.35 Brake=0.00
# to: WARNING / BRAKING
Throttle=0.00 Brake=0.40
# and finally: EMERGENCY BRAKE
Throttle=0.00 Brake=1.00
```
while confirming that the CARLA ego vehicle physically decelerates and stops.

# CARLA YOLOv8 Closed-Loop ADAS Test

## 1. Overview

This script implements a proof-of-concept closed-loop Automatic
Emergency Braking (AEB) system in CARLA using a fine-tuned YOLOv8
detector.

### Pipeline:

- CARLA RGB Camera -> Fine-tuned YOLOv8 (best.pt) -> Object detections -> ADAS risk logic -> Throttle / Brake command -> CARLA ego vehicle -> Next camera frame

- YOLO performs perception only. The ADAS logic converts YOLO detections into vehicle-control commands.

## 2. Purpose

The script verifies that:

1.  CARLA supplies live RGB frames.
2.  The trained YOLOv8 model processes those frames.
3.  Relevant road users are detected.
4.  Detection information reaches the ADAS logic.
5.  The ADAS logic selects throttle/brake commands.
6.  Those commands are applied back to the CARLA vehicle.

This forms the closed perception-to-control feedback loop.

## 3. Requirements

-   CARLA simulator
-   Python 3
-   PyTorch
-   Ultralytics
-   NumPy
-   CARLA Python API
-   OpenCV only if visualization is enabled

Example:
```bash
pip install ultralytics numpy
```
The CARLA Python API must be importable from the environment running the
script.

## 4. Trained Model

The script loads the fine-tuned YOLO checkpoint, for example:
```python
MODEL_PATH = "runs/yolov8s_detect/weights/best.pt"
```
Make sure this points to your actual best.pt.

>Note: The closed-loop script performs inference only. It does not retrain YOLO.

## 5. CARLA Connection

Default connection configurations:
```python
CARLA_HOST = "localhost"
CARLA_PORT = 2000
```
CARLA must already be running before starting the Python script.

## 6. Camera sensor

The ego vehicle has an RGB camera configured approximately as:
```
Resolution: 1280 x 720
FOV: 90 degrees
```
Each frame is converted to a NumPy image and passed to YOLO.

## 7. YOLO Inference

Typical settings:
```python
IMGSZ = 1280
CONF_THRESHOLD = 0.40
```
Detections below the confidence threshold are ignored.

Relevant obstacle classes can include:

-   Car
-   Truck
-   Bus
-   Motorcycle
-   Bicycle
-   Pedestrians


## [To be updated] 8. Current Collision-Risk Logic

The current proof-of-concept does NOT calculate true distance in meters.

Instead, it uses bounding-box height relative to image height as a
simple closeness proxy:

    small bounding box -> probably farther away
    large bounding box -> probably closer

It also checks horizontal bounding-box position so that objects well
outside the approximate ego-lane region do not trigger braking.

Example thresholds:

    WARNING_BOX_RATIO = 0.25
    BRAKE_BOX_RATIO = 0.40

These are experimental thresholds, not physical distances.

## 9. Vehicle Control

For this AEB test, CARLA autopilot is disabled `vehicle.set_autopilot(False)`

The script applies VehicleControl directly:

```python
# Safe:
Throttle = 0.35
Brake = 0.00
# Warning:
Throttle = 0.00
Brake = 0.40
# Emergency:
Throttle = 0.00
Brake = 1.00
```
and applies the steering value based on the CARLA lanes objects

## 10. How to Run

### Step 1 - Start CARLA
Launch CARLA and wait until the world is fully loaded.
### Step 2 - Verify best.pt
Confirm that your trained checkpoint exists at the path configured in `MODEL_PATH`.
### Step 3 - Run the script
Either in cli mode:
```bash
python closed_loop/clsloop.py --model-path $PWD/fine-tuning/runs/yolov8n_detect/weights/best.pt
```
Or in visualization mode, where a cv2 window is opened to show the detected objects and the driving decision
```bash
python closed_loop/clsloop.py --model-path $PWD/fine-tuning/runs/yolov8n_detect/weights/best.pt --visualize
```

### Step 4 - Observe the terminal
```bash
# Normal output with no relevant obstacle:
SAFE | Throttle=0.35 Brake=0.00

# Example as a detected vehicle approaches:
SAFE                | Throttle: 0.35  Brake: 0.00
WARNING / BRAKING   | Throttle: 0.00  Brake: 0.40
EMERGENCY BRAKE     | Throttle: 0.00  Brake: 1.00

# Brake=1.00 means the ADAS controller has commanded full braking.
```

## 11. Recommended Test

For the first controlled experiment:

1. Start CARLA.
2. Run clsloop.py (either in cli or visualization mode).
3. Run the example script (provided by Carla) to generate traffic objects 
```bash
cd ../CarlaUE5/PythonAPI/
python generate_traffic.py
```
5. Let the ego vehicle approach the obstacle.
6. Observe the YOLO/ADAS terminal output.
7. Verify the transition from normal throttle to partial braking.
8. Verify that Brake=1.00 is eventually commanded.
9. Confirm visually in CARLA that the ego vehicle decelerates/stops.

This verifies that detector output is feeding back into CARLA vehicle control.

## 12. Meaning of “No obstacle”

Output such as:
```bash
No obstacle | Throttle=0.35 Brake=0.00
```
is normal when there is no qualifying detection in the approximate ego-lane region.

It also confirms that the frame reached the perception/control pipeline
and a control decision was generated.

## 13. Important Limitations

- [To be updated] Bounding-box size is not true distance
    - The current system uses bounding-box size as a heuristic. Do not report bbox_ratio as distance in meters.
- [To be updated] Thresholds need validation
    - WARNING_BOX_RATIO and BRAKE_BOX_RATIO are experimental and should be tuned systematically.
- No true TTC
    - The current implementation does not yet calculate physical Time-To-Collision.
- Detector errors affect control
    - False negatives can prevent braking. False positives can cause unnecessary braking.

## 14. OpenCV Visualization Error

Some OpenCV installations do not support `cv2.imshow()`.

If you receive an error saying the function is not implemented, remove/comment:
```python
cv2.imshow(...)
cv2.waitKey(...)
cv2.destroyAllWindows()
```
The closed-loop controller does not require the OpenCV window.

## 15. Vehicle Blueprint Error

Some CARLA builds do not contain `vehicle.tesla.model3`. If that produces `“RuntimeError: index out of range”`, select an available vehicle from `world.get_blueprint_library().filter("vehicle.*")`

A robust script can try the Tesla first and fall back to another vehicle.

## 16. Stopping the Program

Press: `Ctrl + C`: The cleanup section should stop/destroy the camera and destroy the spawned ego vehicle.

## 17. Next Steps

After the basic closed-loop test is verified, possible extensions are:

- Metric depth/distance estimation
- Time-To-Collision (TTC)
- Relative velocity estimation
- CARLA ground-truth distance for controlled validation
- More realistic AEB thresholds
- Lane-aware obstacle filtering
- YOLO + RT-DETR + Faster R-CNN ensemble
- Weighted Box Fusion
- Detection/control logging
- Quantitative AEB experiments such as stopping distance and collision rate

## 18. Summary

The main final test is to introduce an obstacle and observe the controller transition:
```bash
# from:
Throttle=0.35 Brake=0.00
# to:
Throttle=0.00 Brake=0.40
# and finally:
Throttle=0.00 Brake=1.00
```
while confirming that the CARLA ego vehicle physically decelerates and stops.

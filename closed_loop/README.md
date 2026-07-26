# CARLA YOLOv8 Closed-Loop ADAS Test

## 1. Overview

This script implements a proof-of-concept closed-loop Automatic
Emergency Braking (AEB) system in CARLA using a fine-tuned YOLOv8
detector.

### Pipeline:

- CARLA RGB Camera -> Fine-tuned YOLOv8 (best.pt) -> Object detections -> Depth Anything V2 -> Object distances -> ADAS risk logic -> Throttle / Brake command -> CARLA ego vehicle -> Next camera frame

- YOLO performs perception only. The ADAS logic converts YOLO detections into vehicle-control commands.

## 2. Purpose

The script verifies that:

1.  CARLA supplies live RGB frames.
2.  The trained YOLOv8 model processes those frames.
3.  Relevant road users are detected.
4.  The depth estimator model (Depth Anything V2) processes the frames.
5.  Detection information reaches the ADAS logic.
6.  The ADAS logic selects throttle/brake commands.
7.  Those commands are applied back to the CARLA vehicle.

This forms the closed perception-to-control feedback loop.

## 3. YOLOv8 Fine-tuned Model

The script loads the fine-tuned YOLO checkpoint, for example:
```python
--yolo-model-path "~/Downloads/best.pt"
```
Make sure this points to your actual best.pt.

>Note: The closed-loop script performs inference only. It does not retrain YOLO.

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

## 4. Depth Anything V2 Fine-tuned Model

At this moment, the script loads the pretrained Depth Anything V2 checkpoint. You need to clone [the repo](https://github.com/DepthAnything/Depth-Anything-V2) and download the **metric_depth pretrained models** before running. then you may pass the model path as follows:
```python
--depth-model-path "$PWD/closed_loop/DepthAnythingV2/"
```

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
### Step 2 - Verify models checkpoints
Confirm that your trained checkpoints exist at the specified paths.
### Step 3 - Run the script
Either in cli mode:
```bash
python closed_loop/clsloop.py --yolo-model-path $PWD/fine-tuning/runs/yolov8n_detect/weights/best.pt --depth-model-path $PWD/closed_loop/DepthAnythingV2/
```
Or in visualization mode, where a cv2 window is opened to show the detected objects and the driving decision
```bash
python closed_loop/clsloop.py --yolo-model-path $PWD/fine-tuning/runs/yolov8n_detect/weights/best.pt --depth-model-path $PWD/closed_loop/DepthAnythingV2/ --visualize
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

- No true TTC
    - The current implementation does not yet calculate physical Time-To-Collision.
- Detector errors affect control
    - False negatives can prevent braking. False positives can cause unnecessary braking.

## 14. Stopping the Program

Press: `Ctrl + C`: The cleanup section should stop/destroy the camera and destroy the spawned ego vehicle.

## 15. Next Steps

After the basic closed-loop test is verified, possible extensions are:

- Time-To-Collision (TTC)
- Relative velocity estimation
- CARLA ground-truth distance for controlled validation
- More realistic AEB thresholds
- Lane-aware obstacle filtering
- YOLO + RT-DETR + Faster R-CNN ensemble
- Weighted Box Fusion
- Detection/control logging
- Quantitative AEB experiments such as stopping distance and collision rate

## 16. Summary

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

import carla
import cv2
import numpy as np
import time
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

MODEL_PATH = "yolov8n.pt"

CONF_THRESHOLD = 0.40
IMGSZ = 1280

# Classes that can trigger braking
DANGER_CLASSES = {
    "Car",
    "Truck",
    "Bus",
    "Motorcycle",
    "Bicycle",
    "Pedestrians",
}

# Normal driving
CRUISE_THROTTLE = 0.35

# Bounding-box based braking thresholds.
# Larger box height = object is probably closer.
WARNING_BOX_RATIO = 0.25
BRAKE_BOX_RATIO = 0.40

# Only consider objects roughly in our driving lane.
LANE_MIN_X = 0.30
LANE_MAX_X = 0.70


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")


# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(20.0)

world = client.get_world()
blueprints = world.get_blueprint_library()


# ============================================================
# SPAWN VEHICLE
# ============================================================

vehicle_bps = blueprints.filter("vehicle.*")

if not vehicle_bps:
    raise RuntimeError("No vehicle blueprints available in this CARLA world.")

vehicle_bp = np.random.choice(vehicle_bps)

print(f"Using vehicle: {vehicle_bp.id}")


spawn_points = world.get_map().get_spawn_points()

vehicle = world.spawn_actor(
    vehicle_bp,
    np.random.choice(spawn_points)
)

print("Vehicle spawned.")


# ============================================================
# IMPORTANT:
# We control throttle/brake ourselves.
# ============================================================

vehicle.set_autopilot(False)


# ============================================================
# CAMERA
# ============================================================

camera_bp = blueprints.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

print("Camera attached.")


# ============================================================
# ADAS DECISION
# ============================================================

def calculate_control(frame, result):

    height, width = frame.shape[:2]

    danger_level = 0

    closest_object = None
    largest_ratio = 0.0


    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name not in DANGER_CLASSES:
            continue


        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        box_width = x2 - x1
        box_height = y2 - y1


        # ----------------------------------------------------
        # Object horizontal position
        # ----------------------------------------------------

        center_x = (x1 + x2) / 2

        normalized_center_x = center_x / width


        # Ignore objects outside approximate ego lane
        if not (
            LANE_MIN_X
            <= normalized_center_x
            <= LANE_MAX_X
        ):
            continue


        # ----------------------------------------------------
        # Estimate closeness using bbox height
        # ----------------------------------------------------

        box_ratio = box_height / height


        if box_ratio > largest_ratio:

            largest_ratio = box_ratio

            closest_object = {
                "class": class_name,
                "confidence": confidence,
                "ratio": box_ratio,
                "box": (x1, y1, x2, y2)
            }


    # ========================================================
    # ADAS LOGIC
    # ========================================================

    if largest_ratio >= BRAKE_BOX_RATIO:

        # Emergency
        throttle = 0.0
        brake = 1.0

        danger_level = 2


    elif largest_ratio >= WARNING_BOX_RATIO:

        # Slow down
        throttle = 0.0
        brake = 0.40

        danger_level = 1


    else:

        # Safe
        throttle = CRUISE_THROTTLE
        brake = 0.0

        danger_level = 0


    return throttle, brake, danger_level, closest_object


# ============================================================
# CAMERA CALLBACK
# ============================================================

def process_image(image):

    # CARLA BGRA
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = array.reshape(
        (image.height, image.width, 4)
    )


    # Convert BGRA -> BGR
    frame = array[:, :, :3]


    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # ADAS
    # ========================================================

    throttle, brake, danger, obj = calculate_control(
        frame,
        result
    )


    # ========================================================
    # APPLY CONTROL TO CARLA
    # ========================================================

    control = carla.VehicleControl()

    control.throttle = throttle
    control.brake = brake

    # Straight driving for this test
    control.steer = 0.0

    vehicle.apply_control(control)


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    if obj:

        print(
            f"{obj['class']:12s} | "
            f"conf={obj['confidence']:.2f} | "
            f"bbox_ratio={obj['ratio']:.3f} | "
            f"Throttle={throttle:.2f} "
            f"Brake={brake:.2f}"
        )

    else:

        print(
            f"No obstacle | "
            f"Throttle={throttle:.2f} "
            f"Brake={brake:.2f}"
        )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    annotated = result.plot()


    if danger == 2:

        text = "EMERGENCY BRAKE"

    elif danger == 1:

        text = "WARNING / BRAKING"

    else:

        text = "SAFE"


    cv2.putText(
        annotated,
        text,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255) if danger else (0, 255, 0),
        3
    )


    cv2.putText(
        annotated,
        f"Throttle: {throttle:.2f}  Brake: {brake:.2f}",
        (40, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # cv2.imshow(
    #     "YOLO Closed-Loop ADAS",
    #     annotated
    # )


    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     camera.stop()


# ============================================================
# START
# ============================================================

camera.listen(process_image)

print()
print("====================================")
print("Closed-loop ADAS running")
print("====================================")
print("Press Q to stop.")


# ============================================================
# KEEP PROGRAM ALIVE
# ============================================================

try:

    while True:
        time.sleep(0.1)


except KeyboardInterrupt:

    print("\nStopping...")


finally:

    camera.stop()

    camera.destroy()
    vehicle.destroy()

    cv2.destroyAllWindows()

    print("Cleaned up.")
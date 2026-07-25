"""
Closed-Loop ADAS script for CARLA.
This script controls a vehicle using YOLO-based object detection.
"""

import argparse
import logging
import sys
import time
import queue

# ============================================================
# CONFIGURATION & LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ADAS_Controller")

# Graceful imports with missing dependency handling
try:
    import carla
except ImportError:
    logger.critical("CARLA module not found. Please ensure the CARLA PythonAPI is installed.")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError as e:
    logger.critical("Missing required data processing library: %s", e)
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    logger.critical("ultralytics module not found. Please install ultralytics for YOLO.")
    sys.exit(1)


class ADASController:
    """
    A class to encapsulate the ADAS logic and CARLA vehicle control loops.
    """

    def __init__(self,
                 host="localhost",
                 port=2000,
                 model_path="yolov8n.pt",
                 conf_threshold=0.40,
                 img_sz=1280,
                 cruise_throttle=0.35,
                 warning_box_ratio=0.25,
                 brake_box_ratio=0.40,
                 lane_min_x=0.30,
                 lane_max_x=0.70,
                 visualize=False):
        """
        Initializes the ADASController with configuration parameters.

        :param host: CARLA server host address.
        :type host: str
        :param port: CARLA server port.
        :type port: int
        :param model_path: Path to the YOLO weights file.
        :type model_path: str
        :param conf_threshold: Minimum confidence threshold for YOLO detections.
        :type conf_threshold: float
        :param img_sz: Image size for YOLO inference.
        :type img_sz: int
        :param cruise_throttle: Default throttle value for safe driving.
        :type cruise_throttle: float
        :param warning_box_ratio: Bounding box height ratio to trigger a warning/slow down.
        :type warning_box_ratio: float
        :param brake_box_ratio: Bounding box height ratio to trigger emergency braking.
        :type brake_box_ratio: float
        :param lane_min_x: Minimum normalized X coordinate to consider an object in the driving lane.
        :type lane_min_x: float
        :param lane_max_x: Maximum normalized X coordinate to consider an object in the driving lane.
        :type lane_max_x: float
        :param visualize: Whether to display the camera feed and YOLO bounding boxes.
        :type visualize: bool
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_sz = img_sz
        self.cruise_throttle = cruise_throttle
        self.warning_box_ratio = warning_box_ratio
        self.brake_box_ratio = brake_box_ratio
        self.lane_min_x = lane_min_x
        self.lane_max_x = lane_max_x
        self.visualize = visualize

        # Classes that can trigger braking
        self.danger_classes = {"Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians"}

        # CARLA actors and ML components
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.model = None

    def load_model(self):
        """
        Loads the YOLO model for object detection.

        :raises RuntimeError: If the model fails to load or the file is missing.
        """
        logger.info("Loading YOLO model from %s...", self.model_path)
        try:
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise RuntimeError(f"Model loading failed: {e}")

    def connect_carla(self):
        """
        Connects to the CARLA simulator and initializes the world object.

        :raises RuntimeError: If the connection to the CARLA server times out or fails.
        """
        logger.info("Connecting to CARLA server at %s:%s...", self.host, self.port)
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()
            logger.info("Connected to CARLA successfully.")
        except Exception as e:
            logger.error("Failed to connect to CARLA: %s", e)
            raise RuntimeError(f"CARLA connection failed: {e}")

    def spawn_actors(self):
        """
        Spawns the ego vehicle and attaches the RGB camera to it.

        :raises RuntimeError: If no vehicle blueprints or spawn points are available on the map.
        """
        blueprints = self.world.get_blueprint_library()
        vehicle_bps = blueprints.filter("vehicle.*")

        if not vehicle_bps:
            logger.error("No vehicle blueprints available in this CARLA world.")
            raise RuntimeError("No vehicle blueprints available.")

        # vehicle_bp = np.random.choice(vehicle_bps)
        vehicle_bp = blueprints.find("vehicle.ue4.chevrolet.impala")
        logger.info("Selected vehicle blueprint: %s", vehicle_bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("No spawn points found on the current map.")
            raise RuntimeError("No spawn points available.")

        # Spawn vehicle
        spawn_point = np.random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info("Vehicle spawned successfully at %s.", spawn_point.location)

        self.vehicle.set_autopilot(False)

        # Setup camera
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        logger.info("RGB Camera attached to the vehicle.")

    def calculate_control(self, frame, result):
        """
        Calculates the vehicle control (throttle and brake) based on YOLO detections.

        :param frame: The current RGB image frame from the camera.
        :type frame: numpy.ndarray
        :param result: The YOLO model inference result containing bounding boxes.
        :type result: ultralytics.engine.results.Results
        :return: A tuple containing throttle, brake, danger level, and closest object dictionary.
        :rtype: tuple(float, float, int, dict or None)
        """
        height, width = frame.shape[:2]
        danger_level = 0
        closest_object = None
        largest_ratio = 0.0

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            if class_name not in self.danger_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_height = y2 - y1

            center_x = (x1 + x2) / 2
            normalized_center_x = center_x / width

            # Ignore objects outside approximate ego lane
            if not (self.lane_min_x <= normalized_center_x <= self.lane_max_x):
                continue

            box_ratio = box_height / height

            if box_ratio > largest_ratio:
                largest_ratio = box_ratio
                closest_object = {
                    "class": class_name,
                    "confidence": confidence,
                    "ratio": box_ratio,
                    "box": (x1, y1, x2, y2)
                }

        if largest_ratio >= self.brake_box_ratio:
            throttle, brake, danger_level = 0.0, 1.0, 2
        elif largest_ratio >= self.warning_box_ratio:
            throttle, brake, danger_level = 0.0, 0.40, 1
        else:
            throttle, brake, danger_level = self.cruise_throttle, 0.0, 0

        return throttle, brake, danger_level, closest_object

    def calculate_steering(self):
        """
        Calculates the steering angle to keep the vehicle in the center of its lane.

        :return: Steering command in [-1.0, 1.0]
        :rtype: float
        """
        if not self.vehicle or not self.world:
            return 0.0
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Look ahead by 5 meters to target the upcoming path
        next_wps = waypoint.next(5.0)
        if not next_wps:
            return 0.0
            
        target_loc = next_wps[0].transform.location
        v_vec = vehicle_transform.get_forward_vector()
        
        # Vector to the target point, ignoring the Z-axis for a flat 2D steering model
        target_vec = carla.Vector3D(target_loc.x - vehicle_location.x, target_loc.y - vehicle_location.y, 0.0)
        target_vec_mag = np.sqrt(target_vec.x**2 + target_vec.y**2)
        if target_vec_mag == 0:
            return 0.0
            
        target_vec.x /= target_vec_mag
        target_vec.y /= target_vec_mag
        
        # Cross product Z-component yields the signed steering direction (- left, + right)
        cross_z = v_vec.x * target_vec.y - v_vec.y * target_vec.x
        
        # Return proportional control value capped between [-1.0, 1.0]
        return float(np.clip(cross_z * 2.0, -1.0, 1.0))

    def process_image(self, image):
        """
        Callback function to process incoming camera frames and apply control.

        :param image: The raw image data from the CARLA camera sensor.
        :type image: carla.Image
        """
        try:
            # Convert CARLA BGRA to BGR
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3]

            # YOLO Inference
            results = self.model.predict(
                frame,
                imgsz=self.img_sz,
                conf=self.conf_threshold,
                verbose=False
            )
            result = results[0]

            throttle, brake, danger, obj = self.calculate_control(frame, result)

            # Apply Control
            steer = 0.0
            if self.vehicle and self.vehicle.is_alive:
                steer = self.calculate_steering()
                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                self.vehicle.apply_control(control)

            # Log results (Using DEBUG level to avoid terminal spam; change log level if desired)
            if obj:
                logger.debug(
                    "%s | conf=%.2f | bbox_ratio=%.3f | Throttle=%.2f Brake=%.2f Steer=%.2f",
                    obj['class'], obj['confidence'], obj['ratio'], throttle, brake, steer
                )
            else:
                logger.debug("No obstacle | Throttle=%.2f Brake=%.2f Steer=%.2f", throttle, brake, steer)

            self._visualize(result, danger, throttle, brake, steer)

        except Exception as e:
            logger.error("Error occurred during image processing: %s", e, exc_info=True)

    def _visualize(self, result, danger, throttle, brake, steer=0.0):
        """
        Annotates the frame with detection boxes and vehicle telemetry logic.
        
        :param result: The YOLO model inference result.
        :type result: ultralytics.engine.results.Results
        :param danger: Computed danger level.
        :type danger: int
        :param throttle: Current throttle value.
        :type throttle: float
        :param brake: Current brake value.
        :type brake: float
        :param steer: Current steer value.
        :type steer: float
        """
        annotated = result.plot()

        if danger == 2:
            text = "EMERGENCY BRAKE"
            color = (0, 0, 255)
        elif danger == 1:
            text = "WARNING / BRAKING"
            color = (0, 165, 255)  # Orange
        else:
            text = "SAFE"
            color = (0, 255, 0)

        cv2.putText(annotated, text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(
            annotated,
            f"Throttle: {throttle:.2f}  Brake: {brake:.2f}  Steer: {steer:.2f}",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        logger.info(f"{text} | Throttle: {throttle:.2f}  Brake: {brake:.2f}  Steer: {steer:.2f}")
        if self.visualize:
            cv2.imshow("YOLO Closed-Loop ADAS", annotated)
            cv2.waitKey(1)

    def run(self):
        """
        Sets up the environment and starts the ADAS control loop.
        """
        try:
            self.load_model()
            self.connect_carla()
            self.spawn_actors()

            logger.info("Enabling synchronous mode...")
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)

            image_queue = queue.Queue()
            self.camera.listen(image_queue.put)

            logger.info("====================================")
            logger.info("Closed-loop ADAS running in SYNC mode. Press Ctrl+C to stop.")
            logger.info("====================================")

            spectator = self.world.get_spectator()

            while True:
                self.world.tick()
                image = image_queue.get()
                self.process_image(image)
                spectator.set_transform(self.camera.get_transform())

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Stopping...")
        except Exception as e:
            logger.error("An unexpected error occurred in run loop: %s", e, exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Stops sensors and destroys all spawned CARLA actors to clean up the simulation.
        """
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        logger.info("Cleaning up CARLA actors...")
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
            logger.info("Camera destroyed.")

        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            logger.info("Vehicle destroyed.")

        cv2.destroyAllWindows()
        logger.info("Cleanup finished.")


def main():
    """
    Main entry point for the closed loop script.
    """
    parser = argparse.ArgumentParser(description="Closed-Loop ADAS script for CARLA.")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host address (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument("--model-path", type=str, default="yolov8n.pt", help="Path to the YOLO weights file (default: yolov8n.pt)")
    parser.add_argument("--conf-threshold", type=float, default=0.40, help="Minimum confidence threshold for YOLO detections (default: 0.40)")
    parser.add_argument("--img-sz", type=int, default=1280, help="Image size for YOLO inference (default: 1280)")
    parser.add_argument("--cruise-throttle", type=float, default=0.35, help="Default throttle value for safe driving (default: 0.35)")
    parser.add_argument("--warning-box-ratio", type=float, default=0.25, help="Bounding box height ratio to trigger a warning/slow down (default: 0.25)")
    parser.add_argument("--brake-box-ratio", type=float, default=0.40, help="Bounding box height ratio to trigger emergency braking (default: 0.40)")
    parser.add_argument("--lane-min-x", type=float, default=0.30, help="Minimum normalized X coordinate to consider an object in the driving lane (default: 0.30)")
    parser.add_argument("--lane-max-x", type=float, default=0.70, help="Maximum normalized X coordinate to consider an object in the driving lane (default: 0.70)")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of the YOLO detections window")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level (default: INFO)")

    args = parser.parse_args()

    # Update logging level based on arguments
    logger.setLevel(getattr(logging, args.log_level.upper()))

    adas_controller = ADASController(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        img_sz=args.img_sz,
        cruise_throttle=args.cruise_throttle,
        warning_box_ratio=args.warning_box_ratio,
        brake_box_ratio=args.brake_box_ratio,
        lane_min_x=args.lane_min_x,
        lane_max_x=args.lane_max_x,
        visualize=args.visualize
    )
    adas_controller.run()


if __name__ == "__main__":
    main()
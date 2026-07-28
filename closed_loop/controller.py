import logging
import math
import queue
import cv2
import numpy as np
import sys

try:
    import carla
except ImportError:
    logging.critical("CARLA module not found. Please ensure the CARLA PythonAPI is installed.")
    sys.exit(1)

logger = logging.getLogger("ADAS_Controller")

class ADASController:
    """
    A class to encapsulate the ADAS logic and CARLA vehicle control loops.
    """
    def __init__(self, perception_module, host="localhost", port=2000,
                 cruise_throttle=0.35, warning_distance=15.0, brake_distance=7.0,
                 lane_width=3.5, max_speed=30.0, visualize=False):
        self.perception = perception_module
        self.host = host
        self.port = port
        self.cruise_throttle = cruise_throttle
        self.warning_distance = warning_distance
        self.brake_distance = brake_distance
        self.lane_width = lane_width
        self.max_speed = max_speed
        self.visualize = visualize

        self.danger_classes = {"Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians"}
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None

    def connect_carla(self):
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
        blueprints = self.world.get_blueprint_library()
        vehicle_bps = blueprints.filter("vehicle.*")

        if not vehicle_bps:
            logger.error("No vehicle blueprints available in this CARLA world.")
            raise RuntimeError("No vehicle blueprints available.")

        vehicle_bp = np.random.choice(vehicle_bps)
        logger.info("Selected vehicle blueprint: %s", vehicle_bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("No spawn points found on the current map.")
            raise RuntimeError("No spawn points available.")

        spawn_point = np.random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info("Vehicle spawned successfully at %s.", spawn_point.location)

        self.vehicle.set_autopilot(False)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(carla.Location(x=self.vehicle.bounding_box.extent.x + 0.1, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        logger.info("RGB Camera attached to the vehicle.")

    def calculate_control(self, frame, boxes, scores, labels, distances):
        height, width = frame.shape[:2]
        danger_level = 0
        closest_object = None
        min_distance = float('inf')
        detected_objects = []

        focal_length = width / 2.0
        c_x = width / 2.0

        for box, score, label, dist in zip(boxes, scores, labels, distances):
            if dist is None:
                continue
                
            class_name = self.perception.get_class_name(label)
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0

            lateral_distance = dist * (center_x - c_x) / focal_length

            detected_objects.append({
                "class": class_name,
                "confidence": score,
                "distance": dist,
                "lateral_distance": lateral_distance,
                "box": box
            })

            if class_name not in self.danger_classes:
                continue
            if abs(lateral_distance) > (self.lane_width / 2.0):
                continue

            if dist < min_distance:
                min_distance = dist
                closest_object = detected_objects[-1]

        if min_distance <= self.brake_distance:
            throttle, brake, danger_level = 0.0, 1.0, 2
        elif min_distance <= self.warning_distance:
            throttle, brake, danger_level = 0.0, 0.40, 1
        else:
            if self.vehicle:
                v = self.vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                if speed_kmh >= self.max_speed:
                    throttle, brake, danger_level = 0.0, 0.0, 0
                else:
                    throttle, brake, danger_level = self.cruise_throttle, 0.0, 0
            else:
                throttle, brake, danger_level = self.cruise_throttle, 0.0, 0

        return throttle, brake, danger_level, closest_object, detected_objects

    def calculate_steering(self):
        if not self.vehicle or not self.world:
            return 0.0
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        next_wps = waypoint.next(5.0)
        if not next_wps:
            return 0.0
            
        target_loc = next_wps[0].transform.location
        v_vec = vehicle_transform.get_forward_vector()
        
        target_vec = carla.Vector3D(target_loc.x - vehicle_location.x, target_loc.y - vehicle_location.y, 0.0)
        target_vec_mag = np.sqrt(target_vec.x**2 + target_vec.y**2)
        if target_vec_mag == 0:
            return 0.0
            
        target_vec.x /= target_vec_mag
        target_vec.y /= target_vec_mag
        
        cross_z = v_vec.x * target_vec.y - v_vec.y * target_vec.x
        return float(np.clip(cross_z * 2.0, -1.0, 1.0))

    def process_image(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3]

            boxes, scores, labels, distances = self.perception.predict(frame)
            throttle, brake, danger, obj, detected_objects = self.calculate_control(
                frame, boxes, scores, labels, distances
            )

            steer = 0.0
            if self.vehicle and self.vehicle.is_alive:
                steer = self.calculate_steering()
                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                self.vehicle.apply_control(control)

            if obj:
                logger.debug(f"{obj['class']} | conf={obj['confidence']:.2f} | dist={obj['distance']:.2f}m | Th={throttle:.2f} Br={brake:.2f} St={steer:.2f}")
            else:
                logger.debug(f"No obstacle | Th={throttle:.2f} Br={brake:.2f} St={steer:.2f}")

        except Exception as e:
            logger.error("Error occurred during image processing: %s", e, exc_info=True)

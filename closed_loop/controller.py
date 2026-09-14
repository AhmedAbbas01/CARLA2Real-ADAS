import logging
import math
import queue
import cv2
import numpy as np
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_evaluation")))
from evaluate_distance import DistanceEvaluator

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
                 lane_width=3.5, max_speed=30.0, visualize=False, running_mode="online"):
        self.perception = perception_module
        self.evaluator = DistanceEvaluator(center_dist_threshold=50.0) if running_mode == "evaluate_distance_model" else None
        self.running_mode = running_mode
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
        exclude_ids = ["vehicle.firetruck.actors", "vehicle.sprinter.mercedes", "vehicle.ambulance.ford", "vehicle.carlacola.actors"]
        vehicle_bps = [bp for bp in blueprints.filter('vehicle.*') if bp.id not in exclude_ids]

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
            center_y = (y1 + y2) / 2.0

            lateral_distance = dist * (center_x - c_x) / focal_length

            detected_objects.append({
                "class": class_name,
                "confidence": score,
                "distance": dist,
                "lateral_distance": lateral_distance,
                "box": box,
                "center": [center_x, center_y]
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

    def process_image(self, image, frame_count):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3].copy()

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

            if danger == 2:
                text = "EMERGENCY BRAKE"
                color = (0, 0, 255)
            elif danger == 1:
                text = "WARNING / BRAKING"
                color = (0, 165, 255)  # Orange
            else:
                text = "SAFE"
                color = (0, 255, 0)

            logger.info(f"{text} | Throttle: {throttle:.2f}  Brake: {brake:.2f}  Steer: {steer:.2f}")
            
            if self.running_mode == "evaluate_distance_model":
                gt_objects, metrics = self.evaluate_distance(detected_objects, frame_count)

            if self.visualize and self.running_mode == "online":
                self.show_visualization(frame, detected_objects, text, color, throttle, brake, steer)
            
            elif self.visualize and self.running_mode == "evaluate_distance_model":
                self.show_evaluation_visualization(frame, gt_objects, detected_objects, metrics, text, color, throttle, brake, steer)

        except Exception as e:
            logger.error("Error occurred during image processing: %s", e, exc_info=True)

    def show_visualization(self, frame, detected_objects, text, color, throttle, brake, steer=0.0):
        """
        Annotates the frame with detection boxes and vehicle telemetry logic.
        
        :param frame: The current RGB image frame.
        :type frame: numpy.ndarray
        :param detected_objects: List of detected objects with class and distance.
        :type detected_objects: list
        :param color: color to show the danger level.
        :type color: RGB tuple
        :param text: text representing the danger level.
        :type text: str
        :param throttle: Current throttle value.
        :type throttle: float
        :param brake: Current brake value.
        :type brake: float
        :param steer: Current steer value.
        :type steer: float
        """
        annotated = frame.copy()

        for obj in detected_objects:
            if obj['class'] not in self.danger_classes:
                continue

            x1, y1, x2, y2 = obj["box"]
            label = f"{obj['class']} {obj['distance']:.1f}m (Lat: {obj['lateral_distance']:.1f}m)"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        self._draw_driving_decision(annotated, text, color, throttle, brake, steer)

        cv2.imshow("YOLO Closed-Loop ADAS", annotated)
        cv2.waitKey(1)

    def _draw_driving_decision(self, annotated, text, color, throttle, brake, steer):
        cv2.putText(annotated, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(
            annotated,
            f"Throttle: {throttle:.2f}  Brake: {brake:.2f}  Steer: {steer:.2f}",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    def get_third_person_camera_transform(self):
        import math
        # 1. Get the current position of the vehicle
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_rot = vehicle_transform.rotation

        # 2. Calculate third-person offset (yaw matches the car)
        # Convert yaw to radians to calculate X and Y vector offsets
        yaw_rad = math.radians(vehicle_rot.yaw)
        
        # Position the spectator 8 meters behind and 3.5 meters above the car
        spectator_x = vehicle_loc.x - 8.0 * math.cos(yaw_rad)
        spectator_y = vehicle_loc.y - 8.0 * math.sin(yaw_rad)
        spectator_z = vehicle_loc.z + 3.5

        # 3. Angle the spectator slightly downward (-15 degrees pitch)
        spectator_transform = carla.Transform(
            carla.Location(x=spectator_x, y=spectator_y, z=spectator_z),
            carla.Rotation(pitch=-15.0, yaw=vehicle_rot.yaw, roll=0.0)
        )
        return spectator_transform

    def get_gt_objects(self, K, image_w, image_h):
        gt_objects = []
        seen_objects = set()
        camera_transform = self.camera.get_transform()
        w2c = np.array(camera_transform.get_inverse_matrix())
        ego_location = camera_transform.location
        ego_forward = camera_transform.get_forward_vector()

        def project_3d_to_2d(vertices):
            verts_2d = []
            for v in vertices:
                p_world = np.array([v.x, v.y, v.z, 1.0])
                p_camera = np.dot(w2c, p_world)
                if p_camera[0] <= 0.0:
                    continue
                p_cam_std = np.array([p_camera[1], -p_camera[2], p_camera[0]])
                p_img = np.dot(K, p_cam_std)
                if p_img[2] > 0:
                    cx = p_img[0] / p_img[2]
                    cy = p_img[1] / p_img[2]
                    verts_2d.append([cx, cy])
            return verts_2d

        # 1. Dynamic Objects (Actors)
        for actor in self.world.get_actors().filter('*'):
            if self.vehicle and actor.id == self.vehicle.id:
                continue

            type_id = actor.type_id
            if type_id.startswith('vehicle'):
                if type_id.startswith('vehicle.truck'):
                    class_name = 'Truck'
                elif type_id.startswith('vehicle.bus') or type_id.startswith('vehicle.volkswagen'):
                    class_name = 'Bus'
                elif type_id.startswith('vehicle.yamaha') or type_id.startswith('vehicle.kawasaki') or type_id.startswith('vehicle.harley-davidson'):
                    class_name = 'Motorcycle'
                elif type_id.startswith('vehicle.bh') or type_id.startswith('vehicle.diamondback') or type_id.startswith('vehicle.gazelle'):
                    class_name = 'Bicycle'
                else:
                    class_name = 'Car'
            elif type_id.startswith('walker.pedestrian'):
                class_name = 'Pedestrians'
            else:
                continue

            dist = actor.get_transform().location.distance(ego_location)
            if dist > 50.0 or dist < 0.1:
                continue

            ray = actor.get_transform().location - ego_location
            if ego_forward.dot(ray) <= 0:
                continue

            if hasattr(actor, 'bounding_box'):
                verts = actor.bounding_box.get_world_vertices(actor.get_transform())
                verts_2d = project_3d_to_2d(verts)
                if not verts_2d:
                    continue
                verts_2d = np.array(verts_2d)
                x_min, x_max = np.min(verts_2d[:, 0]), np.max(verts_2d[:, 0])
                y_min, y_max = np.min(verts_2d[:, 1]), np.max(verts_2d[:, 1])

                if x_max < 0 or x_min > image_w or y_max < 0 or y_min > image_h:
                    continue

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                
                obj_key = (dist, class_name)
                if obj_key not in seen_objects:
                    seen_objects.add(obj_key)
                    gt_objects.append({'center': [float(cx), float(cy)], 'class': class_name, 'distance': float(dist)})

        # 2. Static Objects (Environment Objects)
        # Map target classes to CARLA CityObjectLabel enums
        TARGET_CLASSES = {
            "Car": carla.CityObjectLabel.Car,
            "Truck": carla.CityObjectLabel.Truck,
            "Bus": carla.CityObjectLabel.Bus,
            "Motorcycle": carla.CityObjectLabel.Motorcycle,
            "Bicycle": carla.CityObjectLabel.Bicycle,
            "Pedestrians": carla.CityObjectLabel.Pedestrians,
        }

        identity_transform = carla.Transform()  # Identity transform for world-space boxes

        # 1. Fetch filtered static environment objects
        for class_name, label in TARGET_CLASSES.items():
            env_objects = self.world.get_environment_objects(label)
            
            for obj in env_objects:
                # Distance calculation relative to ego vehicle
                dist = obj.transform.location.distance(ego_location)
                if dist > 50.0 or dist < 0.1:
                    continue

                ray = obj.transform.location - ego_location
                if ego_forward.dot(ray) <= 0:
                    continue
                
                # 2. Extract 3D vertices (EnvironmentObject bounding boxes are in world coordinates)
                verts = obj.bounding_box.get_world_vertices(identity_transform)
                
                # 3. Project to 2D image plane
                verts_2d = project_3d_to_2d(verts)
                if not verts_2d:
                    continue
                    
                verts_2d = np.array(verts_2d)
                x_min, x_max = np.min(verts_2d[:, 0]), np.max(verts_2d[:, 0])
                y_min, y_max = np.min(verts_2d[:, 1]), np.max(verts_2d[:, 1])

                # 4. Filter out-of-bounds bounding boxes
                if x_max < 0 or x_min > image_w or y_max < 0 or y_min > image_h:
                    continue

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                
                obj_key = (dist, class_name)
                if obj_key not in seen_objects:
                    seen_objects.add(obj_key)
                    gt_objects.append({
                        'center': [float(cx), float(cy)],
                        'class': class_name,
                        'distance': float(dist)
                    })

        return gt_objects

    def evaluate_distance(self, detected_objects, frame_count):
        try:
            
            image_w = float(self.camera.attributes["image_size_x"])
            image_h = float(self.camera.attributes["image_size_y"])
            fov = float(self.camera.attributes["fov"])

            focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
            K = np.identity(3)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = image_w / 2.0
            K[1, 2] = image_h / 2.0
            
            gt_objects = self.get_gt_objects(K, image_w, image_h)
            metrics = self.evaluator.add_frame_predictions(gt_objects, detected_objects, frame_id=frame_count)
            
            if metrics:
                logger.info(f"Frame {frame_count:04d} | Matches: {metrics['frame_matches']}/{len(detected_objects)} | "
                            f"MAE: {metrics['frame_mae']:.2f}m | Total Rel Err: {metrics['running_rel_error'] * 100:.2f}%")
            else:
                logger.info(f"Frame {frame_count:04d} | No matches found for evaluation.")

            return gt_objects, metrics
        except Exception as e:
            logger.error("Error occurred during distance evaluation: %s", e, exc_info=True)
            return [], None

    def show_evaluation_visualization(self, frame, gt_objects, detected_objects, metrics, text_decision, color, throttle, brake, steer):
        """
        Annotates the frame with ground-truth and detected objects for evaluation comparison.
        
        :param frame: The current RGB image frame.
        :param gt_objects: List of ground-truth objects.
        :param detected_objects: List of detected predicted objects.
        :param metrics: Evaluation metrics dictionary for the current frame.
        :param text_decision: text representing the danger level.
        :param color: color to show the danger level.
        :param throttle: Current throttle value.
        :param brake: Current brake value.
        :param steer: Current steer value.
        """
        annotated = frame.copy()
        
        # Draw Ground Truth objects (Green)
        for gt in gt_objects:
            if gt['class'] not in self.danger_classes:
                continue

            cx, cy = map(int, gt['center'])
            label = f"GT {gt['class']}: {gt['distance']:.1f}m"
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(annotated, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Detected objects (Yellow)
        for obj in detected_objects:
            if obj['class'] not in self.danger_classes:
                continue

            cx, cy = map(int, obj['center'])
            label = f"Pred {obj['class']}: {obj['distance']:.1f}m"
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(annotated, label, (cx + 10, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display Metrics
        if metrics:
            text = f"Matches: {metrics['frame_matches']} | Frame MAE: {metrics['frame_mae']:.2f}m"
        else:
            text = "Matches: 0 | Frame MAE: N/A"
            
        cv2.putText(annotated, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        self._draw_driving_decision(annotated, text_decision, color, throttle, brake, steer)

        cv2.imshow("Distance Evaluation", annotated)
        cv2.waitKey(1)
 
    def evaluate(self):
        """
        Finalizes evaluation and generates the evaluation report.
        """
        if self.evaluator:
            self.evaluator.evaluate()

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

        if self.client and self.world:
            actors = self.world.get_actors()
            destroy_batch = [
                carla.command.DestroyActor(x) for x in actors 
                if x.type_id.startswith(('vehicle.', 'sensor.', 'walker.', 'controller.'))
            ]
            if destroy_batch:
                self.client.apply_batch(destroy_batch)
                logger.info("Destroyed %d actors.", len(destroy_batch))

        cv2.destroyAllWindows()
        logger.info("Cleanup finished.")
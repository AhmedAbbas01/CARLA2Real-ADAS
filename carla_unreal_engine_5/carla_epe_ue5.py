import carla
import time
import os
import random
import argparse
import sys
import numpy as np
import math
import json


VERBOSE = False

def log(message, verbose_only=False):
    """Print a message, optionally only when verbose logging is enabled."""
    if verbose_only and not VERBOSE:
        return
    print(message)

try:
    import pyautogui
except Exception:
    pyautogui = None
    log("WARNING: pyautogui is unavailable; some features may be disabled.", verbose_only=False)


def parse_args():
    """Parses command line arguments.
    
    Returns:
        Parsed arguments object containing width, height, output_dir, num_frames_export, and export_step.
    """
    parser = argparse.ArgumentParser(description="Capture frames using pyautogui with specified parameters.")
    parser.add_argument("--width", type=int, default=960, help="Width of the exported frames.")
    parser.add_argument("--height", type=int, default=540, help="Height of the exported frames.")
    parser.add_argument("--output_dir", type=str, default="../Dataset/", help="Directory to save the frames.")
    parser.add_argument("--num_frames_export", type=int, default=5, help="Number of frames to export.")
    parser.add_argument("--export_step", type=int, default=60, help="Step size between frame exports.")
    parser.add_argument("--num_vehicles", type=int, default=70, help="Number of vehicles to spawn.")
    parser.add_argument("--num_walkers", type=int, default=40, help="Number of pedestrians to spawn.")
    parser.add_argument(
        "--bbox_distance_range",
        nargs=2,
        type=float,
        default=[0.1, 50.0],
        metavar=("MIN", "MAX"),
        help="Minimum and maximum allowed distance in meters for bounding boxes.",
    )
    parser.add_argument("--map_name", type=str, default="Town10HD_Opt", help="Name of the CARLA map to load.")
    parser.add_argument("--weather_preset", type=str, default="ClearNoon", help="Name of the CARLA weather preset.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for extra runtime details.")
    return parser.parse_args()

def remove_file_if_exists(file_path):
    """Removes a file at the specified path if it exists.
    
    Args:
        file_path (str): The path to the file to be removed.
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        log(f"{file_path} has been removed.", verbose_only=False)
    else:
        log(f"{file_path} does not exist.", verbose_only=True)

def allow_saving_Gbuffers():
    """Allows saving G-buffers in the Unreal Engine console when pyautogui is available."""
    if pyautogui is None:
        log("pyautogui is unavailable; skipping Unreal console command.", verbose_only=True)
        return

    pyautogui.press('`')
    pyautogui.write('r.BufferVisualizationDumpFrames 1')
    pyautogui.press('enter')
    log("G-buffer saving enabled in Unreal Engine console.", verbose_only=False)

def spawn_traffic(client, world, num_vehicles, num_walkers):
    # 1. Setup Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if num_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif num_vehicles > number_of_spawn_points:
        log(f"Requested {num_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points", verbose_only=True)
        num_vehicles = number_of_spawn_points

    # --------------
    # Spawn Vehicles
    # --------------
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(carla.command.SpawnActor(blueprint, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

    vehicles_id = []
    for response in client.apply_batch_sync(batch, True):
        if not response.error:
            vehicles_id.append(response.actor_id)
            
    # Set lane change attributes post-spawn
    all_vehicles = world.get_actors(vehicles_id)
    for vehicle in all_vehicles:
        if vehicle is not None:
            traffic_manager.auto_lane_change(vehicle, True)

    # -------------
    # Spawn Walkers
    # -------------
    walker_spawn_points = []
    for i in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point = carla.Transform()
            spawn_point.location = loc
            # CRITICAL FIX FOR UE5-DEV: Add vertical offset to prevent bounding box collision with ground
            spawn_point.location.z += 0.6 
            walker_spawn_points.append(spawn_point)
            
    batch = []
    walker_speeds = []
    for spawn_point in walker_spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        
        # Safe handling of walker speeds
        if walker_bp.has_attribute('speed'):
            speeds = walker_bp.get_attribute('speed').recommended_values
            # index 1 usually represents running/fast walking, 0 is walking. Let's sample safely:
            walker_speeds.append(speeds[1] if len(speeds) > 1 else speeds[0])
        else:
            walker_speeds.append(0.0)
            
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
    results = client.apply_batch_sync(batch, True)
    
    # Track successfully spawned walkers and their designated speeds cleanly
    walkers_list = []
    final_walker_speeds = []
    for i in range(len(results)):
        if not results[i].error:
            walkers_list.append({"id": results[i].actor_id})
            final_walker_speeds.append(walker_speeds[i])

    # -------------------------
    # Spawn Walker Controllers
    # -------------------------
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        
    results = client.apply_batch_sync(batch, True)
    
    # Match controllers to walkers
    for i in range(len(results)):
        if not results[i].error:
            walkers_list[i]["con"] = results[i].actor_id

    # Flatten IDs for a single world fetch
    all_id = []
    for i in range(len(walkers_list)):
        if "con" in walkers_list[i]: # Ensure controller spawned successfully
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
            
    all_actors = world.get_actors(all_id)

    # Sync with world before driving AI
    world.tick()

    # Initialize Walker AI Movements
    controller_index = 0
    for i in range(0, len(all_id), 2):
        controller = all_actors[i]
        if controller is not None:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(final_walker_speeds[controller_index]))
        controller_index += 1

    log(f"Spawned {len(vehicles_id)} vehicles and {len(all_id)//2} walkers.", verbose_only=False)
    return vehicles_id, all_id

def randomize_traffic(client, world, args, vehicle):
    """Destroy existing traffic actors and respawn a new randomized set.

    Returns:
        tuple: (vehicles_id, all_id) new lists after respawn
    """
    log("Randomizing traffic to resolve potential jam...", verbose_only=False)
    destroy_commands = []

    all_actors = world.get_actors()

    # Filter out the ego vehicle
    non_ego_actors = [actor for actor in all_actors if actor.id != vehicle.id]
        
    destroy_commands.extend([carla.command.DestroyActor(x) for x in non_ego_actors if x.type_id.startswith('vehicle.') or x.type_id.startswith('walker.') or x.type_id.startswith('controller.')])
    
    if destroy_commands:
        try:
            client.apply_batch_sync(destroy_commands, True)
        except Exception as e:
            log(f"Warning: error destroying actors during randomize: {e}", verbose_only=True)

    # small pause to let the world stabilize
    time.sleep(2)

    # respawn traffic
    try:
        new_vehicles, new_all = spawn_traffic(client, world, args.num_vehicles, args.num_walkers)
        return new_vehicles, new_all
    except Exception as e:
        log(f"Failed to respawn traffic: {e}", verbose_only=False)
        return [], []

def setup_carla(args):
    """Connects to CARLA and configures the world settings only.

    Returns:
        tuple: (client, world, original_settings)
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)

        world = client.get_world()
        original_settings = world.get_settings()
        
        world = adjust_map_and_weather(client, world, args)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        return client, world, original_settings
    except Exception as e:
        log(f"Error setting up CARLA: {e}", verbose_only=False)
        sys.exit(1)


def spawn_actors(client, world, width, height, num_vehicles, num_walkers):
    """Spawns ego vehicle, camera, and traffic (vehicles and walkers).

    Returns:
        tuple: vehicle, camera, vehicles_id, all_id
    """
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

    if spawn_points:
        random_spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, random_spawn_point)
        log(f"Vehicle spawned at location: {random_spawn_point.location}", verbose_only=True)
    else:
        log("No spawn points available in the map.", verbose_only=False)
        sys.exit(1)

    vehicle.set_autopilot(True)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.auto_lane_change(vehicle, True)
    traffic_manager.ignore_lights_percentage(vehicle, 100)
    traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)

    vehicles_id, all_id = spawn_traffic(client, world, num_vehicles, num_walkers)

    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=vehicle.bounding_box.extent.x + 0.1, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    if camera is None:
        log("ERROR: Failed to spawn semantic segmentation camera.", verbose_only=False)
    else:
        log(f"Semantic segmentation camera spawned with id {camera.id}.", verbose_only=True)

    return vehicle, camera, vehicles_id, all_id


def convert_image_to_array(image):
    """Converts a CARLA image to a NumPy array using the same convention as the experiment code."""
    if image is None:
        return None

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    return array


def is_semantic_bbox_visible(bbox, semantic_image):
    """Keeps a projected bbox only when it overlaps visible semantic evidence."""
    if semantic_image is None:
        return True

    x_min, y_min, x_max, y_max = bbox
    x_min = int(max(0, min(x_min, semantic_image.shape[1] - 1)))
    y_min = int(max(0, min(y_min, semantic_image.shape[0] - 1)))
    x_max = int(max(x_min + 1, min(x_max, semantic_image.shape[1])))
    y_max = int(max(y_min + 1, min(y_max, semantic_image.shape[0])))

    if x_max <= x_min or y_max <= y_min:
        return False

    patch = semantic_image[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return False

    if patch.ndim == 3:
        semantic_values = patch[:, :, :3].astype(np.uint32)
        semantic_values = semantic_values[:, :, 0] + 256 * semantic_values[:, :, 1] + 256 ** 2 * semantic_values[:, :, 2]
    else:
        semantic_values = patch.astype(np.uint32)

    visible_fraction = np.count_nonzero(semantic_values > 0) / float(semantic_values.size)
    return visible_fraction >= 0.01


def get_bounding_boxes(world, camera, K, semantic_image=None, bbox_distance_range=(0.1, 50.0)):
    bounding_boxes = []
    min_distance, max_distance = bbox_distance_range
    
    camera_transform = camera.get_transform()
    c2w = np.array(camera_transform.get_matrix())
    w2c = np.linalg.inv(c2w)
    
    image_w = K[0, 2] * 2
    image_h = K[1, 2] * 2

    def project_verts_to_bbox(verts):
        verts_camera = []
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1.0])
            p_camera = np.dot(w2c, p_world)
            verts_camera.append(p_camera)
            
        if any(p[0] <= 0 for p in verts_camera):
            return None
            
        verts_2d = []
        for p_camera in verts_camera:
            p_cam_std = np.array([p_camera[1], -p_camera[2], p_camera[0]])
            
            p_img = np.dot(K, p_cam_std)
            p_img[0] /= p_img[2]
            p_img[1] /= p_img[2]
            verts_2d.append(p_img[:2])
            
        verts_2d = np.array(verts_2d)
        
        x_min = np.min(verts_2d[:, 0])
        x_max = np.max(verts_2d[:, 0])
        y_min = np.min(verts_2d[:, 1])
        y_max = np.max(verts_2d[:, 1])
        
        x_min = max(0.0, min(x_min, image_w))
        x_max = max(0.0, min(x_max, image_w))
        y_min = max(0.0, min(y_min, image_h))
        y_max = max(0.0, min(y_max, image_h))

        if x_min >= x_max or y_min >= y_max:
            return None
            
        return [float(x_min), float(y_min), float(x_max), float(y_max)]

    # 2. Process Objects by type
    labels = [
        carla.CityObjectLabel.Car,
        carla.CityObjectLabel.Truck,
        carla.CityObjectLabel.Bus,
        carla.CityObjectLabel.Motorcycle,
        carla.CityObjectLabel.Bicycle,
        carla.CityObjectLabel.Rider,
        carla.CityObjectLabel.Train,
        carla.CityObjectLabel.RailTrack,
        carla.CityObjectLabel.TrafficSigns,
        carla.CityObjectLabel.TrafficLight,
    ]
    
    rendered_centers = set()
    
    for label_index, label in enumerate(labels):
        try:
            level_bbs = world.get_level_bbs(label)
        except Exception:
            level_bbs = []
            
        for bb in level_bbs:
            dist = bb.location.distance(camera_transform.location)
            if dist > max_distance or dist < min_distance:
                continue
                
            center = (round(bb.location.x, 1), round(bb.location.y, 1), round(bb.location.z, 1))
            if center in rendered_centers:
                continue
            rendered_centers.add(center)
            
            verts = bb.get_world_vertices(carla.Transform())
            bbox_2d = project_verts_to_bbox(verts)
            
            if bbox_2d and is_semantic_bbox_visible(bbox_2d, semantic_image):
                type_name = f"{label.name}" if hasattr(label, 'name') else f"{str(label)}"
                obj_id = 1000+label_index
                bounding_boxes.append({
                    "type": type_name,
                    "id": obj_id,
                    "distance": float(dist),
                    "bbox": bbox_2d
                })

    # Filter overlapping bounding boxes of the same type
    filtered_bounding_boxes = []
    
    def get_area(box_dict):
        b = box_dict["bbox"]
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    # Sort boxes by area descending to easily keep the larger one
    bounding_boxes.sort(key=get_area, reverse=True)

    for box in bounding_boxes:
        should_keep = True
        for kept_box in filtered_bounding_boxes:
            b1 = box["bbox"]
            b2 = kept_box["bbox"]
            
            x_min_inter = max(b1[0], b2[0])
            y_min_inter = max(b1[1], b2[1])
            x_max_inter = min(b1[2], b2[2])
            y_max_inter = min(b1[3], b2[3])

            inter_width = max(0.0, x_max_inter - x_min_inter)
            inter_height = max(0.0, y_max_inter - y_min_inter)
            inter_area = inter_width * inter_height

            area1 = get_area(box)
            area2 = get_area(kept_box)
            smaller_area = min(area1, area2)
            
            if smaller_area > 0:
                overlap_ratio = inter_area / smaller_area
                if overlap_ratio >= 0.85 and box["distance"] >= kept_box["distance"]:
                    should_keep = False
                    break
        if should_keep:
            filtered_bounding_boxes.append(box)

    return filtered_bounding_boxes
    
def run_simulation(world, vehicle, camera, args, frame_counter=0, client=None, vehicles_id=None, all_id=None):
    """Runs the CARLA simulation loop and handles data generation.
    
    Args:
        world: The CARLA world object.
        vehicle: The CARLA ego vehicle actor.
        camera: The CARLA camera sensor.
        args: The parsed command-line arguments.
        
    Returns:
        int: The final frame counter to assist with cleanup.
    """
    output_dir = args.output_dir
    width = args.width
    height = args.height
    num_frames_export = args.num_frames_export
    export_step = args.export_step

    semantic_dir, frames_dir, bbox_dir = create_directories(output_dir)

    fov = 90
    focal = width / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    spectator = world.get_spectator()
    data = {}

    def save_segmentation_image(image):
        if not data:
            data["semantic_segmentation"] = image 

    camera.listen(save_segmentation_image)

    frame_exported_counter = 0
    world_tick_counter = 0
    stall_count = 0
    # threshold in meters under which the ego is considered not moved between processed frames
    stall_threshold = 0.5
    last_processed_loc = None

    if vehicles_id is None:
        vehicles_id = []
    if all_id is None:
        all_id = []

    log("Starting simulation... Press Ctrl+C to stop.", verbose_only=False)
    while True:
        processing_start_time = time.time()
        vehicle.set_autopilot(True)
        spectator.set_transform(camera.get_transform())
        try:
            world.tick()
        except Exception as e:
            log(f"Error occurred while ticking the world: {e}", verbose_only=False)

        # process and possibly export a frame at configured steps
        start_wait = time.time()
        while time.time() - start_wait < 5.0:
            if "semantic_segmentation" in data:
                break
            time.sleep(0.001)

        if "semantic_segmentation" not in data:
            log("WARNING: Timed out waiting for semantic segmentation data; stopping capture.", verbose_only=False)
            log("INFO: Check that the semantic camera is attached correctly and that the sensor callback is firing.", verbose_only=False)
            break

        world_tick_counter += 1

        if world_tick_counter % export_step == 0:
            # capture ego location at processing time
            try:
                cur_loc = vehicle.get_transform().location if vehicle is not None else None
            except Exception:
                cur_loc = None

            frame_counter = process_frame(
                width,
                height,
                semantic_dir,
                frames_dir,
                bbox_dir,
                data,
                frame_counter,
                world,
                camera,
                K,
                args.bbox_distance_range,
            )
            # time.sleep(1)

            # After processing a frame, update stall detection based on processed-frame positions
            try:
                if cur_loc is not None:
                    if last_processed_loc is None:
                        # first processed frame, initialize reference
                        stall_count = 0
                    else:
                        dist_moved = math.sqrt((cur_loc.x - last_processed_loc.x) ** 2 + (cur_loc.y - last_processed_loc.y) ** 2 + (cur_loc.z - last_processed_loc.z) ** 2)
                        if dist_moved < stall_threshold:
                            stall_count += 1
                        else:
                            stall_count = 0
                    last_processed_loc = cur_loc
            except Exception:
                stall_count = 0

            if stall_count >= 3:
                log(f"Detected possible traffic jam (ego stalled for {stall_count} processed frames). Randomizing traffic...", verbose_only=False)
                if client is not None:
                    try:
                        vehicles_id, all_id = randomize_traffic(client, world, args, vehicle)
                    except Exception as e:
                        log(f"Error during traffic randomization: {e}", verbose_only=False)
                stall_count = 0

            frame_exported_counter += 1
            if frame_exported_counter == num_frames_export:
                log("Finished data generation...", verbose_only=True)
                break
            log(f"Processed frame {frame_counter} in time {time.time() - processing_start_time:.2f} seconds)", verbose_only=False)
    time.sleep(0.1)

    return frame_counter, vehicles_id, all_id

def create_directories(output_dir):
    semantic_dir = os.path.abspath(os.path.join(output_dir, "Semantic"))
    frames_dir = os.path.abspath(os.path.join(output_dir, "Frames"))
    bbox_dir = os.path.abspath(os.path.join(output_dir, "BoundingBoxes"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    return semantic_dir, frames_dir, bbox_dir

def process_frame(width, height, semantic_dir, frames_dir, bbox_dir, data, frame_counter, world, camera, K, bbox_distance_range=(0.1, 50.0)):
    """Processes a single frame: pauses the simulation, captures the frame, saves the segmentation image, extracts bboxes, and resumes the simulation."""
    log("Pausing simulation...", verbose_only=True)
    if pyautogui is not None:
        pyautogui.press('`')
        pyautogui.write(f'HighResShot filename={os.path.join(frames_dir, f"{frame_counter}.png")} {width}x{height}')
        pyautogui.press('enter')
        log("Console command executed.", verbose_only=True)
    else:
        log("pyautogui is unavailable; skipping HighResShot console command.", verbose_only=True)

    if "semantic_segmentation" not in data or data["semantic_segmentation"] is None:
        log("WARNING: Missing semantic segmentation data for this frame; skipping export.", verbose_only=False)
        return frame_counter

    semantic_image = convert_image_to_array(data["semantic_segmentation"])
    if semantic_image is None:
        log("WARNING: Semantic image conversion returned no data for this frame; skipping export.", verbose_only=False)
        return frame_counter

    data["semantic_segmentation"].save_to_disk(os.path.join(semantic_dir, f"segmentation_{frame_counter-1}.png"))
    log(f"Saved semantic segmentation to {os.path.join(semantic_dir, f'segmentation_{frame_counter-1}.png')}", verbose_only=True)
    bboxes = get_bounding_boxes(world, camera, K, semantic_image=semantic_image, bbox_distance_range=bbox_distance_range)
    bbox_file = os.path.join(bbox_dir, f"bboxes_{frame_counter}.json")
    with open(bbox_file, 'w') as f:
        json.dump(bboxes, f, indent=4)

    data.clear()
    frame_counter += 1
            
    log(f"Resuming simulation...", verbose_only=True)
    return frame_counter

def cleanup(client, world, original_settings, vehicle, camera, args, vehicles_id, all_id):
    """Cleans up actors, resets world settings, and deletes trailing HighResShot frames.
    
    Args:
        client: The CARLA client object.
        world: The CARLA world object.
        original_settings: The original settings to restore.
        vehicle: The spawned CARLA vehicle.
        camera: The spawned CARLA camera.
        args: The command-line arguments.
        vehicles_id (list): List of IDs for spawned vehicles.
        all_id (list): List of IDs for spawned walkers and controllers.
    """
    log("Cleaning up actors and resetting world settings...", verbose_only=False)
    if camera:
        camera.destroy()
    if vehicle:
        vehicle.destroy()
    destroy_commands = []
    if vehicles_id:
        destroy_commands.extend([carla.command.DestroyActor(x) for x in vehicles_id])
    if all_id:
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        all_actors = world.get_actors(all_id)
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        destroy_commands.extend([carla.command.DestroyActor(x) for x in all_id])
    if destroy_commands:
        results = client.apply_batch_sync(destroy_commands)
    log(f"Successfully sent cleanup command for {len(vehicles_id) + len(all_id)} actors.", verbose_only=True)
    # if world and original_settings:
    #     world.apply_settings(original_settings)

    semantic_dir, frames_dir, bbox_dir = create_directories(args.output_dir)
    remove_file_if_exists(os.path.join(semantic_dir, f"segmentation_0.png"))
    remove_file_if_exists(os.path.join(bbox_dir, f"bboxes_{args.num_frames_export}.json"))
    for filename in os.listdir(frames_dir):
        if filename.startswith(str(args.num_frames_export)):
            remove_file_if_exists(os.path.join(frames_dir, filename))


def rename_output_dir(args):
    """Renames the output directory based on the selected map and weather.
    
    Args:
        args: The parsed command-line arguments.
    """
    base_output_dir = os.path.abspath(args.output_dir)
    new_output_dir = os.path.join(base_output_dir, f"{args.map_name}_{args.weather_preset}")
    args.output_dir = new_output_dir
    log(f"Output directory set to: {args.output_dir}", verbose_only=False)


def adjust_map_and_weather(client, world, args):
    """Adjusts the CARLA map and weather settings.

    This will attempt to load the requested map (if available) and apply the
    requested weather preset. If the map name is invalid it will keep the
    current world. Returns the (possibly new) world object.
    """
    available_maps = [
        'Mine_01', 'Town10HD_Opt',
    ]

    available_weathers = [
        'ClearNoon','CloudyNoon','WetNoon','WetCloudyNoon','SoftRainNoon','MidRainyNoon','HardRainNoon',
        'ClearSunset','CloudySunset','WetSunset','WetCloudySunset','SoftRainSunset','MidRainSunset','HardRainSunset'
    ]

    # Load map if requested and available
    map_name = getattr(args, 'map_name', None)
    if map_name and map_name in available_maps:
        current_map = world.get_map().name
        if current_map.split('/')[-1] != map_name:
            try:
                world = client.load_world(map_name, map_layers=carla.MapLayer.All)
                log(f"Loaded map: {map_name}", verbose_only=False)
            except Exception as e:
                log(f"Failed to load map '{map_name}': {e}", verbose_only=False)
        else:
            log(f"Map '{map_name}' is already loaded. Skipping map load.", verbose_only=False)
    else:
        if map_name:
            log(f"Map '{map_name}' not recognized. Keeping current map.", verbose_only=False)

    # Apply weather preset if available
    weather_name = getattr(args, 'weather_preset', None)
    if weather_name and weather_name in available_weathers:
        wp = getattr(carla.WeatherParameters, weather_name, None)
        if wp is not None:
            current_weather = world.get_weather()
            if current_weather != wp:
                try:
                    world.set_weather(wp)
                    log(f"Applied weather preset: {weather_name}", verbose_only=False)
                except Exception as e:
                    log(f"Failed to apply weather '{weather_name}': {e}", verbose_only=False)
            else:
                log(f"Weather preset '{weather_name}' is already active. Skipping weather load.", verbose_only=False)
        else:
            log(f"Weather preset '{weather_name}' not found in carla.WeatherParameters.", verbose_only=False)
    else:
        if weather_name:
            log(f"Weather preset '{weather_name}' not recognized. Keeping current weather.", verbose_only=False)

    return world
    
if __name__ == "__main__":
    """Main entry point for the script."""
    args = parse_args()
    VERBOSE = args.verbose
    rename_output_dir(args)
    client = None
    world = None
    vehicle = None
    camera = None
    original_settings = None
    vehicles_id = []
    all_id = []
    frame_counter = 1

    try:
        client, world, original_settings = setup_carla(args)
        time.sleep(2)
        allow_saving_Gbuffers()
        time.sleep(1)
        vehicle, camera, vehicles_id, all_id = spawn_actors(client, world, args.width, args.height, args.num_vehicles, args.num_walkers)
        time.sleep(1)
        frame_counter, vehicles_id, all_id = run_simulation(world, vehicle, camera, args, frame_counter, client, vehicles_id, all_id)
        
    except KeyboardInterrupt:
        log("\nSimulation stopped.", verbose_only=False)
    finally:
        cleanup(client, world, original_settings, vehicle, camera, args, vehicles_id, all_id)

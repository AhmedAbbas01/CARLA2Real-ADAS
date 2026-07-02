import carla
import time
import os
import pyautogui
import random
import argparse
import sys
import numpy as np
import math
import json


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
    parser.add_argument("--num_vehicles", type=int, default=100, help="Number of vehicles to spawn.")
    parser.add_argument("--num_walkers", type=int, default=40, help="Number of pedestrians to spawn.")
    return parser.parse_args()

def remove_file_if_exists(file_path):
    """Removes a file at the specified path if it exists.
    
    Args:
        file_path (str): The path to the file to be removed.
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"{file_path} has been removed.")
    else:
        print(f"{file_path} does not exist.")

def allow_saving_Gbuffers():
    """Allows saving G-buffers in the Unreal Engine console.
    """    
    # Allow saving G-buffers in the Unreal Engine console
    pyautogui.press('`')
    pyautogui.write('r.BufferVisualizationDumpFrames 1')
    pyautogui.press('enter')
    print("G-buffer saving enabled in Unreal Engine console.")

def spawn_traffic(client, world, num_vehicles, num_walkers):
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
        print(f"Requested {num_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points")
        num_vehicles = number_of_spawn_points

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
        if response.error:
            pass
        else:
            vehicles_id.append(response.actor_id)

    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
            
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
        else:
            walker_speed.append(0.0)
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
    results = client.apply_batch_sync(batch, True)
    walkers_list = []
    all_id = []
    for i in range(len(results)):
        if not results[i].error:
            walkers_list.append({"id": results[i].actor_id})
            
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    
    for i in range(len(results)):
        if not results[i].error:
            walkers_list[i]["con"] = results[i].actor_id
            
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    world.tick()

    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print(f"Spawned {len(vehicles_id)} vehicles and {len(walkers_list)} walkers.")
    
    return vehicles_id, all_id

def setup_carla(width, height, num_vehicles, num_walkers):
    """Connects to CARLA, configures the world, and spawns the necessary actors.
    
    Args:
        width (int): Width of the camera resolution.
        height (int): Height of the camera resolution.
        num_vehicles (int): Number of vehicles to spawn.
        num_walkers (int): Number of walkers to spawn.
        
    Returns:
        tuple: A tuple containing the CARLA client, world, original settings, vehicle, camera, vehicles_id, and all_id.
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)

        world = client.get_world()
        original_settings = world.get_settings()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]

        if spawn_points:
            random_spawn_point = random.choice(spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, random_spawn_point)
            print(f"Vehicle spawned at location: {random_spawn_point.location}")
        else:
            print("No spawn points available in the map.")
            sys.exit(1)

        vehicle.set_autopilot(True)

        vehicles_id, all_id = spawn_traffic(client, world, num_vehicles, num_walkers)

        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(carla.Location(x=vehicle.bounding_box.extent.x + 0.1, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        return client, world, original_settings, vehicle, camera, vehicles_id, all_id
    except Exception as e:
        print(f"Error setting up CARLA: {e}")
        sys.exit(1)

def get_bounding_boxes(world, camera, K):
    bounding_boxes = []
    
    camera_transform = camera.get_transform()
    c2w = np.array(camera_transform.get_matrix())
    w2c = np.linalg.inv(c2w)
    
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')
    traffic_signs = actors.filter('traffic.traffic_sign.*')
    traffic_lights = actors.filter('traffic.traffic_light.*')
    
    all_objects = list(vehicles) + list(walkers) + list(traffic_signs) + list(traffic_lights)
    
    for obj in all_objects:
        dist = obj.get_transform().location.distance(camera_transform.location)
        
        if dist > 50 or dist < 0.1:
            continue
            
        if not hasattr(obj, 'bounding_box') or obj.bounding_box is None:
            continue

        bb = obj.bounding_box
        verts = bb.get_world_vertices(obj.get_transform())
        
        verts_camera = []
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1.0])
            p_camera = np.dot(w2c, p_world)
            verts_camera.append(p_camera)
            
        if any(p[0] <= 0 for p in verts_camera):
            continue
            
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
        
        image_w = K[0, 2] * 2
        image_h = K[1, 2] * 2
        
        x_min = max(0.0, min(x_min, image_w))
        x_max = max(0.0, min(x_max, image_w))
        y_min = max(0.0, min(y_min, image_h))
        y_max = max(0.0, min(y_max, image_h))

        if x_min >= x_max or y_min >= y_max:
            continue
            
        bounding_boxes.append({
            "type": obj.type_id,
            "id": obj.id,
            "distance": float(dist),
            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
        })
            
    return bounding_boxes
    
def run_simulation(world, vehicle, camera, args, frame_counter=0):
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

    semantic_dir = os.path.abspath(os.path.join(output_dir, "Semantic"))
    frames_dir = os.path.abspath(os.path.join(output_dir, "Frames"))
    bbox_dir = os.path.abspath(os.path.join(output_dir, "BoundingBoxes"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

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

    print("Starting simulation... Press Ctrl+C to stop.")
    time.sleep(2)
    while True:
        vehicle.set_autopilot(True)
        spectator.set_transform(camera.get_transform())
        world.tick()

        while True:
            if "semantic_segmentation" in data:
                break
            time.sleep(0.001)

        world_tick_counter += 1

        if world_tick_counter % export_step == 0:
            frame_counter = process_frame(width, height, semantic_dir, frames_dir, bbox_dir, data, frame_counter, world, camera, K)
            time.sleep(1)

            frame_exported_counter += 1
            if frame_exported_counter == num_frames_export:
                print("Finished data generation...")
                break
    time.sleep(0.1)
    remove_file_if_exists(os.path.join(semantic_dir, f"segmentation_0.png"))
    remove_file_if_exists(os.path.join(frames_dir, f"{args.num_frames_export}.png"))
    remove_file_if_exists(os.path.join(bbox_dir, f"bboxes_{args.num_frames_export}.json"))

    return frame_counter

def process_frame(width, height, semantic_dir, frames_dir, bbox_dir, data, frame_counter, world, camera, K):
    """Processes a single frame: pauses the simulation, captures the frame, saves the segmentation image, extracts bboxes, and resumes the simulation."""
    print("Pausing simulation...")
    pyautogui.press('`')
    pyautogui.write(f'HighResShot filename={os.path.join(frames_dir, f"{frame_counter}.png")} {width}x{height}')
    pyautogui.press('enter')
    print("Console command executed.")

    data["semantic_segmentation"].save_to_disk(os.path.join(semantic_dir, f"segmentation_{frame_counter-1}.png"))
    bboxes = get_bounding_boxes(world, camera, K)
    bbox_file = os.path.join(bbox_dir, f"bboxes_{frame_counter}.json")
    with open(bbox_file, 'w') as f:
        json.dump(bboxes, f, indent=4)

    data.clear()
    frame_counter += 1
            
    print("Resuming simulation...")
    return frame_counter

def cleanup(client, world, original_settings, vehicle, camera, frame_counter, vehicles_id, all_id):
    """Cleans up actors, resets world settings, and deletes trailing HighResShot frames.
    
    Args:
        client: The CARLA client object.
        world: The CARLA world object.
        original_settings: The original settings to restore.
        vehicle: The spawned CARLA vehicle.
        camera: The spawned CARLA camera.
        frame_counter (int): The final frame counter to identify the extra generated frame.
        vehicles_id (list): List of IDs for spawned vehicles.
        all_id (list): List of IDs for spawned walkers and controllers.
    """
    
    if camera:
        camera.destroy()
    if vehicle:
        vehicle.destroy()

    if vehicles_id:
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_id])
    if all_id:
        # all_actors = world.get_actors(all_id)
        # for i in range(0, len(all_id), 2):
        #     all_actors[i].stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
    if world and original_settings:
        world.apply_settings(original_settings)

    # time.sleep(1)
    # print("Deleting the additional HighResShot frames...")
    # number_prefix = str(frame_counter - 1)
    # print(number_prefix)

    # if os.path.exists(frames_dir):
    #     for filename in os.listdir(frames_dir):
    #         if filename.startswith(number_prefix):
    #             file_path = os.path.join(frames_dir, filename)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)
    #                 print(f"Deleted: {file_path}")


if __name__ == "__main__":
    """Main entry point for the script."""
    args = parse_args()
    
    client = None
    world = None
    vehicle = None
    camera = None
    original_settings = None
    vehicles_id = []
    all_id = []
    frame_counter = 1

    try:
        client, world, original_settings, vehicle, camera, vehicles_id, all_id = setup_carla(args.width, args.height, args.num_vehicles, args.num_walkers)
        time.sleep(2)
        allow_saving_Gbuffers()
        frame_counter = run_simulation(world, vehicle, camera, args, frame_counter)
        
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        cleanup(client, world, original_settings, vehicle, camera, frame_counter, vehicles_id, all_id)

import argparse
import logging
import math
import sys
import os
import queue
import time
import numpy as np
import pandas as pd

# Add relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "closed_loop")))
from perception import EnsemblePerception, SinglePerception
from controller import ADASController

try:
    import carla
except ImportError:
    logging.critical("CARLA module not found. Please ensure CARLA PythonAPI is installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScenarioEvaluator")

class ScenarioEvaluator:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # Load perception
        if args.perception_mode == "ensemble":
            self.perception = EnsemblePerception(
                yolo_path=args.yolo_model,
                rtdetr_path=args.rtdetr_model,
                faster_path=args.faster_model,
                depth_path=args.depth_model,
                num_classes=args.num_classes
            )
        else:
            self.perception = SinglePerception(
                yolo_path=args.yolo_model,
                depth_path=args.depth_model,
                conf_threshold=args.conf_threshold
            )
        self.perception.load_models()
        
        # Create ADAS Controller instance for logic
        self.controller = ADASController(
            perception_module=self.perception,
            cruise_throttle=args.cruise_throttle,
            warning_distance=args.warning_distance,
            brake_distance=args.brake_distance,
            max_speed=args.max_speed
        )
        self.controller.world = self.world
        self.controller.client = self.client
        
        self.metrics = []
        self.ego = None
        self.camera = None
        self.collision_sensor = None
        self.image_queue = queue.Queue()
        self.collision_queue = queue.Queue()

    def enable_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def disable_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def find_straight_road(self):
        # for sp in self.map.get_spawn_points():
        #     wp = self.map.get_waypoint(sp.location)
        #     next_wps = wp.next(60.0)
        #     if next_wps:
        #         print(f"Found straight road at {wp.transform.location} with next waypoint at {next_wps[0].transform.location}")
        #         return wp
        returned = self.map.get_waypoint(self.map.get_spawn_points()[10].location)
        print(f"No straight road found. Defaulting to waypoint at {returned.transform.location}")
        return returned

    def spawn_ego(self, transform):
        bp = self.world.get_blueprint_library().find('vehicle.ue4.mercedes.ccc')
        # print(bp.id, transform)
        # self.world.tick()
        self.ego = self.world.spawn_actor(bp, transform)
        self.controller.vehicle = self.ego
        
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=self.ego.bounding_box.extent.x + 0.1, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.ego)
        
        # Flush queue
        while not self.image_queue.empty():
            self.image_queue.get()
        self.camera.listen(self.image_queue.put)

        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego)
        
        while not self.collision_queue.empty():
            self.collision_queue.get()
        self.collision_sensor.listen(self.collision_queue.put)

    def cleanup_actors(self, target_actor):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        if self.ego:
            self.ego.destroy()
        if target_actor:
            target_actor.destroy()
        self.world.tick()

    def run_scenario(self, scenario_name, wp_ego, wp_target):
        # Ensure ego speed limit based on scenario (50 km/h for A/C, 30 km/h for B)
        self.controller.max_speed = 50.0 if scenario_name in ["Scenario_A", "Scenario_C"] else 30.0
        
        self.spawn_ego(wp_ego.transform)
        target_actor = None
        
        if scenario_name == "Scenario_A":
            bp = self.world.get_blueprint_library().filter("vehicle.taxi.ford")[0]
            target_actor = self.world.spawn_actor(bp, wp_target.transform)
            target_actor.apply_control(carla.VehicleControl(hand_brake=True))
            
        elif scenario_name == "Scenario_B":
            right_vector = wp_target.transform.get_right_vector()
            target_loc = wp_target.transform.location + right_vector * 4.0
            target_loc.z += 2
            yaw = wp_target.transform.rotation.yaw - 90
            target_transform = carla.Transform(target_loc, carla.Rotation(yaw=yaw))
            # Pick a pedestrian blueprint
            bp = np.random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
            bp.set_attribute('is_invincible', 'false')
            # target_loc = wp_target.transform.location
            # target_loc.z += 2
            # target_transform = carla.Transform(target_loc)
            target_actor = self.world.spawn_actor(bp, target_transform)
            
        elif scenario_name == "Scenario_C":
            bp = self.world.get_blueprint_library().filter("vehicle.taxi.ford")[0]
            target_actor = self.world.spawn_actor(bp, wp_target.transform)
            target_actor.apply_control(carla.VehicleControl(throttle=0.5))
            
        self.world.tick() # flush initial
        
        has_collided = False
        emergency_brakes = 0
        min_ttc = float('inf')
        min_margin = float('inf')
        latencies = []
        
        spectator = self.world.get_spectator()
        start_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        logger.info(f"Running {scenario_name}...")
        while True:
            transform = carla.Transform(
                self.ego.get_transform().transform(carla.Location(x=-4, z=2.5)),
                self.ego.get_transform().rotation
            )
            spectator.set_transform(transform)
            self.world.tick()
            t = self.world.get_snapshot().timestamp.elapsed_seconds - start_time
            
            if not self.collision_queue.empty():
                has_collided = True
                logger.warning(f"Collision detected in {scenario_name} at t={t:.2f}s!")
                break
                
            if scenario_name == "Scenario_C" and t > 3.0:
                if target_actor and target_actor.is_alive:
                    target_actor.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                    
            if scenario_name == "Scenario_B" and target_actor and target_actor.is_alive:
                dist = self.ego.get_location().distance(target_actor.get_location())
                if dist < 25.0:
                    vec = target_actor.get_transform().get_forward_vector()
                    target_actor.apply_control(carla.WalkerControl(direction=vec, speed=20.0))
                    logger.info(f"Pedestrian moving at t={t:.2f}s, distance={dist:.2f}m")
            
            if not self.image_queue.empty():
                image = self.image_queue.get()
                # Empty queue to keep up with latest frame
                while not self.image_queue.empty():
                    image = self.image_queue.get()
                    
                array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
                frame = array[:, :, :3].copy()
                
                # High-precision timer block
                t0 = time.perf_counter()

                boxes, scores, labels, distances = self.perception.predict(frame)
                throttle, brake, danger, obj, detected_objects = self.controller.calculate_control(
                    frame, boxes, scores, labels, distances
                )
                steer = self.controller.calculate_steering()
                    
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

                t1 = time.perf_counter()
                
                latencies.append((t1 - t0) * 1000.0) # Convert to ms
                
                control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
                self.ego.apply_control(control)
                
                if brake >= 0.8:
                    emergency_brakes += 1
                
                # Compute TTC and Margin using Ground Truth info for validation
                if target_actor and target_actor.is_alive:
                    ego_vel = self.ego.get_velocity()
                    target_vel = target_actor.get_velocity()
                    
                    v_ego = np.sqrt(ego_vel.x**2 + ego_vel.y**2)
                    v_target = np.sqrt(target_vel.x**2 + target_vel.y**2)
                    v_rel = v_ego - v_target
                    
                    d_rel = self.ego.get_location().distance(target_actor.get_location())
                    d_rel = max(0.0, d_rel - 4.5) # Deduct bounding box approx distance
                    
                    if v_rel > 0.5:
                        ttc = d_rel / v_rel
                        min_ttc = min(min_ttc, ttc)
                    
                    # Check stopping distance margin
                    if v_ego < 0.1 and d_rel < 20.0:
                        min_margin = min(min_margin, d_rel)
                        if t > 5.0: # End early if fully stopped near target
                            logger.info(f"Vehicle stopped safely. Margin: {min_margin:.2f}m")
                            break
                            
            if t > 15.0:
                logger.info("Scenario timeout reached.")
                break
                
        self.cleanup_actors(target_actor)
        
        avg_latency = np.mean(latencies) if latencies else 0.0
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        return {
            "Scenario": scenario_name,
            "Collision": int(has_collided),
            "Emergency Brakes": emergency_brakes,
            "Min TTC (s)": min_ttc if min_ttc != float('inf') else None,
            "Margin (m)": min_margin if min_margin != float('inf') else None,
            "Latency (ms)": avg_latency,
            "FPS": fps
        }

    def run_experiments(self, runs_per_scenario=3):
        self.enable_sync()
        wp_ego = self.find_straight_road()
        
        for i in range(runs_per_scenario):
            logger.info(f"--- Iteration {i+1}/{runs_per_scenario} ---")
            res_a = self.run_scenario("Scenario_A", wp_ego, wp_ego.next(30.0)[0])
            self.metrics.append(res_a)
            
            res_b = self.run_scenario("Scenario_B", wp_ego, wp_ego.next(10.0)[0])
            self.metrics.append(res_b)
            
            res_c = self.run_scenario("Scenario_C", wp_ego, wp_ego.next(25.0)[0])
            self.metrics.append(res_c)
            
        self.disable_sync()
        self.generate_report()
        
    def generate_report(self):
        df = pd.DataFrame(self.metrics)
        
        # Aggregate metrics for output
        summary = df.groupby("Scenario").agg({
            "Collision": lambda x: f"{(x.mean() * 100):.1f}%",
            "Emergency Brakes": "sum",
            "Min TTC (s)": "mean",
            "Margin (m)": "mean",
            "Latency (ms)": "mean",
            "FPS": "mean"
        }).reset_index()
        
        summary = summary.rename(columns={"Collision": "Collision Rate"})
        
        # Format floats
        summary["Min TTC (s)"] = summary["Min TTC (s)"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        summary["Margin (m)"] = summary["Margin (m)"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        summary["Latency (ms)"] = summary["Latency (ms)"].map(lambda x: f"{x:.1f}")
        summary["FPS"] = summary["FPS"].map(lambda x: f"{x:.1f}")

        print("\n" + "="*50)
        print("### Quantitative Results (Markdown)")
        print("="*50 + "\n")
        print(summary.to_markdown(index=False))
        
        print("\n" + "="*50)
        print("### Quantitative Results (LaTeX)")
        print("="*50 + "\n")
        print(summary.to_latex(index=False, caption="Closed-Loop ADAS Evaluation Metrics", label="tab:adas_metrics"))

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
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()
        if self.client and self.world:
            actors = self.world.get_actors()
            destroy_batch = [
                carla.command.DestroyActor(x) for x in actors 
                if x.type_id.startswith(('vehicle.', 'sensor.', 'walker.', 'controller.'))
            ]
            if destroy_batch:
                self.client.apply_batch(destroy_batch)
                logger.info("Destroyed %d actors.", len(destroy_batch))
        logger.info("Cleanup finished.")


def main():
    parser = argparse.ArgumentParser(description="Closed-Loop CARLA Quantitative Evaluation Module")
    parser.add_argument("--perception_mode", type=str, choices=["single", "ensemble"], default="ensemble")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--rtdetr-model", type=str, default="rtdetr.pt")
    parser.add_argument("--faster-model", type=str, default="faster_rcnn.pth")
    parser.add_argument("--depth-model", type=str, default="depth_anything_v2_metric_hypersim_vits.pth")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.40)
    parser.add_argument("--cruise-throttle", type=float, default=0.50)
    parser.add_argument("--warning-distance", type=float, default=15.0)
    parser.add_argument("--brake-distance", type=float, default=7.0)
    parser.add_argument("--max-speed", type=float, default=50.0)
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per scenario")

    args = parser.parse_args()
    
    try:
        evaluator = ScenarioEvaluator(args)
        evaluator.run_experiments(runs_per_scenario=args.runs)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping...")
    except Exception as e:
        logger.error("An unexpected error occurred in run loop: %s", e, exc_info=True)
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    main()
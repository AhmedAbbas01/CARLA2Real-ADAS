import argparse
import logging
import sys
import queue
from perception import EnsemblePerception, SinglePerception
from controller import ADASController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("Main")

def main():
    parser = argparse.ArgumentParser(description="Closed-Loop ADAS script for CARLA.")
    parser.add_argument("--mode", type=str, choices=["single", "ensemble"], default="ensemble", help="Perception mode")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host address")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--rtdetr-model", type=str, default="rtdetr.pt", help="Path to RT-DETR weights")
    parser.add_argument("--faster-model", type=str, default="faster_rcnn.pth", help="Path to Faster R-CNN weights")
    parser.add_argument("--depth-model", type=str, default="depth_anything_v2_metric_hypersim_vits.pth", help="Path to Depth model weights")
    parser.add_argument("--num-classes", type=int, default=8, help="Number of classes for Faster R-CNN")
    parser.add_argument("--conf-threshold", type=float, default=0.40, help="Confidence threshold for YOLO")
    parser.add_argument("--cruise-throttle", type=float, default=0.35, help="Cruise control throttle")
    parser.add_argument("--warning-distance", type=float, default=15.0, help="Warning distance for braking")
    parser.add_argument("--brake-distance", type=float, default=7.0, help="Brake distance for braking")
    parser.add_argument("--lane-width", type=float, default=3.5, help="Lane width for lateral distance calculation")
    parser.add_argument("--max-speed", type=float, default=30.0, help="Maximum safe cruising speed in km/h")
    parser.add_argument("--visualize", action="store_true", help="Enable OpenCV visualization")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Initialize Perception Modality
    if args.mode == "ensemble":
        perception = EnsemblePerception(
            yolo_path=args.yolo_model,
            rtdetr_path=args.rtdetr_model,
            faster_path=args.faster_model,
            depth_path=args.depth_model,
            num_classes=args.num_classes
        )
    else:
        perception = SinglePerception(
            yolo_path=args.yolo_model,
            depth_path=args.depth_model,
            conf_threshold=args.conf_threshold
        )

    try:
        perception.load_models()
    except Exception as e:
        logger.critical(f"Failed to load models: {e}")
        sys.exit(1)

    # Initialize Controller
    controller = ADASController(
        perception_module=perception,
        host=args.host, port=args.port,
        cruise_throttle=args.cruise_throttle,
        warning_distance=args.warning_distance,
        brake_distance=args.brake_distance,
        lane_width=args.lane_width,
        max_speed=args.max_speed,
        visualize=args.visualize
    )
    
    try:
        controller.connect_carla()
        controller.spawn_actors()

        logger.info("Enabling synchronous mode...")
        settings = controller.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        controller.world.apply_settings(settings)

        image_queue = queue.Queue()
        controller.camera.listen(image_queue.put)
        spectator = controller.world.get_spectator()

        logger.info("Closed-loop ADAS running in SYNC mode. Press Ctrl+C to stop.")

        while True:
            controller.world.tick()
            image = image_queue.get()
            controller.process_image(image)
            spectator.set_transform(controller.get_third_person_camera_transform())
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping...")
    except Exception as e:
        logger.error("An unexpected error occurred in run loop: %s", e, exc_info=True)
    finally:
        controller.cleanup()

if __name__ == "__main__":
    main()
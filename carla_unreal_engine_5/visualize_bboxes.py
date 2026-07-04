import os
import json
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on frames.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to the directory containing frames.")
    parser.add_argument("--bboxes_dir", type=str, required=True, help="Path to the directory containing bounding box JSONs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the visualized images.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    frames = [f for f in os.listdir(args.frames_dir) if f.endswith('.png')]
    
    if not frames:
        print(f"No frames found in {args.frames_dir}")
        return

    for frame_name in frames:
        frame_counter = os.path.splitext(frame_name)[0].split('-')[-1]
        
        frame_path = os.path.join(args.frames_dir, frame_name)
        bbox_name = f"bboxes_{frame_counter}.json"
        bbox_path = os.path.join(args.bboxes_dir, bbox_name)
        
        if not os.path.exists(bbox_path):
            print(f"Warning: Bounding box file {bbox_path} not found for frame {frame_name}.")
            continue
            
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read image {frame_path}.")
            continue
            
        with open(bbox_path, 'r') as f:
            bboxes = json.load(f)
                
        for obj in bboxes:
            obj_type = obj.get('type', 'unknown')
            dist = obj.get('distance', 0.0)
            bbox = obj.get('bbox', [0, 0, 0, 0])
            
            x_min, y_min, x_max, y_max = map(int, bbox)
            type_label = obj_type.split('.')[-1] if isinstance(obj_type, str) else str(obj_type)
            
            # Pick a color based on the object type (BGR format)
            if type_label in {"Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Train"}:
                color = (0, 255, 0) # Green
            elif type_label in {"Rider", "Pedestrian"}:
                color = (0, 0, 255) # Red
            elif type_label in {"TrafficSigns"}:
                color = (0, 255, 255) # Yellow
            elif type_label in {"TrafficLight"}:
                color = (255, 0, 255) # Magenta
            elif type_label in {"RailTrack"}:
                color = (255, 255, 255) # White
            else:
                color = (255, 0, 0) # Blue
                
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            
            label = f"{type_label} ({dist:.1f}m)"
            print(f"Drawing bbox for {obj_type} at distance {dist:.1f}m")
            cv2.putText(img, label, (x_min, max(y_min - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        output_path = os.path.join(args.output_dir, f"visualized_{frame_name}")
        cv2.imwrite(output_path, img)
        print(f"Saved visualized frame to {output_path}")

if __name__ == "__main__":
    main()
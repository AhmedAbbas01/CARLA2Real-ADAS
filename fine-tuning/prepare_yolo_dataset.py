"""
Prepare Extracted dataset for YOLOv8 fine-tuning.

The yolo dataset will be built with the following structure:
yolo_kitti_dataset/
├── train/
│   ├── images/
│   │   ├── img_001.png
│   │   └── img_002.png
│   └── labels/
│       ├── img_001.txt
│       └── img_002.txt
├── val/
│   ├── images/
│   │   └── img_003.png
│   └── labels/
│       └── img_003.txt
└── test/ 
    ├── images/
    │   └── img_004.png
    └── labels/
        └── img_004.txt

Expected raw layout under --data-root:
  Carla2KKitti/FinalColor-62.png
  BoundingBoxes/bboxes-62.json
"""
# Standard library imports
import os
import json
import glob
import shutil
import argparse
from typing import List, Dict

# Third-party imports
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml

# Default relative directory names inside each dataset sequence
DEFAULT_DESIRED_MODEL = "Carla2Kitti"
INPUT_BBOX_DIR = "BoundingBoxes"
DEFAULT_OUTPUT_DIR = "yolo_kitti_dataset"
CLASS_MAPPING = {
    "Car": 0,
    "Truck": 1,
    "Bus": 2,
    "Motorcycle": 3,
    "Bicycle": 4,
    "TrafficSigns": 5,
    "TrafficLight": 6,
    "Pedestrians": 7,
}


def collect_pairs(parent_dir: str, frames_dir_name: str,
                  bbox_dir_name: str = INPUT_BBOX_DIR) -> List[Dict]:
    """Traverse immediate subfolders under parent_dir and collect matching
    image/json pairs. Each sequence folder is expected to contain two
    subdirectories: frames_dir_name and bbox_dir_name.

    Returns a list of dicts with keys: img_path, json_path, frame_idx,
    source_folder_name.
    """
    pairs = []
    # iterate over immediate children of parent_dir
    for entry in os.listdir(parent_dir):
        folder = os.path.join(parent_dir, entry)
        if not os.path.isdir(folder):
            continue

        frames_dir = os.path.join(folder, frames_dir_name)
        bbox_dir = os.path.join(folder, bbox_dir_name)

        if not os.path.exists(frames_dir) or not os.path.exists(bbox_dir):
            print(f"Warning: Missing Frames or BoundingBoxes in {folder}. Skipping.")
            continue

        image_files = glob.glob(os.path.join(frames_dir, "*.png"))
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            frame_idx = base_name.split('-')[-1]
            json_path = os.path.join(bbox_dir, f"bboxes_{frame_idx}.json")
            if os.path.exists(json_path):
                pairs.append({
                    "img_path": img_path,
                    "json_path": json_path,
                    "frame_idx": frame_idx,
                    "source_folder_name": os.path.basename(os.path.normpath(folder)),
                })

    return pairs


def prepare_output_dirs(output_dir: str):
    """Create the standard YOLO folders for train/val/test splits."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)


def write_yolo_label_file(yolo_lines: List[str], out_path: str):
    """Write YOLO formatted lines to disk."""
    with open(out_path, "w") as out_f:
        out_f.write("\n".join(yolo_lines))


def process_pair(pair: Dict, output_dir: str, split_name: str):
    """Process a single (image,json) pair: copy image and write label file."""
    img_path = pair["img_path"]
    json_path = pair["json_path"]
    frame_idx = pair["frame_idx"]
    src_folder = pair["source_folder_name"]

    # Read image to get dimensions
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    unique_name = f"{src_folder}_frame_{frame_idx}"

    # copy image
    shutil.copy(img_path, os.path.join(output_dir, split_name, "images", f"{unique_name}.png"))

    # parse json and format YOLO lines
    yolo_lines: List[str] = []
    with open(json_path, "r") as f:
        bbox_data = json.load(f)
        if isinstance(bbox_data, dict):
            bbox_data = [bbox_data]

        for obj in bbox_data:
            obj_type = obj.get("type")
            if obj_type not in CLASS_MAPPING:
                print(f"Warning: Object type '{obj_type}' not in CLASS_MAPPING. Skipping object in {json_path}.")
                continue

            class_id = CLASS_MAPPING[obj_type]
            xmin, ymin, xmax, ymax = obj["bbox"]

            box_w = xmax - xmin
            box_h = ymax - ymin
            x_center = xmin + (box_w / 2.0)
            y_center = ymin + (box_h / 2.0)

            norm_x = x_center / img_w
            norm_y = y_center / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            yolo_lines.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    txt_out_path = os.path.join(output_dir, split_name, "labels", f"{unique_name}.txt")
    write_yolo_label_file(yolo_lines, txt_out_path)


def process_split(pairs: List[Dict], output_dir: str, split_name: str):
    """Process all pairs belonging to a split."""
    for pair in pairs:
        process_pair(pair, output_dir, split_name)


def create_dataset_config_yaml_file(output_dir: str):
    """Create a dataset.yaml file for YOLOv8 training."""
    yaml_content = {
        'path': os.path.abspath(output_dir), # Absolute path prevents layout resolution issues in YOLO
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {v: k for k, v in CLASS_MAPPING.items()} # Inverts CLASS_MAPPING to: {0: 'Vehicle', 1: 'Person'...}
    }

    yaml_out_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_out_path, 'w') as yaml_file:
        # sort_keys=False preserves the sequential integer ordering of the class IDs
        yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

    return yaml_out_path
    

def main():
    """Main entrypoint: parse args, collect data, split and process."""
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from extracted frames and bboxes.")
    parser.add_argument("parent_dir", help="Parent directory containing sequence subfolders")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output yolo dataset directory")
    parser.add_argument("--desired-model", default=DEFAULT_DESIRED_MODEL, help="Desired CARLA2Real enhanced images model")
    args = parser.parse_args()

    all_data_pairs = collect_pairs(args.parent_dir, args.desired_model)
    print(f"Total valid image/json pairs found across all folders: {len(all_data_pairs)}")

    # Perform a 3-way split (70% train, 15% val, 15% test)
    train_pairs, temp_pairs = train_test_split(all_data_pairs, test_size=0.3, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    prepare_output_dirs(args.output_dir)

    print("Processing training split...")
    process_split(train_pairs, args.output_dir, "train")

    print("Processing validation split...")
    process_split(val_pairs, args.output_dir, "val")

    print("Processing testing split...")
    process_split(test_pairs, args.output_dir, "test")

    yaml_out_path = create_dataset_config_yaml_file(args.output_dir)
    print(f"\nConfiguration file created successfully at: {yaml_out_path}")

    print(f"Finished structure migration! Output saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

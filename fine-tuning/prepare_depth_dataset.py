"""
Prepare Extracted dataset for Apple ml-depth-pro fine-tuning.

The depth dataset will be built with the following structure:
depth_kitti_dataset/
├── images/
│   ├── img_1.png
│   └── img_2.png
└── labels/
    ├── bboxes_1.json
    └── bboxes_2.json

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
DEFAULT_OUTPUT_DIR = "depth_kitti_dataset"
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


def process_pair(pair: Dict, output_dir: str, split_name: str):
    """Process a single (image,json) pair: copy image and write label file."""
    img_path = pair["img_path"]
    json_path = pair["json_path"]
    frame_idx = pair["frame_idx"]
    src_folder = pair["source_folder_name"]
    unique_name = f"{src_folder}_frame_{frame_idx}"

    # copy image
    shutil.copy(img_path, os.path.join(output_dir, split_name, "images", f"{unique_name}.png"))
    # copy labels
    shutil.copy(json_path, os.path.join(output_dir, split_name, "labels", f"{unique_name}.json"))



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

    print(f"Finished structure migration! Output saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

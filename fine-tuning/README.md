# Fine-tuning dataset preparation

This folder contains a helper script, prepare_yolo_dataset.py, that converts the extracted CARLA/ADAS dataset into the directory layout expected by YOLOv8 training.

## Why this script exists

The project produces paired image and bounding-box files in a dataset structure similar to:

- Carla2Kitti/ for rendered frames
- BoundingBoxes/ for per-frame JSON annotations

YOLO training expects a different layout: separate train/val/test folders, one image file per sample, one label file per image, and a dataset.yaml file describing class names and paths. This script prepares that format automatically.

## Input format

The script expects a parent directory that contains one or more sequence folders. Each sequence folder should have:

- Carla2Kitti/ containing PNG frame files
- BoundingBoxes/ containing JSON files named like bboxes_0001.json

A typical input layout looks like this:

```text
Dataset/
├── Town10HD_Opt_ClearNoon/
│   ├── Carla2Kitti/
│   │   └── FinalColor-62.png
│   └── BoundingBoxes/
│       └── bboxes-62.json
└── Town10HD_Opt_CloudyNoon/
    ├── Carla2Kitti/
    └── BoundingBoxes/
```


## Usage

Run the script from the repository root:

```bash
conda activate carla
python3 fine-tuning/prepare_yolo_dataset.py ./Dataset 
```

Arguments:

- positional parent_dir: directory containing the sequence subfolders
- --output-dir: destination folder for the YOLO-ready dataset (Optional. defaults to yolo_kitti_dataset)

The script will:

1. find matching image and annotation pairs across all sequence folders
2. split the data into train/val/test sets using a 70/15/15 ratio
3. copy each image into the appropriate split folder
4. write YOLO-format label files with normalized bounding boxes
5. create dataset.yaml for YOLOv8

## Expected output

After a successful run, the output directory will look like this:

```text
yolo_kitti_dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   │   └── Town10HD_Opt_ClearNoon_frame_62.png
│   └── labels/
│       └── Town10HD_Opt_ClearNoon_frame_62.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label file contains one line per detected object in YOLO format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The class mapping used by the script is:

- Car: 0
- Truck: 1
- Bus: 2
- Motorcycle: 3
- Bicycle: 4
- TrafficSigns: 5
- TrafficLight: 6
- Pedestrians: 7

## Notes

- Only objects whose type is recognized by the built-in class mapping are exported.
- Images with missing matching bounding-box JSON files are skipped.
- The generated dataset.yaml points YOLOv8 to the created train/val/test image folders.

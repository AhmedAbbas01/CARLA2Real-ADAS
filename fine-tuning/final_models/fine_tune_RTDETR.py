# %%
import os
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import RTDETR
from pathlib import Path

CLASS_NAMES = {
    0: "Car",
    1: "Truck",
    2: "Bus",
    3: "Motorcycle",
    4: "Bicycle",
    5: "TrafficSigns",
    6: "TrafficLight",
    7: "Pedestrians",
}

CLASS_COLORS = {
    0: (63, 0, 255),    # Neon Red
    1: (255, 123, 0),  # Electric Blue
    2: (0, 255, 0),    # Lime Green
    3: (0, 215, 255),  # Vibrant Yellow
    4: (255, 0, 255),  # Bright Magenta
    5: (255, 255, 0),  # Cyan / Aqua
    6: (0, 103, 255),  # Vivid Orange
    7: (255, 0, 157)   # Hot Purple
}

YOLO_DATASET = "yolo_kitti_dataset"

# %%
def _matches_frame_name(path, frame_idx):
    stem = os.path.splitext(os.path.basename(path))[0]
    frame_token = str(frame_idx)
    return (
        stem == frame_token
        or stem.endswith(f"frame_{frame_token}")
        or stem.endswith(f"_{frame_token}")
        or stem.endswith(f"frame{frame_token}")
    )


def visualize_yolo_frame(yolo_dir, frame_idx, show=True, output_path=None):
    """Show a single image from a YOLO training directory with its bounding boxes."""
    frame_idx = str(frame_idx)
    yolo_dir = os.path.abspath(yolo_dir)

    if not os.path.isdir(yolo_dir):
        raise FileNotFoundError(f"YOLO directory not found: {yolo_dir}")

    image_candidates = []
    for root, _, files in os.walk(yolo_dir):
        for filename in files:
            if filename.lower().endswith((".png")):
                if _matches_frame_name(filename, frame_idx):
                    image_candidates.append(os.path.join(root, filename))

    if not image_candidates:
        raise FileNotFoundError(f"No image found for frame index {frame_idx} under {yolo_dir}")

    image_path = image_candidates[0]
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    label_path = None
    for root, _, files in os.walk(yolo_dir):
        for filename in files:
            if filename == f"{image_stem}.txt":
                label_path = os.path.join(root, filename)
                break
        if label_path:
            break

    if label_path is None:
        for root, _, files in os.walk(yolo_dir):
            for filename in files:
                if filename == "labels.txt":
                    label_path = os.path.join(root, filename)
                    break
            if label_path:
                break

    if label_path is None:
        raise FileNotFoundError(f"No label file found for frame index {frame_idx} under {yolo_dir}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, box_w, box_h = map(float, parts[1:5])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        if max(abs(x_center), abs(y_center), abs(box_w), abs(box_h)) <= 1.0:
            x_min = int((x_center - box_w / 2.0) * img_w)
            y_min = int((y_center - box_h / 2.0) * img_h)
            x_max = int((x_center + box_w / 2.0) * img_w)
            y_max = int((y_center + box_h / 2.0) * img_h)
        else:
            x_min = int(x_center)
            y_min = int(y_center)
            x_max = int(x_center + box_w)
            y_max = int(y_center + box_h)

        x_min = max(0, min(img_w - 1, x_min))
        y_min = max(0, min(img_h - 1, y_min))
        x_max = max(0, min(img_w - 1, x_max))
        y_max = max(0, min(img_h - 1, y_max))

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        label_text = CLASS_NAMES[class_id]
        cv2.putText(
            img,
            label_text,
            (x_min, max(y_min - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        print(label_text)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Frame {frame_idx} - {os.path.basename(image_path)}")
        plt.show()


# %%
frame_idx = random.randint(1, 300)
#visualize_yolo_frame(f"/DataStorage/{YOLO_DATASET}/train", frame_idx)

# %%
EPOCHS = 50
BATCH = 16
IMGSZ = 1280
MODEL = "rtdetr-l.pt"  
DEVICE = 0

data_yaml = Path(os.getcwd()) / "yolo_kitti_dataset" / "dataset.yaml"
assert data_yaml.is_file(), f"Missing {data_yaml} — run prepare_yolo_dataset.py first"

rtdetr = RTDETR(MODEL)
results = rtdetr.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    device=DEVICE,
    project=f"{os.getcwd()}/runs",
    name=f"{MODEL[:-3]}_detect",
    exist_ok=True,
    pretrained=True,
    plots=True,
)

BEST_WEIGHTS = Path(results.save_dir) / "weights" / "best.pt"
print("Best weights:", BEST_WEIGHTS)
print("\nEvaluating best model...\n")

best_model = RTDETR(str(BEST_WEIGHTS))

metrics = best_model.val(
    data=str(data_yaml),
    imgsz=IMGSZ,
    batch=BATCH,
    plots=True,
    save_json=True,
    project="evaluation",
    name="rtdetr_eval",
)

print("=" * 60)
print("Overall Performance")
print("=" * 60)

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

print("=" * 60)

print("Per-class Performance")

names = best_model.names

for i, c in enumerate(metrics.box.ap_class_index):

    print(
        f"{names[c]:20s}"
        f" mAP50={metrics.box.ap50[i]:.4f}"
        f" mAP50-95={metrics.box.ap[i]:.4f}"
    )

# %%
# Training curves (if generated)
#from IPython.display import Image, display

#results_png = Path(results.save_dir) / "results.png"
#if results_png.is_file():
 #   display(Image(filename=str(results_png), width=700))

# %%
import json
import joblib
import random
import cv2
import matplotlib.pyplot as plt



def resize_and_write_image(img_path, dataset_name):
    resized_img = cv2.resize(cv2.imread(img_path), (960, 540))
    resized_img_name = f"/tmp/{dataset_name}_resized_img.png"
    cv2.imwrite(resized_img_name, resized_img)
    return Path(resized_img_name)

MODEL = "rtdetr-l.pt"  
BEST_WEIGHTS = Path(os.getcwd()) / "runs" / f"{MODEL[:-3]}_detect" / "weights" / "best.pt"
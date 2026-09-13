"""
Faster R-CNN (ResNet50-FPN v2) training on a YOLO-format KITTI dataset.

Expected layout:
    yolo_kitti_dataset/
        dataset.yaml        # same file used for the RT-DETR run (must have `names:`)
        train/images/*.jpg  train/labels/*.txt
        val/images/*.jpg    val/labels/*.txt
        test/images/*.jpg   test/labels/*.txt

Each label .txt line is standard YOLO format (normalized):
    class_id  cx  cy  w  h

pip install torch torchvision torchmetrics pyyaml --break-system-packages
"""

from pathlib import Path
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

DATASET_ROOT = Path("yolo_kitti_dataset")
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
PROJECT = Path("runs")
NAME = "fasterrcnn_r50_detect"
RUN_DIR = PROJECT / NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 8          # start here on a 3090 24GB; raise/lower based on nvidia-smi headroom
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
AMP = True
SEED = 0

torch.manual_seed(SEED)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class YoloKittiDataset(Dataset):
    """Reads a YOLO-format split (images/ + labels/) for torchvision detection models."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Path, split: str, transforms=None):
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"
        self.transforms = transforms
        self.img_paths = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(str(img_path))  # CHW, uint8
        if image.shape[0] == 1:  # grayscale safety
            image = image.expand(3, -1, -1)
        _, h, w = image.shape

        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                # clamp to image bounds
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # +1: torchvision reserves 0 for background

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # Wrap as tv_tensors so v2 transforms (e.g. SanitizeBoundingBoxes) can find and
        # operate on them correctly -- this also works for images with zero boxes.
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        image = tv_tensors.Image(image.float() / 255.0)
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def _labels_getter(inputs):
    # SanitizeBoundingBoxes calls this with the full (image, target) sample,
    # not the target dict directly -- unpack it first.
    _, target = inputs
    return target["labels"], target["area"], target["iscrowd"]


def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.SanitizeBoundingBoxes(labels_getter=_labels_getter),
    ])


def get_eval_transforms():
    return T.Compose([T.SanitizeBoundingBoxes(labels_getter=_labels_getter)])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes_with_bg: int):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=AMP):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    result = metric.compute()
    # torchmetrics' MeanAveragePrecision doesn't expose a single "precision" scalar
    # the way Ultralytics does (box.mp) -- mar_100 (mean average recall @ 100 dets/img)
    # is the closest standard COCO-style counterpart to "recall".
    return {
        "recall": float(result["mar_100"]),
        "map50": float(result["map_50"]),
        "map50_95": float(result["map"]),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    with open(DATASET_YAML) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]
    num_classes_with_bg = len(class_names) + 1

    train_ds = YoloKittiDataset(DATASET_ROOT, "train", get_train_transforms())
    val_ds = YoloKittiDataset(DATASET_ROOT, "val", get_eval_transforms())
    test_ds = YoloKittiDataset(DATASET_ROOT, "test", get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = build_model(num_classes_with_bg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_map = -1.0
    epochs_no_improve = 0
    best_path = RUN_DIR / "best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, scaler)
        scheduler.step()
        val_metrics = evaluate(model, val_loader)
        print(f"[epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} "
              f"val_map50={val_metrics['map50']:.4f} val_map50_95={val_metrics['map50_95']:.4f}")

        if val_metrics["map50_95"] > best_map:
            best_map = val_metrics["map50_95"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # Reload best weights for final evaluation
    model.load_state_dict(torch.load(best_path))
    val_final = evaluate(model, val_loader)
    test_final = evaluate(model, test_loader)

    summary = {
        "validation": val_final,
        "test": test_final,
    }
    with open(RUN_DIR / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))
    print("Training complete.")
    print("Best weights:", best_path)


if __name__ == "__main__":
    main()

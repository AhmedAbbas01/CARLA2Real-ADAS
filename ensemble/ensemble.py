import torch
import numpy as np

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from torchvision.models.detection import fasterrcnn_resnet50_fpn


############################################################
# Load Models
############################################################

# YOLOv8
yolo = YOLO("weights/yolov8_best.pt")

# RT-DETR
rtdetr = YOLO("weights/rtdetr_best.pt")

# Faster RCNN
faster = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
checkpoint = torch.load("weights/faster_rcnn_best.pth",
                        map_location="cpu")
faster.load_state_dict(checkpoint)
faster.eval()


############################################################
# Prediction Functions
############################################################

def predict_yolo(image):

    result = yolo.predict(image, verbose=False)[0]

    boxes = []
    scores = []
    labels = []
    distances = []

    h, w = image.shape[:2]

    for b in result.boxes:

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

        boxes.append([
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        ])

        scores.append(float(b.conf))
        labels.append(int(b.cls))

        # assumes distance regression exists
        distances.append(float(b.data[0][-1]))

    return boxes, scores, labels, distances


def predict_rtdetr(image):

    result = rtdetr.predict(image, verbose=False)[0]

    boxes = []
    scores = []
    labels = []
    distances = []

    h, w = image.shape[:2]

    for b in result.boxes:

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

        boxes.append([
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        ])

        scores.append(float(b.conf))
        labels.append(int(b.cls))
        distances.append(float(b.data[0][-1]))

    return boxes, scores, labels, distances


def predict_faster(image, transform):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = faster(img)[0]

    h, w = image.shape[:2]

    boxes = []
    scores = []
    labels = []
    distances = []

    for i in range(len(pred["boxes"])):

        x1, y1, x2, y2 = pred["boxes"][i].numpy()

        boxes.append([
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        ])

        scores.append(float(pred["scores"][i]))
        labels.append(int(pred["labels"][i]))

        distances.append(float(pred["distances"][i]))

    return boxes, scores, labels, distances


############################################################
# Ensemble
############################################################

def ensemble_prediction(image, transform):

    y_boxes, y_scores, y_labels, y_dist = predict_yolo(image)

    r_boxes, r_scores, r_labels, r_dist = predict_rtdetr(image)

    f_boxes, f_scores, f_labels, f_dist = predict_faster(image,
                                                         transform)

    boxes = [
        y_boxes,
        r_boxes,
        f_boxes
    ]

    scores = [
        y_scores,
        r_scores,
        f_scores
    ]

    labels = [
        y_labels,
        r_labels,
        f_labels
    ]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(

        boxes,
        scores,
        labels,

        weights=[2, 2, 1],      # YOLO and RT-DETR trusted more

        iou_thr=0.55,

        skip_box_thr=0.25
    )

    ##########################################################
    # Fuse Distances
    ##########################################################

    fused_distances = []

    all_boxes = y_boxes + r_boxes + f_boxes
    all_scores = y_scores + r_scores + f_scores
    all_labels = y_labels + r_labels + f_labels
    all_dist = y_dist + r_dist + f_dist

    for fb, fl in zip(fused_boxes, fused_labels):

        weighted_sum = 0
        total_weight = 0

        for box, score, label, dist in zip(
                all_boxes,
                all_scores,
                all_labels,
                all_dist):

            if label != fl:
                continue

            xx1 = max(box[0], fb[0])
            yy1 = max(box[1], fb[1])
            xx2 = min(box[2], fb[2])
            yy2 = min(box[3], fb[3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (fb[2] - fb[0]) * (fb[3] - fb[1])

            union = area1 + area2 - inter

            iou = inter / union if union > 0 else 0

            if iou > 0.5:

                weighted_sum += dist * score
                total_weight += score

        if total_weight == 0:
            fused_distances.append(None)
        else:
            fused_distances.append(weighted_sum / total_weight)

    return fused_boxes, fused_scores, fused_labels, fused_distances
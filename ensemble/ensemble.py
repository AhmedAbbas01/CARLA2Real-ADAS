import logging
import sys

# ---------------------------------------------------------
# Setup Logger
# ---------------------------------------------------------
def setup_logger(verbosity=logging.INFO):
    """
    Sets up the logger with the specified verbosity.

    :param verbosity: Logging level (e.g., logging.INFO, logging.DEBUG).
    :type verbosity: int
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    logger = logging.getLogger("ensemble_logger")
    logger.setLevel(verbosity)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(verbosity)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = setup_logger(logging.INFO)

# ---------------------------------------------------------
# Handle Imports
# ---------------------------------------------------------
try:
    import torch
    import numpy as np
    from ultralytics import YOLO, RTDETR
    from ensemble_boxes import weighted_boxes_fusion
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    logger.error(f"Missing import: {e}. Please install the required packages.")
    raise


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
def load_models(yolo_weights_path, rtdetr_weights_path, faster_rcnn_weights_path, depth_weights_path, num_classes, depth_encoder='vitl', max_depth=80):
    """
    Loads the YOLOv8, RT-DETR, Faster R-CNN, and DepthAnythingV2 models.

    :param yolo_weights_path: Path to the YOLOv8 weights file.
    :type yolo_weights_path: str
    :param rtdetr_weights_path: Path to the RT-DETR weights file.
    :type rtdetr_weights_path: str
    :param faster_rcnn_weights_path: Path to the Faster R-CNN weights file.
    :type faster_rcnn_weights_path: str
    :param depth_weights_path: Path to the DepthAnythingV2 metric depth weights file.
    :type depth_weights_path: str
    :param num_classes: Number of classes for Faster R-CNN initialization.
    :type num_classes: int
    :param depth_encoder: Encoder type for DepthAnythingV2 (e.g., 'vitl', 'vitb', 'vits').
    :type depth_encoder: str
    :param max_depth: Max depth parameter for the metric depth model (e.g., 80 for outdoor ADAS).
    :type max_depth: int
    :return: A tuple containing the loaded (yolo_model, rtdetr_model, faster_rcnn_model, depth_model).
    :rtype: tuple
    """
    try:
        logger.info(f"Loading YOLO model from {yolo_weights_path}...")
        yolo_model = YOLO(yolo_weights_path)

        logger.info(f"Loading RT-DETR model from {rtdetr_weights_path}...")
        rtdetr_model = RTDETR(rtdetr_weights_path)

        logger.info(f"Loading Faster R-CNN model from {faster_rcnn_weights_path} with {num_classes} classes...")
        faster_rcnn_model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        checkpoint = torch.load(faster_rcnn_weights_path, map_location="cpu")
        faster_rcnn_model.load_state_dict(checkpoint)
        faster_rcnn_model.eval()

        logger.info(f"Loading DepthAnythingV2 metric depth model from {depth_weights_path}...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        depth_model = DepthAnythingV2(**{**model_configs[depth_encoder], 'max_depth': max_depth})
        depth_model.load_state_dict(torch.load(depth_weights_path, map_location="cpu"))
        depth_model.eval()

        logger.info("Successfully loaded all object detection models.")
        return yolo_model, rtdetr_model, faster_rcnn_model, depth_model

    except Exception as e:
        logger.error(f"An error occurred while loading the object detection models: {e}")
        raise


# ---------------------------------------------------------
# Prediction Functions
# ---------------------------------------------------------
def predict_yolo(model, image):
    """
    Runs inference using the YOLO model and extracts bounding boxes, scores, and labels.

    :param model: The loaded YOLO model.
    :type model: ultralytics.YOLO
    :param image: Input image array.
    :type image: np.ndarray
    :return: Tuple containing (boxes, scores, labels).
    :rtype: tuple(list, list, list)
    """
    logger.debug("Running YOLO inference...")
    try:
        result = model.predict(image, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO prediction failed: {e}")
        return [], [], []

    boxes, scores, labels = [], [], []
    h, w = image.shape[:2]

    for b in result.boxes:
        try:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(float(b.conf))
            labels.append(int(b.cls))
        except Exception as e:
            logger.error(f"Unexpected error while parsing YOLO boxes: {e}")

    return boxes, scores, labels


def predict_rtdetr(model, image):
    """
    Runs inference using the RT-DETR model and extracts bounding boxes, scores, and labels.

    :param model: The loaded RT-DETR model.
    :type model: ultralytics.RTDETR
    :param image: Input image array.
    :type image: np.ndarray
    :return: Tuple containing (boxes, scores, labels).
    :rtype: tuple(list, list, list)
    """
    logger.debug("Running RT-DETR inference...")
    try:
        result = model.predict(image, verbose=False)[0]
    except Exception as e:
        logger.error(f"RT-DETR prediction failed: {e}")
        return [], [], []

    boxes, scores, labels = [], [], []
    h, w = image.shape[:2]

    for b in result.boxes:
        try:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(float(b.conf))
            labels.append(int(b.cls))
        except Exception as e:
            logger.error(f"Unexpected error while parsing RT-DETR boxes: {e}")

    return boxes, scores, labels


def predict_faster(model, image, transform):
    """
    Runs inference using the Faster R-CNN model and extracts bounding boxes, scores, and labels.

    :param model: The loaded Faster R-CNN model.
    :type model: torch.nn.Module
    :param image: Input image array.
    :type image: np.ndarray
    :param transform: Transformation function to apply to the image.
    :type transform: callable
    :return: Tuple containing (boxes, scores, labels).
    :rtype: tuple(list, list, list)
    """
    logger.debug("Running Faster R-CNN inference...")
    try:
        img = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = model(img)[0]
    except Exception as e:
        logger.error(f"Faster R-CNN prediction failed: {e}")
        return [], [], []

    h, w = image.shape[:2]
    boxes, scores, labels = [], [], []

    for i in range(len(pred["boxes"])):
        try:
            x1, y1, x2, y2 = pred["boxes"][i].cpu().numpy()
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(float(pred["scores"][i]))
            labels.append(int(pred["labels"][i]))
        except Exception as e:
            logger.error(f"Unexpected error while parsing Faster R-CNN boxes: {e}")

    return boxes, scores, labels


def predict_depth(model, image):
    """
    Predicts the metric depth map using DepthAnythingV2.

    :param model: The loaded DepthAnythingV2 model.
    :type model: DepthAnythingV2
    :param image: Input image array (H, W, 3).
    :type image: np.ndarray
    :return: Depth map array (H, W).
    :rtype: np.ndarray
    """
    logger.debug("Running DepthAnythingV2 inference...")
    try:
        depth_map = model.infer_image(image)
        return depth_map
    except Exception as e:
        logger.error(f"Depth prediction failed: {e}")
        return None


# ---------------------------------------------------------
# Ensemble
# ---------------------------------------------------------
def ensemble_prediction(yolo_model, rtdetr_model, faster_model, depth_model, image, transform):
    """
    Fuses predictions from YOLO, RT-DETR, and Faster R-CNN using Weighted Boxes Fusion (WBF).
    Extracts representative metric depth per fused bounding box using DepthAnythingV2.

    :param yolo_model: The loaded YOLO model.
    :type yolo_model: ultralytics.YOLO
    :param rtdetr_model: The loaded RT-DETR model.
    :type rtdetr_model: ultralytics.YOLO
    :param faster_model: The loaded Faster R-CNN model.
    :type faster_model: torch.nn.Module
    :param depth_model: The loaded DepthAnythingV2 model.
    :type depth_model: DepthAnythingV2
    :param image: Input image array.
    :type image: np.ndarray
    :param transform: Transformation function for Faster R-CNN.
    :type transform: callable
    :return: Tuple containing fused (boxes, scores, labels, distances).
    :rtype: tuple(list, list, list, list)
    """
    logger.info("Executing ensemble predictions...")

    y_boxes, y_scores, y_labels = predict_yolo(yolo_model, image)
    r_boxes, r_scores, r_labels = predict_rtdetr(rtdetr_model, image)
    f_boxes, f_scores, f_labels = predict_faster(faster_model, image, transform)

    boxes = [y_boxes, r_boxes, f_boxes]
    scores = [y_scores, r_scores, f_scores]
    labels = [y_labels, r_labels, f_labels]

    try:
        logger.debug("Fusing bounding boxes using weighted_boxes_fusion...")
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=[2, 2, 1],  # YOLO and RT-DETR trusted more
            iou_thr=0.55,
            skip_box_thr=0.25
        )
    except Exception as e:
        logger.error(f"Weighted boxes fusion failed: {e}")
        return [], [], [], []

    # ---------------------------------------------------------
    # Predict & Assign Distances
    # ---------------------------------------------------------
    fused_distances = []
    depth_map = predict_depth(depth_model, image)
    h, w = image.shape[:2]

    if depth_map is not None:
        logger.debug("Extracting distances from depth map...")
        for box in fused_boxes:
            # box coordinates are normalized [0, 1]
            x1, y1, x2, y2 = box
            ix1, iy1 = int(x1 * w), int(y1 * h)
            ix2, iy2 = int(x2 * w), int(y2 * h)

            # Clamp to image boundaries
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(w - 1, ix2), min(h - 1, iy2)

            if ix1 >= ix2 or iy1 >= iy2:
                fused_distances.append(None)
                continue

            # Extract region of interest from depth map and use median depth
            roi_depth = depth_map[iy1:iy2, ix1:ix2]
            fused_distances.append(float(np.median(roi_depth)))
    else:
        logger.warning("Depth map generation failed. Returning None for distances.")
        fused_distances = [None] * len(fused_boxes)

    logger.info(f"Ensemble completed successfully. Returning {len(fused_boxes)} fused boxes.")
    return fused_boxes, fused_scores, fused_labels, fused_distances
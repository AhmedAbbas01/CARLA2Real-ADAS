import sys
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import logging

logger = logging.getLogger("Perception")

try:
    from ensemble import load_models, ensemble_prediction
except ImportError as e:
    logger.warning(f"Could not import ensemble logic. Ensure ensemble.py exists. Error: {e}")

try:
    from ultralytics import YOLO
except ImportError:
    pass

class EnsemblePerception:
    """Perception layer leveraging YOLO, RT-DETR, Faster R-CNN, and DepthAnythingV2 (Ensemble)."""
    def __init__(self, yolo_path, rtdetr_path, faster_path, depth_path, num_classes=91, depth_encoder='vits', max_depth=20):
        self.yolo_path = yolo_path
        self.rtdetr_path = rtdetr_path
        self.faster_path = faster_path
        self.depth_path = depth_path
        self.num_classes = num_classes
        self.depth_encoder = depth_encoder
        self.max_depth = max_depth
        
        self.transform = T.Compose([T.ToTensor()])
        self.models_loaded = False
        self.class_names = {}

    def load_models(self):
        logger.info("Loading models for Ensemble Prediction...")
        self.yolo, self.rtdetr, self.faster, self.depth = load_models(
            self.yolo_path, self.rtdetr_path, self.faster_path, self.depth_path,
            self.num_classes, self.depth_encoder, self.max_depth
        )
        self.class_names = self.yolo.names
        self.models_loaded = True

    def predict(self, frame):
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # ensemble_prediction returns normalized [0, 1] boxes
        boxes, scores, labels, distances = ensemble_prediction(
            self.yolo, self.rtdetr, self.faster, self.depth, frame, self.transform
        )
        
        h, w = frame.shape[:2]
        abs_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            abs_boxes.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])
            
        return abs_boxes, scores, labels, distances
        
    def get_class_name(self, cls_id):
        return self.class_names.get(int(cls_id), str(cls_id))


class SinglePerception:
    """Perception layer leveraging a single YOLO model and DepthAnythingV2."""
    def __init__(self, yolo_path, depth_path, depth_encoder='vits', max_depth=20, conf_threshold=0.40):
        self.yolo_path = yolo_path
        self.depth_path = depth_path
        self.depth_encoder = depth_encoder
        self.max_depth = max_depth
        self.conf_threshold = conf_threshold
        self.models_loaded = False
        self.class_names = {}

    def load_models(self):
        logger.info(f"Loading YOLO model from {self.yolo_path} ...")
        self.yolo = YOLO(self.yolo_path)
        self.class_names = self.yolo.names
        
        # Load DepthAnythingV2 dynamically mapping dependencies 
        import sys
        from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        
        logger.info(f"Loading DepthAnythingV2 from {self.depth_path} ...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.depth = DepthAnythingV2(**{**model_configs[self.depth_encoder], 'max_depth': self.max_depth})
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth.load_state_dict(torch.load(self.depth_path, map_location=device))
        self.depth = self.depth.to(device).eval()
        
        self.models_loaded = True

    def predict(self, frame):
        """
        Runs inference using the YOLO model and DepthAnythingV2 to extract bounding boxes,
        scores, labels, and representative metric depth per bounding box.

        :param frame: Input image array (H, W, 3).
        :type frame: np.ndarray
        :return: Tuple containing (boxes, scores, labels, distances).
        :rtype: tuple(list, list, list, list)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        logger.debug("Running SinglePerception (YOLO + Depth) inference...")
        
        boxes, scores, labels, distances = [], [], [], []
        h, w = frame.shape[:2]

        # YOLO Prediction
        try:
            results = self.yolo.predict(frame, conf=self.conf_threshold, verbose=False)
            yolo_result = results[0]
        except Exception as e:
            logger.error(f"YOLO prediction failed in SinglePerception: {e}")
            return [], [], [], []

        # Depth Prediction
        depth_map = None
        try:
            depth_map = self.depth.infer_image(frame)
        except Exception as e:
            logger.error(f"Depth prediction failed in SinglePerception: {e}")
            logger.warning("Returning None for distances due to depth prediction failure.")

        for b in yolo_result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(b.conf))
            labels.append(int(b.cls))

            if depth_map is not None:
                # Extract region of interest from depth map and use median depth
                roi_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]
                if roi_depth.size > 0:
                    distances.append(float(np.median(roi_depth)))
                else:
                    distances.append(None)
            else:
                distances.append(None)

        return boxes, scores, labels, distances

    def get_class_name(self, cls_id):
        return self.class_names.get(int(cls_id), str(cls_id))
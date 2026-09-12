import numpy as np
import pandas as pd
import json
import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("DistanceEvaluator")

class DistanceEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.records = []
        self.running_abs_error = 0.0
        self.running_matches = 0

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Computes IoU between two 2D bounding boxes [xmin, ymin, xmax, ymax]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def add_frame_predictions(self, gt_objects: List[Dict], pred_objects: List[Dict]) -> Dict:
        """
        gt_objects: [{'bbox': [x1,y1,x2,y2], 'class': 'Car', 'distance': 12.4}, ...]
        pred_objects: [{'bbox': [x1,y1,x2,y2], 'class': 'Car', 'distance': 12.1}, ...]
        """
        matched_gt = set()
        frame_abs_error = 0.0
        frame_matches = 0

        for pred in pred_objects:
            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_objects):
                if idx in matched_gt or gt['class'] != pred['class']:
                    continue
                iou = self.compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                gt_match = gt_objects[best_gt_idx]
                matched_gt.add(best_gt_idx)
                
                gt_dist = gt_match['distance']
                pred_dist = pred['distance']
                err = pred_dist - gt_dist
                abs_err = abs(err)

                # Classify into distance range bins
                if gt_dist <= 10.0:
                    dist_range = "0-10m (Near)"
                elif gt_dist <= 30.0:
                    dist_range = "10-30m (Mid)"
                else:
                    dist_range = "30m+ (Far)"

                self.records.append({
                    'class': pred['class'],
                    'gt_distance': gt_dist,
                    'pred_distance': pred_dist,
                    'abs_error': abs_err,
                    'sq_error': err ** 2,
                    'rel_error': abs_err / gt_dist if gt_dist > 0 else 0,
                    'dist_range': dist_range
                })
                
                self.running_abs_error += abs_err
                self.running_matches += 1
                frame_abs_error += abs_err
                frame_matches += 1
            
        if frame_matches > 0:
            return {
                'frame_mae': frame_abs_error / frame_matches,
                'running_mae': self.running_abs_error / self.running_matches,
                'frame_matches': frame_matches
            }
        return None

    def evaluate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(self.records)
        if df.empty:
            logger.warning("No matching ground-truth and prediction pairs found!")
            return df, df

        # Overall summary
        overall_mae = df['abs_error'].mean()
        overall_rmse = np.sqrt(df['sq_error'].mean())
        overall_rel = df['rel_error'].mean()

        logger.info("=== OVERALL DISTANCE EVALUATION ===")
        logger.info(f"Total Matched Objects: {len(df)}")
        logger.info(f"MAE:  {overall_mae:.3f} m")
        logger.info(f"RMSE: {overall_rmse:.3f} m")
        logger.info(f"Rel:  {overall_rel * 100:.2f} %\n")

        # Per-Class Summary Table
        class_summary = df.groupby('class').agg(
            Count=('abs_error', 'count'),
            MAE=('abs_error', 'mean'),
            RMSE=('sq_error', lambda x: np.sqrt(np.mean(x))),
            Rel_Error=('rel_error', 'mean')
        ).reset_index()

        # Per-Range Summary Table
        range_summary = df.groupby('dist_range').agg(
            Count=('abs_error', 'count'),
            MAE=('abs_error', 'mean'),
            RMSE=('sq_error', lambda x: np.sqrt(np.mean(x))),
            Rel_Error=('rel_error', 'mean')
        ).reset_index()

        logger.info("=== PER-CLASS DISTANCE ACCURACY ===")
        logger.info("\n" + class_summary.to_string(index=False))
        
        logger.info("=== PER-RANGE DISTANCE ACCURACY ===")
        logger.info("\n" + range_summary.to_string(index=False))

        return class_summary, range_summary

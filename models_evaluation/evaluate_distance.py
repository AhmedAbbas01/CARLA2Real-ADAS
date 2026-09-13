import numpy as np
import pandas as pd
import json
import os
import logging
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("DistanceEvaluator")

class DistanceEvaluator:
    """!
    @brief A class to evaluate distance estimation accuracy for detected objects.
    
    This class matches ground-truth and predicted objects using Intersection over Union (IoU) 
    and computes distance error metrics such as MAE, RMSE, and Relative Error.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """!
        @brief Initializes the DistanceEvaluator.
        
        @param iou_threshold The minimum IoU required to match a prediction to a ground-truth object.
        """
        self.iou_threshold = iou_threshold
        self.records = []
        self.running_abs_error = 0.0
        self.running_matches = 0

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """!
        @brief Computes Intersection over Union (IoU) between two 2D bounding boxes.
        
        @param box1 The first bounding box [xmin, ymin, xmax, ymax].
        @param box2 The second bounding box [xmin, ymin, xmax, ymax].
        @return The IoU score (float) between the two bounding boxes.
        """
        # Determine the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute the area of intersection rectangle
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        # Compute the area of both bounding boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # Compute the union
        union = area1 + area2 - intersection

        # Return the IoU score
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_center_distance(box1: List[float], box2: List[float]) -> float:
        """!
        @brief Computes the normalized Euclidean distance between the centers of two bounding boxes.
        
        @param box1 The first bounding box [xmin, ymin, xmax, ymax].
        @param box2 The second bounding box [xmin, ymin, xmax, ymax].
        @return The normalized distance between the centers (float).
        """
        cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
        cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
        
        center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        
        # Diagonal of the enclosing bounding box
        diag = np.sqrt((max(box1[2], box2[2]) - min(box1[0], box2[0])) ** 2 + 
                       (max(box1[3], box2[3]) - min(box1[1], box2[1])) ** 2)
        
        return center_dist / diag if diag > 0 else 0.0

    def add_frame_predictions(self, gt_objects: List[Dict], pred_objects: List[Dict], frame_id: int = None, log_dir: str = "evaluation_logs") -> Dict:
        """!
        @brief Matches predicted objects to ground-truth objects for a single frame and records the distance errors.
        
        @param gt_objects A list of ground-truth object dictionaries: [{'box': [x1,y1,x2,y2], 'class': 'Car', 'distance': 12.4}, ...]
        @param pred_objects A list of predicted object dictionaries: [{'box': [x1,y1,x2,y2], 'class': 'Car', 'distance': 12.1}, ...]
        @param frame_id An optional frame identifier to save the frame results to a JSON file.
        @param log_dir The directory where JSON logs should be saved if frame_id is provided.
        @return A dictionary containing the frame MAE, running MAE, and the number of matches in the frame, or None if no matches found.
        """
        frame_abs_error = 0.0
        frame_matches = 0

        if len(gt_objects) > 0 and len(pred_objects) > 0:
            # Create a cost matrix for the Hungarian algorithm
            # Initialize with a high cost to penalize invalid matches
            cost_matrix = np.full((len(pred_objects), len(gt_objects)), 1e6)

            for i, pred in enumerate(pred_objects):
                for j, gt in enumerate(gt_objects):
                    if pred['class'] == gt['class']:
                        iou = self.compute_iou(pred['box'], gt['box'])
                        if iou >= self.iou_threshold:
                            # Incorporate both IoU and normalized center distance into the cost
                            norm_center_dist = self.compute_center_distance(pred['box'], gt['box'])
                            cost_matrix[i, j] = 0.8 * (1.0 - iou) + 0.2 * norm_center_dist

            # Apply the Hungarian algorithm (linear sum assignment)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= 1.0:  # Check if it's a valid match (IoU >= threshold)
                    pred = pred_objects[i]
                    gt_match = gt_objects[j]

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

                    # Record the matching details
                    self.records.append({
                        'class': pred['class'],
                        'gt_distance': gt_dist,
                        'pred_distance': pred_dist,
                        'abs_error': abs_err,
                        'sq_error': err ** 2,
                        'rel_error': abs_err / gt_dist if gt_dist > 0 else 0,
                        'dist_range': dist_range
                    })
                    
                    # Update running totals
                    self.running_abs_error += abs_err
                    self.running_matches += 1
                    frame_abs_error += abs_err
                    frame_matches += 1
            
        metrics = None
        if frame_matches > 0:
            metrics = {
                'frame_mae': frame_abs_error / frame_matches,
                'running_mae': self.running_abs_error / self.running_matches,
                'frame_matches': frame_matches
            }

        # Save to JSON log if frame_id is provided
        if frame_id is not None:
            log_data = {
                "frame_id": frame_id,
                "ground_truth": gt_objects,
                "predictions": pred_objects,
                "evaluation_results": metrics
            }
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"frame_{frame_id:04d}.json"), "w") as f:
                json.dump(log_data, f, indent=4)

        return metrics

    def evaluate(self, html_report_path: str = "distance_evaluation_report.html") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """!
        @brief Evaluates all recorded distance predictions, logs the summaries, and generates an HTML report.
        
        @param html_report_path The file path where the HTML report will be saved.
        @return A tuple containing the class summary DataFrame and range summary DataFrame.
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            logger.warning("No matching ground-truth and prediction pairs found!")
            return df, df

        # Overall summary
        overall_mae = df['abs_error'].mean()
        overall_rmse = np.sqrt(df['sq_error'].mean())
        overall_rel = df['rel_error'].mean()

        # Log overall summary
        logger.debug("=== OVERALL DISTANCE EVALUATION ===")
        logger.debug(f"Total Matched Objects: {len(df)}")
        logger.debug(f"MAE:  {overall_mae:.3f} m")
        logger.debug(f"RMSE: {overall_rmse:.3f} m")
        logger.debug(f"Rel:  {overall_rel * 100:.2f} %\n")

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

        # Log per-class and per-range summaries
        logger.debug("=== PER-CLASS DISTANCE ACCURACY ===")
        logger.debug("\n" + class_summary.to_string(index=False))
        
        logger.debug("=== PER-RANGE DISTANCE ACCURACY ===")
        logger.debug("\n" + range_summary.to_string(index=False))

        # Generate HTML report
        try:
            html_content = (
                "<html>\n<head>\n<title>Distance Evaluation Report</title>\n"
                "<style>\n"
                "  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }\n"
                "  h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }\n"
                "  h2 { color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }\n"
                "  .container { background-color: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.08); max-width: 960px; margin: auto; }\n"
                "  .summary-list { list-style-type: none; padding: 0; display: flex; justify-content: space-between; gap: 20px; }\n"
                "  .summary-list li { background: #ecf0f1; flex: 1; padding: 20px; border-radius: 8px; text-align: center; border-bottom: 4px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }\n"
                "  .summary-list li span { display: block; font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }\n"
                "  .summary-list li b { font-size: 1.5em; color: #2c3e50; }\n"
                "  table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 1em; }\n"
                "  th, td { border: 1px solid #ecf0f1; padding: 14px 16px; text-align: center; }\n"
                "  th { background-color: #34495e; color: #ffffff; font-weight: 600; text-transform: uppercase; font-size: 0.9em; }\n"
                "  tr:nth-of-type(even) { background-color: #f9fbfb; }\n"
                "  tr:hover { background-color: #f1f5f6; }\n"
                "</style>\n"
                "</head>\n<body>\n"
                "<div class='container'>\n"
                "  <h1>Distance Evaluation Report</h1>\n"
                "  <h2>Overall Distance Evaluation</h2>\n"
                "  <ul class='summary-list'>\n"
                f"    <li><span>Total Matched Objects</span><b>{len(df)}</b></li>\n"
                f"    <li><span>MAE</span><b>{overall_mae:.3f} m</b></li>\n"
                f"    <li><span>RMSE</span><b>{overall_rmse:.3f} m</b></li>\n"
                f"    <li><span>Relative Error</span><b>{overall_rel * 100:.2f} %</b></li>\n"
                "  </ul>\n"
                "  <h2>Per-Class Distance Accuracy</h2>\n"
                f"  {class_summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')}\n"
                "  <h2>Per-Range Distance Accuracy</h2>\n"
                f"  {range_summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')}\n"
                "</div>\n"
                "</body>\n</html>"
            )
            
            with open(html_report_path, "w") as f:
                f.write(html_content)
                
            logger.info(f"HTML report successfully generated at: {html_report_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")

        return class_summary, range_summary

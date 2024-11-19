import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

# Fix seeds
SEED = 42
np.random.seed(SEED)

# [ Parameters ]
PUBLIC_GT_CSV_PATH = 'ground_truth.csv'
PUBLIC_PREDICTED_CSV = 'inference_results.csv'

COLUMNS = ['image_id', 'label', 'xc', 'yc', 'w', 'h', 'w_img', 'h_img', 'score', 'time_spent']

# Validate image dimensions between ground truth and predictions
def validate_image_dimensions(gt_df: pd.DataFrame, predicted_df: pd.DataFrame):
    merged = pd.merge(
        gt_df[['w_img', 'h_img']],
        predicted_df[['w_img', 'h_img']],
        left_index=True,
        right_index=True,
        suffixes=('_gt', '_pred')
    )
    mismatched = merged[(merged['w_img_gt'] != merged['w_img_pred']) | (merged['h_img_gt'] != merged['h_img_pred'])]
    if not mismatched.empty:
        print("Mismatched image dimensions:")
        print(mismatched)
        raise ValueError(f"Image dimension mismatch for image_ids: {mismatched.index.tolist()}")

# Align predicted dimensions to ground truth
def align_image_dimensions(gt_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
    for col in ['w_img', 'h_img']:
        if col in predicted_df.columns:
            predicted_df[col] = gt_df[col].reindex(predicted_df.index).fillna(predicted_df[col])
    return predicted_df

# Add masks to DataFrame
def add_masks(df: pd.DataFrame) -> pd.DataFrame:
    try:
        masks = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating masks"):
            xc, yc, w, h = row['xc'], row['yc'], row['w'], row['h']
            w_img = int(row['w_img']) if row['w_img'] > 0 else 1920
            h_img = int(row['h_img']) if row['h_img'] > 0 else 1080
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h / 2) * h_img)
            mask[max(y1, 0):min(y2, h_img), max(x1, 0):min(x2, w_img)] = 1
            masks.append(mask)
        df['mask'] = masks
    except Exception as error:
        raise Exception(f"Error generating masks: {error}")
    return df

# Optimized IoU computation using jit
@jit(nopython=True)
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

# Optimized image processing using jit
@jit(nopython=True)
def process_image(pred_masks: np.ndarray, gt_masks: np.ndarray, thresholds: np.ndarray) -> dict:
    results = {}
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)

    if num_pred == 0 and num_gt == 0:
        for t in thresholds:
            results[t] = {'tp': 0, 'fp': 0, 'fn': 0}
        return results
    elif num_pred == 0:
        for t in thresholds:
            results[t] = {'tp': 0, 'fp': 0, 'fn': num_gt}
        return results
    elif num_gt == 0:
        for t in thresholds:
            results[t] = {'tp': 0, 'fp': num_pred, 'fn': 0}
        return results

    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = compute_iou(pred_masks[i], gt_masks[j])

    for t in thresholds:
        tp, fp, fn = 0, num_pred, num_gt
        for i in range(num_pred):
            for j in range(num_gt):
                if iou_matrix[i, j] >= t:
                    tp += 1
                    fp -= 1
                    fn -= 1
        results[t] = {'tp': tp, 'fp': fp, 'fn': fn}
    return results

# Compute metric
def compute_overall_metric(predicted_df: pd.DataFrame, gt_df: pd.DataFrame, thresholds: np.ndarray) -> float:
    gt_df = add_masks(gt_df)
    predicted_df = add_masks(predicted_df)
    unique_image_ids = list(set(predicted_df.index) | set(gt_df.index))
    
    total_tp = {t: 0 for t in thresholds}
    total_fp = {t: 0 for t in thresholds}
    total_fn = {t: 0 for t in thresholds}

    for image_id in tqdm(unique_image_ids, desc="Processing images"):
        pred_masks = np.array(predicted_df.loc[image_id]['mask'].tolist()) if image_id in predicted_df.index else np.empty((0,))
        gt_masks = np.array(gt_df.loc[image_id]['mask'].tolist()) if image_id in gt_df.index else np.empty((0,))

        # Debugging: Handle cases with no masks
        if pred_masks.size == 0 and gt_masks.size == 0:
            print(f"Warning: No masks for image_id {image_id}. Skipping.")
            continue

        results = process_image(pred_masks, gt_masks, thresholds)
        for t in thresholds:
            total_tp[t] += results[t]['tp']
            total_fp[t] += results[t]['fp']
            total_fn[t] += results[t]['fn']

    metric = sum((1 + t) * total_tp[t] / (1 + t * total_fp[t] + t * total_fn[t]) for t in thresholds) / len(thresholds)
    return round(metric, 4)

# Main script
if __name__ == "__main__":
    gt_df = pd.read_csv(PUBLIC_GT_CSV_PATH)
    predicted_df = pd.read_csv(PUBLIC_PREDICTED_CSV)

    # Ensure consistent dimensions
    predicted_df = align_image_dimensions(gt_df, predicted_df)

    # Validate dimensions
    validate_image_dimensions(gt_df, predicted_df)

    # Filter image IDs
    predicted_df = predicted_df[predicted_df['image_id'].isin(gt_df['image_id'])]

    thresholds = np.array([0.3, 0.5, 0.7])
    try:
        metric = compute_overall_metric(predicted_df, gt_df, thresholds)
        print(f"Calculated metric: {metric}")
    except Exception as e:
        print(f"Error during metric calculation: {e}")

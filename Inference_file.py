import os
import pandas as pd
from ultralytics import YOLO
import cv2
from PIL import Image as Im

# Paths
path_val_imgs = '/home/dima/data/validation/images'
path_model = '/home/dima/yolo_bigger/training/full_dataset/train10/weights/best.pt'
output_csv = './inference_results.csv'

# Initialize YOLO model
model = YOLO(path_model)

# Prepare a list to store prediction results
results_list = []

# Iterate through validation images
for filename in os.listdir(path_val_imgs):
    image_path = os.path.join(path_val_imgs, filename)

    # Perform prediction
    results = model.predict(source=image_path, conf=0.3)
    detections = results[0].boxes  # Get bounding boxes
    
    if detections is not None:
        # Loop through detections and collect data
        for box in detections:
            # Extract bounding box data
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]  # Class label
            
            # Image information
            img = cv2.imread(image_path)
            h_img, w_img, _ = img.shape
            xc = ((x_min + x_max) / 2) / w_img  # Normalized center x
            yc = ((y_min + y_max) / 2) / h_img  # Normalized center y
            w = (x_max - x_min) / w_img  # Normalized width
            h = (y_max - y_min) / h_img  # Normalized height

            # Append result to the list
            results_list.append({
                'image_id': filename,
                'label': int(cls),  # Convert class to integer
                'xc': float(xc),
                'yc': float(yc),
                'w': float(w),
                'h': float(h),
                'w_img': float(w_img),
                'h_img': float(h_img),
                'score': float(conf),
                'time_spent': results[0].speed['inference'] / 1000.0  # Convert ms to seconds
            })

# Save all results into a CSV file
df = pd.DataFrame(results_list)
df.to_csv(output_csv, index=False)

print(f"Inference results saved to {output_csv}")

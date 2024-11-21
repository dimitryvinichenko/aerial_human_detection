# plotting gt and prediction

import os
import pandas as pd

# Define the paths
path_gt_dir = '/home/dima/data/validation/labels'  # Directory containing ground truth TXT files
output_csv = './ground_truth.csv'

# Prepare a list to store ground truth data
gt_data = []

# Iterate through all TXT files in the directory
for filename in os.listdir(path_gt_dir):
    if filename.endswith('.txt'):  # Ensure only TXT files are processed
        file_path = os.path.join(path_gt_dir, filename)
        image_id = os.path.splitext(filename)[0]  # Use the filename without extension as image_id

        # Read the content of the TXT file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # Ensure the line has the correct number of elements
                    label = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Append data to the list
                    gt_data.append({
                        'image_id': image_id + '.jpg',
                        'label': label,
                        'xc': xc,
                        'yc': yc,
                        'w': w,
                        'h': h,
                        'w_img': 0,  # Placeholder, as image dimensions may not be known
                        'h_img': 0,  # Placeholder, as image dimensions may not be known
                        'score': None,  # No score for ground truth
                        'time_spent': 0  # Placeholder time spent
                    })

# Convert the list to a DataFrame
gt_df = pd.DataFrame(gt_data)

# Save the DataFrame to a CSV file
gt_df.to_csv(output_csv, index=False)

print(f"Ground truth CSV saved to {output_csv}")

import os

path_val_imgs = '/home/dima/data/validation/images'
path_val_annotations = '/home/dima/data/validation/labels'

path_model = '/home/dima/yolo_bigger/training/full_dataset/train10/weights/best.pt'

model = YOLO(path_model)

for filename in os.listdir(path_val_imgs):

    image = os.path.join(path_val_imgs, filename)
    results = model.predict(source=image, conf=0.3)
    data = results[0].plot()  # This plots the detections on the image
    img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    img = Im.fromarray(img_rgb, 'RGB')
    img.save('./inferrence/' + filename)




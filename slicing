import cv2
import os

def split_img(image_path, label_path, output_dir):
    try:
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        height, width = image.shape[:2]
        print(f"Image dimensions (HxW): {height}x{width}")

        if width <= 3000:
            rows, cols = 1, 1
        elif 4000 < width <= 6000:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 3

        print(f"Splitting into {rows} rows and {cols} columns")

        crop_h, crop_w = height // rows, width // cols

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Invalid label format in {label_path}: {line}")
                        continue
                    class_id, x_center, y_center, w, h = parts
                    labels.append({
                        'class_id': int(class_id),
                        'x_center': float(x_center),
                        'y_center': float(y_center),
                        'width': float(w),
                        'height': float(h)
                    })
        else:
            print(f"No label file found for {image_path}.")
            labels = []

        for row in range(rows):
            for col in range(cols):
                x_start, x_end = col * crop_w, (col + 1) * crop_w
                y_start, y_end = row * crop_h, (row + 1) * crop_h

                cropped_image = image[y_start:y_end, x_start:x_end]

                image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{row}_{col}.jpg"
                cropped_image_path = os.path.join(images_dir, image_name)
                cv2.imwrite(cropped_image_path, cropped_image)

                cropped_labels = []
                for label in labels:
                    x_center_pixel = label['x_center'] * width
                    y_center_pixel = label['y_center'] * height
                    box_width_pixel = label['width'] * width
                    box_height_pixel = label['height'] * height

                    x_min = x_center_pixel - box_width_pixel / 2
                    x_max = x_center_pixel + box_width_pixel / 2
                    y_min = y_center_pixel - box_height_pixel / 2
                    y_max = y_center_pixel + box_height_pixel / 2

                    if x_max <= x_start or x_min >= x_end or y_max <= y_start or y_min >= y_end:
                        continue 

                    new_x_min = max(x_min, x_start) - x_start
                    new_x_max = min(x_max, x_end) - x_start
                    new_y_min = max(y_min, y_start) - y_start
                    new_y_max = min(y_max, y_end) - y_start

                    new_box_width = new_x_max - new_x_min
                    new_box_height = new_y_max - new_y_min

                    new_x_center = new_x_min + new_box_width / 2
                    new_y_center = new_y_min + new_box_height / 2

                    x_center_normalized = new_x_center / crop_w
                    y_center_normalized = new_y_center / crop_h
                    width_normalized = new_box_width / crop_w
                    height_normalized = new_box_height / crop_h

                    x_center_normalized = max(min(x_center_normalized, 1.0), 0.0)
                    y_center_normalized = max(min(y_center_normalized, 1.0), 0.0)
                    width_normalized = max(min(width_normalized, 1.0), 0.0)
                    height_normalized = max(min(height_normalized, 1.0), 0.0)

                    cropped_labels.append(f"{label['class_id']} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}")

                label_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{row}_{col}.txt"
                label_path_out = os.path.join(labels_dir, label_name)
                with open(label_path_out, "w") as out_label_file:
                    for lbl in cropped_labels:
                        out_label_file.write(lbl + "\n")

        print(f"Processed {image_path} into {rows * cols} parts.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")


image_dataset_dir = '/home/dima/data/validation/images'
label_dataset_dir = '/home/dima/data/validation/labels'
output_dataset_dir = '/home/dima/data_splited/train' 

image_files = [f for f in os.listdir(image_dataset_dir) if f.endswith(('.JPG', '.jpg'))]

for image_file in image_files:
    image_path = os.path.join(image_dataset_dir, image_file)
    label_path = os.path.join(label_dataset_dir, os.path.splitext(image_file)[0] + ".txt")
    split_img(image_path, label_path, output_dataset_dir)


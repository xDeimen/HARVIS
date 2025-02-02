import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = "CarEngineBayDataset"  # Adjust this if needed
image_dir = os.path.join(data_dir, "images/train")
label_dir = os.path.join(data_dir, "labels/train")

def load_yolo_labels(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                data = list(map(float, line.strip().split()))
                class_id, x_center, y_center, width, height = data
                boxes.append((x_center, y_center, width, height, class_id))
    return boxes

def draw_boxes(image, boxes):
    h, w, _ = image.shape
    for x_center, y_center, width, height, class_id in boxes:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def preview_dataset(num_samples=5):
    image_files = sorted(os.listdir(image_dir))[:num_samples]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for ax, img_file in zip(axes, image_files):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = load_yolo_labels(label_path)
        image = draw_boxes(image, boxes)
        
        ax.imshow(image)
        ax.set_title(img_file)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_dataset(num_samples=4)

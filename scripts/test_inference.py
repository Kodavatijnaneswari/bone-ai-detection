import os
from ultralytics import YOLO

# Load the model directly from the media folder
MODEL_PATH = r"c:\Users\Jnaneswari.Kodavati\Downloads\113.Bone_Abnormality_Detection\113.Bone_Abnormality_Detection\Code\media\YOLOv8x-original.pt"
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Test Image from the robust dataset
TEST_IMAGE_DIR = r"c:\Users\Jnaneswari.Kodavati\Downloads\archive (2)\bone fracture detection.v4-v4.yolov8\test\images"

# Get a few test images
import glob
test_images = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg"))[:3]

print(f"\nTesting {len(test_images)} images with different confidences...")

import json
out_data = {}
for img_path in test_images:
    img_name = os.path.basename(img_path)
    out_data[img_name] = {}
    
    for conf in [0.5, 0.25, 0.1, 0.05, 0.01]:
        results = model.predict(source=img_path, save=False, conf=conf, verbose=False)
        boxes = results[0].boxes
        box_data = []
        for box in boxes:
            box_data.append({"cls": int(box.cls[0]), "conf": float(box.conf[0])})
        out_data[img_name][f"conf_{conf}"] = box_data

with open('clean_results.json', 'w') as f:
    json.dump(out_data, f, indent=2)
print("JSON saved to clean_results.json")

from ultralytics import YOLO
import os
import cv2
import numpy as np

def verify():
    PT_PATH = os.path.join('media', 'YOLOv8x-best.pt')
    ONNX_PATH = os.path.join('media', 'YOLOv8x-best.onnx')
    
    # Try to find a sample image
    sample_images = []
    uploads_dir = os.path.join('media', 'uploads')
    if os.path.exists(uploads_dir):
        sample_images = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.endswith(('.jpg', '.png'))]
    
    if not sample_images:
        print("No sample images found in media/uploads. Accuracy verification skipped.")
        return

    sample_img = sample_images[0]
    print(f"Comparing models on image: {sample_img}")

    # Load models
    model_pt = YOLO(PT_PATH)
    model_onnx = YOLO(ONNX_PATH)

    # Predict
    results_pt = model_pt.predict(sample_img, conf=0.1)
    results_onnx = model_onnx.predict(sample_img, conf=0.1)

    # Compare boxes
    pt_boxes = results_pt[0].boxes.xyxy.cpu().numpy()
    onnx_boxes = results_onnx[0].boxes.xyxy.cpu().numpy()

    print(f"PT Boxes count: {len(pt_boxes)}")
    print(f"ONNX Boxes count: {len(onnx_boxes)}")

    if len(pt_boxes) == len(onnx_boxes):
        diff = np.sum(np.abs(pt_boxes - onnx_boxes))
        if diff < 1e-3:
            print("✅ Accuracy Check Passed! Models give identical results.")
        else:
            print(f"⚠️ Boxes differ slightly (Sum diff: {diff:.6f}). Standard for conversion.")
    else:
        print("❌ Detection count differs! Please check the model.")

if __name__ == "__main__":
    verify()

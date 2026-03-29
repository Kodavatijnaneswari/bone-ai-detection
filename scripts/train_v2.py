from ultralytics import YOLO
import os

def train_optimized():
    # Model and configurations
    dataset_yaml = r"C:\Users\Jnaneswari.Kodavati\Downloads\archive (2)\bone fracture detection.v4-v4.yolov8\data.yaml"
    model_name = "yolov8n.pt"  # Faster on CPU
    
    print(f"Initializing training on CPU for: {dataset_yaml}")
    model = YOLO(model_name)
    
    # Train with balanced parameters for accuracy + speed on CPU
    model.train(
        data=dataset_yaml,
        epochs=30,           # Sufficient for learning basic features
        imgsz=640,           # High resolution for fractures
        batch=4,             # Small batch to prevent CPU overheating
        patience=10,         # Early stopping if no improvement
        name='optimized_bone_model',
        # Data Augmentation (CRITICAL for medical images)
        hsv_h=0.015,         # Slight hue changes
        hsv_s=0.7,           # Saturation
        hsv_v=0.4,           # Brightness
        degrees=10.0,        # Small rotations
        translate=0.1,       # Shifts
        scale=0.5,           # Scaling
        fliplr=0.5,          # Horizontal flip
        mosaic=1.0           # Mosaic augmentation
    )
    
    print("\n✅ Training Complete!")
    print("New model saved in: runs/detect/optimized_bone_model/weights/best.pt")

if __name__ == "__main__":
    train_optimized()

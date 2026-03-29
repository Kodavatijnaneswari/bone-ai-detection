from ultralytics import YOLO
import sys

try:
    model = YOLO("c:/Users/Jnaneswari.Kodavati/Downloads/113.Bone_Abnormality_Detection/113.Bone_Abnormality_Detection/Code/yolov8s.pt")
    print("Model classes:")
    print(model.names)
except Exception as e:
    print(f"Error: {e}")

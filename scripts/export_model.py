from ultralytics import YOLO
import os
import sys

def main():
    # 1. Define paths
    MODEL_DIR = os.path.join(os.getcwd(), 'media')
    INPUT_PT = os.path.join(MODEL_DIR, 'YOLOv8x-best.pt')
    
    if not os.path.exists(INPUT_PT):
        print(f"**Error: Could not find {INPUT_PT}**")
        sys.exit(1)

    print(f"--- Exporting {INPUT_PT} to ONNX (Accuracy Match Mode) ---")
    
    # 2. Load model
    model = YOLO(INPUT_PT)
    
    # 3. Export to ONNX
    # We use format='onnx' and simplify=True for maximum CPU efficiency.
    # Note: Accuracy is maintained as FP32/FP16 depending on export settings.
    # Default is FP32 which preserves accuracy 100%.
    try:
        onnx_file = model.export(format='onnx', simplify=True)
        print(f"✅ Export Success! ONNX model saved at: {onnx_file}")
        
    except Exception as e:
        print(f"**Export Failed: {e}**")
        sys.exit(1)

if __name__ == "__main__":
    main()

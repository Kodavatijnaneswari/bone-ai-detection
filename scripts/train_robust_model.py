from ultralytics import YOLO
import os

def train_robust_model():
    # Hardcoded path for the easiest user experience
    dataset_yaml_path = r"C:\Users\Jnaneswari.Kodavati\Downloads\archive (2)\bone fracture detection.v4-v4.yolov8\data.yaml"

    import glob
    model_path = "yolov8n.pt"
    resume_mode = False
    
    # Automatically find the absolute latest interrupted run
    runs_dir = os.path.join("runs", "detect")
    run_folders = glob.glob(os.path.join(runs_dir, "clean_bone_model*"))
    
    if run_folders:
        run_folders.sort(key=os.path.getmtime, reverse=True)
        for folder in run_folders:
            candidate_weights = os.path.join(folder, "weights", "last.pt")
            if os.path.exists(candidate_weights):
                print(f"**Found interrupted training! Resuming automatically from: {candidate_weights}**")
                model_path = candidate_weights
                resume_mode = True
                break

    if not resume_mode:
        print("Initializing YOLOv8n (Nano) backbone for much faster CPU training...")

    model = YOLO(model_path)

    if resume_mode:
        print("\n⏳ Resuming training exactly where you were. (Do not close the terminal until 100%)...")
        results = model.train(resume=True)
    else:
        print(f"\nStarting fresh training with dataset: {dataset_yaml_path}")
        print("Applying HIGH RESOLUTION clean training (No image shrinking, No destroying fractures!)")
        results = model.train(
            data=dataset_yaml_path,
            epochs=10,                  # Quick run, 10 epochs is enough for YOLOv8n to learn basic fractures
            imgsz=640,                  # CRITICAL: Keep full resolution so it sees the tiny fractures
            batch=8,                    # Small batch size to avoid freezing CPU
            name='clean_bone_model'     # Entirely new folder so it doesn't resume the broken one
        )
    
    print("\n✅ Training Complete!")
    print("Your new, highly robust model is saved in your 'runs/detect/' folder inside 'weights/best.pt'")
    print("You can copy this 'best.pt' file and replace your old 'C:/Users/.../media/YOLOv8x-best.pt' file to test the web app again!")

if __name__ == "__main__":
    print("\n--- Bone Fracture Retraining Script (Clean High Res) ---")
    train_robust_model()

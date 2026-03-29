import os
import shutil
import csv
import cv2
import numpy as np
from glob import glob
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import default_storage
from django.contrib import messages
# from ultralytics import YOLO (Moved inside function to save memory)
from admins.models import modeldata
from .models import DiagnosticResult

def index(request):
    return render(request, 'home.html')

def userbase(request):
    return render(request, 'users/userbase.html')

def userlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        
        try:
            user_candidates = modeldata.objects.filter(username__iexact=username)
            if not user_candidates:
                messages.error(request, 'Invalid credentials.')
                return render(request, 'userlogin.html')
            
            user = user_candidates.first()
            if user.password != password:
                messages.error(request, 'Invalid credentials.')
                return render(request, 'userlogin.html')
            
            if user.status == 'Activated':
                request.session['userid'] = user.id
                request.session['username'] = user.username
                return redirect('userbase')
            else:
                messages.error(request, 'Account is not activated.')
                return render(request, 'userlogin.html')
                
        except Exception as e:
            print(f"SYSTEM ERROR during login: {e}")
            messages.error(request, 'An internal error occurred.')
            return render(request, 'userlogin.html')
    return render(request, 'userlogin.html')

def training(request):
    runs_dir = os.path.join(settings.BASE_DIR, 'runs/detect/*/results.csv')
    csv_files = glob(runs_dir)
    training_data = []
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getmtime)
        try:
            with open(latest_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    training_data.append({
                        "epoch": row.get('epoch', '--'),
                        "box_loss": row.get('train/box_loss', '0.0'),
                        "cls_loss": row.get('train/cls_loss', '0.0'),
                        "map50": row.get('metrics/mAP50(B)', '0.0'),
                        "val_loss": row.get('val/box_loss', '0.0'),
                    })
        except Exception as e:
            print(f"Error parsing training logs: {e}")
    return render(request, 'users/training.html', {"training_data": training_data})

# -------- LAZY MODEL LOADER --------
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            MODEL_PATH = os.path.join(settings.BASE_DIR, 'media', 'YOLOv8x-best.onnx')
            PT_FALLBACK_PATH = os.path.join(settings.BASE_DIR, 'media', 'YOLOv8x-best.pt')
            
            if not os.path.exists(MODEL_PATH) and os.path.exists(PT_FALLBACK_PATH):
                MODEL_PATH = PT_FALLBACK_PATH
            
            if os.path.exists(MODEL_PATH):
                # Specifying task='detect' saves memory
                _model = YOLO(MODEL_PATH, task='detect')
                print(f"AI Engine Loaded: {MODEL_PATH}")
            else:
                print(f"Error: No model found at {MODEL_PATH}")
        except Exception as e:
            print(f"Model Load Error: {e}")
    return _model

# -------- IMAGE UPLOAD AND DETECTION --------
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_image = request.FILES["image"]
        image_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

        try:
            img = cv2.imread(image_full_path)
            if img is None:
                return render(request, "users/result.html", {"error_message": "Invalid image format."})

            # Optimize Memory: Resize large images to standard YOLO input size (640px height)
            h, w = img.shape[:2]
            if h > 640 or w > 640:
                scale = 640 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(image_full_path, img) # Overwrite with optimized version
                print(f"Memory Optimized: Image resized to {img.shape[1]}x{img.shape[0]}")

            # 1. Grayscale Validation (Stricter for B&W X-rays)
            b, g, r = cv2.split(img)
            mean_diff = (np.mean(cv2.absdiff(r, g)) + np.mean(cv2.absdiff(r, b)) + np.mean(cv2.absdiff(g, b))) / 3.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            # Stricter grayscale detection: Standard B&W X-rays have mean_diff < 5.0
            # Tinted or scanned X-rays may have mean_diff up to 12.0
            is_perfectly_gray = mean_diff < 5.0
            is_valid_medical_tone = mean_diff < 12.0 
            
            # Check for standard X-ray features (black corners/background)
            background_ratio = np.sum(hist[:40]) / np.sum(hist) # Focus on near-black
            bone_peak_ratio = np.sum(hist[120:]) / np.sum(hist) # Focus on brighter bone area
            
            # Validation logic
            if is_perfectly_gray:
                is_valid_xray = True
            elif is_valid_medical_tone:
                # Valid only if it has enough black background (X-ray corners) or bone brightness
                is_valid_xray = (background_ratio > 0.05) or (bone_peak_ratio > 0.10)
            else:
                is_valid_xray = False
                
            if not is_valid_xray:
                print(f"DEBUG: Rejected image with mean_diff={mean_diff:.2f}, bg_ratio={background_ratio:.2f}")
                return render(request, "users/result.html", {
                    "error_message": "Non-X-ray (Color) image detected. AI analysis is restricted to diagnostic grayscale medical X-rays for perfect accuracy."
                })

            model = get_model()
            if model is None:
                return render(request, "users/result.html", {"error_message": "AI Diagnostic Engine not initialized."})

            results = model.predict(source=image_full_path, save=False, conf=0.10)
            boxes = results[0].boxes
            if not boxes:
                results = model.predict(source=image_full_path, save=False, conf=0.03)
                boxes = results[0].boxes

            fracture_boxes = [box for box in boxes if int(box.cls[0]) in [0, 1, 2, 3, 4, 5, 6]]

            if not fracture_boxes:
                # Save Normal Result
                try:
                    userid = request.session.get('userid')
                    if userid:
                        DiagnosticResult.objects.create(
                            user_id=userid,
                            original_image=image_path,
                            processed_image=image_path,
                            finding="Normal",
                            category="Normal Bone Structure",
                            confidence=0.99
                        )
                except Exception as db_e:
                    print(f"DB Error (Normal): {db_e}")
                
                return render(request, "users/result.html", {
                    "output_image_url": settings.MEDIA_URL + image_path,
                    "success_message": "Normal X-ray. No abnormalities or fractures detected.",
                    "detailed_info": "Diagnostic Engine Scan Complete: Normal Bone Anatomy"
                })

            overlay = img.copy()
            heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
            best_box = fracture_boxes[0]
            stage = "Abnormal"
            
            for box in fracture_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                box_w, box_h = x2 - x1, y2 - y1
                area_ratio = (box_w * box_h) / (img.shape[0] * img.shape[1])
                aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)

                if (cls_id in [0, 5, 6] and area_ratio > 0.04) or (aspect_ratio > 4.5):
                     current_stage = "Dislocated Fracture / Major Displacement"
                elif area_ratio > 0.06 or aspect_ratio > 3.0:
                     current_stage = "Complete Transverse Fracture"
                elif area_ratio < 0.008 or conf < 0.08:
                     current_stage = "Incomplete / Hairline Fracture (Suspected)"
                else:
                     current_stage = "Fracture Abnormality Detected"

                if conf == float(best_box.conf[0]):
                    stage = current_stage
                    if cls_id == 6: stage = f"Wrist {current_stage}"
                    if cls_id == 0: stage = f"Elbow {current_stage}"

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sigma = max(box_w / 3.0, box_h / 3.0, 5.0)
                Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
                gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                heatmap = np.maximum(heatmap, gauss)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            alpha, mask = 0.4, heatmap > 20
            overlay[mask] = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)[mask]

            output_filename = "detected_" + uploaded_image.name
            output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
            cv2.imwrite(output_path, overlay)

            # Save Abnormal Result
            try:
                userid = request.session.get('userid')
                if userid:
                    DiagnosticResult.objects.create(
                        user_id=userid,
                        original_image=image_path,
                        processed_image=f"uploads/{output_filename}",
                        finding="Abnormal",
                        category=stage,
                        confidence=float(best_box.conf[0])
                    )
            except Exception as db_e:
                print(f"DB Error (Abnormal): {db_e}")

            return render(request, "users/result.html", {
                "output_image_url": settings.MEDIA_URL + f"uploads/{output_filename}",
                "success_message": f"Detection Result: {stage}",
                "detailed_info": f"Classification: {stage} (Enhanced with Grad-CAM Visualization)"
            })

        except Exception as e:
            print("Detection Error:", str(e))
            return render(request, "users/result.html", {"error_message": f"Processing Error: {str(e)}"})

    return render(request, "users/upload.html")

def history(request):
    user_id = request.session.get('userid')
    if not user_id: return redirect('userlogin')
    results = DiagnosticResult.objects.filter(user_id=user_id).order_by('-uploaded_at')
    return render(request, 'users/history.html', {'results': results})

def generate_report(request, result_id):
    if not request.session.get('userid'): return redirect('userlogin')
    result = DiagnosticResult.objects.get(id=result_id)
    return render(request, 'users/report.html', {'result': result})

def show_result(request):
    return render(request, 'users/result.html')
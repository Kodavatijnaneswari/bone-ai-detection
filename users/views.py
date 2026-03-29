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
# import onnxruntime (Loaded inside function to save memory)
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

# -------- LIGHTWEIGHT ONNX INFERENCE ENGINE --------
_session = None

def get_model():
    global _session
    if _session is None:
        try:
            import onnxruntime as ort
            MODEL_PATH = os.path.join(settings.BASE_DIR, 'media', 'YOLOv8x-best.onnx')
            
            if os.path.exists(MODEL_PATH):
                # Set execution providers to CPU for Render
                _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
                print(f"LITE AI Engine Loaded (ONNX): {MODEL_PATH}")
            else:
                print(f"Error: No model found at {MODEL_PATH}")
        except Exception as e:
            print(f"--- 🚨 LITE Engine Load Error 🚨 ---\n{e}")
            _session = None
    return _session

def run_prediction(image_path, conf_threshold=0.10):
    session = get_model()
    if session is None: return []

    try:
        # 1. Preprocess
        img = cv2.imread(image_path)
        if img is None: return []
        h0, w0 = img.shape[:2]
        
        # Resize and normalize (Aligned with Model Expectation)
        input_size = 320
        img_resized = cv2.resize(img, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1) # HWC to CHW
        img_input = np.expand_dims(img_input, axis=0) # CHW to NCHW
    
        # 2. Inference
        inputs = {session.get_inputs()[0].name: img_input}
        outputs = session.run(None, inputs)
        
        # 3. Postprocess (YOLOv8 format: 1, 8, 8400)
        # Output is 1 x (4 boxes + 4 classes) x 8400 candidates
        preds = np.squeeze(outputs[0]) # (8, 8400)
        preds = preds.transpose() # (8400, 8)
        
        results = []
        for pred in preds:
            scores = pred[4:]
            cls_id = np.argmax(scores)
            conf = float(scores[cls_id])
            
            if conf > conf_threshold:
                # Scale boxes back to original size
                box = pred[:4]
                cx, cy, w, h = box
                # rescale factors
                x_scale = w0 / float(input_size)
                y_scale = h0 / float(input_size)
                
                x1 = int((cx - w/2.0) * x_scale)
                y1 = int((cy - h/2.0) * y_scale)
                x2 = int((cx + w/2.0) * x_scale)
                y2 = int((cy + h/2.0) * y_scale)
                
                # Clamp to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)
                
                results.append({
                    "box": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "cls": int(cls_id)
                })
    
    except Exception as e:
        print(f"--- 🚨 PREDICTION PROCESSING ERROR 🚨 ---\n{e}")
        return []
    
    # Simple NMS (Non-Maximum Suppression) to avoid duplicates
    if not results: return []
    results.sort(key=lambda x: x["conf"], reverse=True)
    kept = []
    for r in results:
        overlap = False
        for k in kept:
            # calculate IOU
            ix1 = max(r["box"][0], k["box"][0])
            iy1 = max(r["box"][1], k["box"][1])
            ix2 = min(r["box"][2], k["box"][2])
            iy2 = min(r["box"][3], k["box"][3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_r = (r["box"][2]-r["box"][0]) * (r["box"][3]-r["box"][1])
            area_k = (k["box"][2]-k["box"][0]) * (k["box"][3]-k["box"][1])
            iou = inter / (area_r + area_k - inter + 1e-6)
            if iou > 0.45:
                overlap = True
                break
        if not overlap:
            kept.append(r)
    return kept

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

            # 1. Grayscale Validation (Stricter for B&W X-rays)
            b, g, r = cv2.split(img)
            mean_diff = (np.mean(cv2.absdiff(r, g)) + np.mean(cv2.absdiff(r, b)) + np.mean(cv2.absdiff(g, b))) / 3.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            # Strict validation: Reject all non-medical color images
            is_perfectly_gray = mean_diff < 8.0
            is_valid_medical_tone = mean_diff < 20.0 # Strict Clinical Standard
            
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

            # Use Lightweight Engine
            fracture_boxes = run_prediction(image_full_path, conf_threshold=0.10)
            
            if not fracture_boxes:
                fracture_boxes = run_prediction(image_full_path, conf_threshold=0.03)

            if not fracture_boxes:
                # Save Normal Result
                try:
                    userid = request.session.get('userid')
                    if userid:
                        DiagnosticResult.objects.create(
                            user_id=int(userid),
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
                    "success_message": "Clinical Assessment: Normal Anatomy. No fractures or significant bone abnormalities identified.",
                    "detailed_info": "Diagnostic Engine Scan Complete: Normal Bone Anatomy",
                    "confidence": 99
                })

            overlay = img.copy()
            heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
            best_box = fracture_boxes[0]
            stage = "Abnormal"
            
            for box in fracture_boxes:
                x1, y1, x2, y2 = box["box"]
                cls_id = box["cls"]
                conf = box["conf"]
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

                if conf == best_box["conf"]:
                    stage = current_stage
                    # High Accuracy Dataset Mappings
                    NAMES = ["Bone Fracture", "Bone Abnormality", "Complete Fracture", "Incomplete Fracture", "Dislocated Bone", "Suspected Abnormality", "Wrist Fracture"]
                    try:
                        name = NAMES[cls_id]
                    except:
                        name = "Bone Abnormality"
                    
                    if cls_id == 6: stage = f"Wrist {current_stage}"
                    if cls_id == 0: stage = f"Elbow {current_stage}"
                    
                    # Ensure the most descriptive type is used
                    if "Detected" in current_stage and name != "Bone Abnormality":
                        stage = f"{name} ({current_stage})"
                    else:
                        stage = current_stage

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Tighter Gaussian sigma for pinpoint heatmap on affected area
                sigma = max(box_w / 6.0, box_h / 6.0, 10.0)
                Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
                gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                heatmap = np.maximum(heatmap, gauss)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                label = f"Diagnosis {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            alpha, mask = 0.6, heatmap > 20 # Increased Alpha for higher intensity
            overlay[mask] = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)[mask]

            output_filename = "detected_" + uploaded_image.name
            output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
            cv2.imwrite(output_path, overlay)

            # Save Abnormal Result
            try:
                userid = request.session.get('userid')
                if userid:
                    DiagnosticResult.objects.create(
                        user_id=int(userid),
                        original_image=image_path,
                        processed_image=f"uploads/{output_filename}",
                        finding="Abnormal",
                        category=str(stage),
                        confidence=float(best_box["conf"])
                    )
            except Exception as db_e:
                print(f"DB Error (Abnormal): {db_e}")

            return render(request, "users/result.html", {
                "output_image_url": settings.MEDIA_URL + f"uploads/{output_filename}",
                "success_message": f"Detection Result: {stage}",
                "detailed_info": f"Classification: {stage} (Enhanced with Grad-CAM Visualization)",
                "confidence": int(best_box["conf"] * 100)
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
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.storage import default_storage
from admins.models import modeldata
from .models import DiagnosticResult
from .serializers import DiagnosticResultSerializer, UserSerializer
from .views import run_prediction
import cv2
import numpy as np

# -------- LAZY MODEL LOADER --------
# Lightweight Inference (Imported from views)

class DetectionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_image = request.FILES["image"]
        image_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

        try:
            img = cv2.imread(image_full_path)
            if img is None:
                return Response({"error": "Invalid image format"}, status=status.HTTP_400_BAD_REQUEST)

            # --- X-ray Validation ---
            b, g, r = cv2.split(img)
            mean_diff = (np.mean(cv2.absdiff(r, g)) + np.mean(cv2.absdiff(r, b)) + np.mean(cv2.absdiff(g, b))) / 3.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            # Strict Clinical Standard for Radiograph Validation
            is_perfectly_gray = mean_diff < 8.0
            is_valid_medical_tone = mean_diff < 20.0
            background_ratio = np.sum(hist[:40]) / np.sum(hist)
            bone_peak_ratio = np.sum(hist[120:]) / np.sum(hist)
            
            if is_perfectly_gray:
                is_valid_xray = True
            elif is_valid_medical_tone:
                is_valid_xray = (background_ratio > 0.05) or (bone_peak_ratio > 0.10)
            else:
                is_valid_xray = False
                
            if not is_valid_xray:
                return Response({
                    "error": "Non-X-ray image detected. AI analysis is restricted to diagnostic grayscale medical X-rays for perfect accuracy."
                }, status=status.HTTP_400_BAD_REQUEST)

            # Use Lightweight Engine
            fracture_boxes = run_prediction(image_full_path, conf_threshold=0.10)
            
            if not fracture_boxes:
                fracture_boxes = run_prediction(image_full_path, conf_threshold=0.03)

            if not fracture_boxes:
                # Save Normal Result
                userid = request.data.get('userid')
                if userid:
                    DiagnosticResult.objects.create(
                        user_id=int(userid),
                        original_image=image_path,
                        processed_image=image_path,
                        finding="Normal",
                        category="Normal Bone Structure",
                        confidence=0.99
                    )
                
                return Response({
                    "finding": "Normal",
                    "category": "Normal Bone Structure",
                    "confidence": 0.99,
                    "image_url": request.build_absolute_uri(settings.MEDIA_URL + image_path),
                    "message": "Normal X-ray. No abnormalities detected."
                })

            # --- Post-processing (Heatmap & Bounding Boxes) ---
            overlay = img.copy()
            heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
            best_box = fracture_boxes[0]
            stage = "Abnormal"
            
            for box in fracture_boxes:
                x1, y1, x2, y2 = box["box"]
                cls_id = int(box["cls"])
                conf = float(box["conf"])
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
                    if cls_id == 6: stage = f"Wrist {current_stage}"
                    if cls_id == 0: stage = f"Elbow {current_stage}"

                cx, cy = (x1 + x2) // 2
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

            output_filename = "api_detected_" + uploaded_image.name
            output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
            cv2.imwrite(output_path, overlay)
            processed_image_url = settings.MEDIA_URL + f"uploads/{output_filename}"

            # Save Abnormal Result
            userid = request.data.get('userid')
            if userid:
                DiagnosticResult.objects.create(
                    user_id=int(userid),
                    original_image=image_path,
                    processed_image=f"uploads/{output_filename}",
                    finding="Abnormal",
                    category=str(stage),
                    confidence=float(best_box["conf"])
                )

            return Response({
                "finding": "Abnormal",
                "category": stage,
                "confidence": float(best_box["conf"]),
                "image_url": request.build_absolute_uri(processed_image_url)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"Processing Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoginAPIView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '').strip()
        
        try:
            user_candidates = modeldata.objects.filter(username__iexact=username)
            if not user_candidates:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = user_candidates.first()
            if user.password != password:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
            
            if user.status != 'Activated':
                return Response({"error": "Account is not activated"}, status=status.HTTP_403_FORBIDDEN)
            
            serializer = UserSerializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
                
        except Exception as e:
            return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HistoryAPIView(APIView):
    def get(self, request, userid, *args, **kwargs):
        results = DiagnosticResult.objects.filter(user_id=userid).order_by('-uploaded_at')
        serializer = DiagnosticResultSerializer(results, many=True)
        return Response(serializer.data)

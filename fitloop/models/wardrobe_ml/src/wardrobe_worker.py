"""
Wardrobe ML Out-of-Process Worker
=================================
Runs YOLO garment detection, MobileNetV3 embedding extraction, and
CIE LAB color extraction in an isolated Python process.

This isolation is required on macOS Apple Silicon to avoid OpenMP
and C-runtime conflicts (pthread_mutex_init / libomp collision) between
MediaPipe (TensorFlow Lite C++) and PyTorch/TorchVision.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prevent duplicate OpenMP library crashes inside this worker process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torchvision.models as models
from ultralytics import YOLO

# Import local color extractor
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from color_extractor import GarmentColorExtractor


class WardrobeInferenceWorker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.yolo = None
        if model_path and Path(model_path).exists():
            try:
                self.yolo = YOLO(model_path)
            except Exception as e:
                sys.stderr.write(f"Failed to load YOLO weights from {model_path}: {e}\n")

        # Setup MobileNetV3 feature extractor
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.embed_model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            self.embed_model = models.mobilenet_v3_small(pretrained=True)

        self.embed_model.classifier = torch.nn.Identity()
        self.embed_model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # Setup color extractor
        self.color_extractor = GarmentColorExtractor()

    def extract_embedding(self, crop_bgr: np.ndarray) -> List[float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return [0.0] * 576
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            t = ((t - self.mean) / self.std).unsqueeze(0)
            with torch.no_grad():
                feat = self.embed_model(t).squeeze(0).numpy()
            norm = float(np.linalg.norm(feat))
            if norm < 1e-12:
                return [0.0] * 576
            return (feat / norm).astype(float).tolist()
        except Exception as e:
            sys.stderr.write(f"Embedding extraction error: {e}\n")
            return [0.0] * 576

    def extract_colors(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        try:
            res = self.color_extractor.extract(crop_bgr)
            dom = res.get("dominant_color", {}) or {}
            sec = res.get("secondary_color", {}) or {}
            
            dom_hex = dom.get("hex", "#2B4C7E")
            sec_hex = sec.get("hex", "#E0E6ED")
            dom_name = dom.get("name", "Garment Color")
            sec_name = sec.get("name", "Secondary Color")
            
            # Compute LAB coordinates for dominant color
            dom_rgb = dom.get("rgb", [43, 76, 126])
            bgr_f = np.float32([[[dom_rgb[2] / 255.0, dom_rgb[1] / 255.0, dom_rgb[0] / 255.0]]])
            lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB)[0, 0].tolist()

            return {
                "dominant_hex": dom_hex,
                "secondary_hex": sec_hex,
                "dominant_name": dom_name,
                "secondary_name": sec_name,
                "dominant_lab": lab,
            }
        except Exception as e:
            sys.stderr.write(f"Color extraction error: {e}\n")
            return {
                "dominant_hex": "#2B4C7E",
                "secondary_hex": "#E0E6ED",
                "dominant_name": "Garment Color",
                "secondary_name": "Secondary Color",
                "dominant_lab": [50.0, 0.0, 0.0],
            }

    def process_scan(
        self,
        image_path: str,
        crops_dir: str,
        scan_uid: str,
        conf_thresh: float = 0.25,
    ) -> List[Dict[str, Any]]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return []

        h, w = img_bgr.shape[:2]
        detections = []

        if self.yolo is not None:
            try:
                results = self.yolo.predict(
                    source=img_bgr, conf=conf_thresh, device="cpu", verbose=False
                )
                box_counter = 1
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = (
                            r.names[cls_id]
                            if hasattr(r, "names") and cls_id in r.names
                            else "apparel"
                        )

                        x1, y1, x2, y2 = xyxy
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop_bgr = img_bgr[y1:y2, x1:x2]

                        crop_filename = f"crop_{scan_uid}_{box_counter}.jpg"
                        crop_path = Path(crops_dir) / crop_filename

                        if crop_bgr.size > 0:
                            cv2.imwrite(str(crop_path), crop_bgr)
                            crop_url = f"/uploads/crops/{crop_filename}"
                        else:
                            crop_url = f"/uploads/scans/{Path(image_path).name}"
                            crop_bgr = img_bgr

                        color_info = self.extract_colors(crop_bgr)
                        emb = self.extract_embedding(crop_bgr)

                        detections.append({
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [x1, y1, x2, y2],
                            "box_id": box_counter,
                            "crop_url": crop_url,
                            "dominant_hex": color_info["dominant_hex"],
                            "secondary_hex": color_info["secondary_hex"],
                            "dominant_name": color_info["dominant_name"],
                            "secondary_name": color_info["secondary_name"],
                            "dominant_lab": color_info["dominant_lab"],
                            "embedding": emb,
                        })
                        box_counter += 1
            except Exception as e:
                sys.stderr.write(f"YOLO predict error: {e}\n")

        # Fallback if no detections found
        if not detections:
            crop_filename = f"crop_{scan_uid}_1.jpg"
            crop_path = Path(crops_dir) / crop_filename
            cv2.imwrite(str(crop_path), img_bgr)
            color_info = self.extract_colors(img_bgr)
            emb = self.extract_embedding(img_bgr)
            detections.append({
                "class_name": "clothing item",
                "confidence": 0.75,
                "bbox": [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)],
                "box_id": 1,
                "crop_url": f"/uploads/crops/{crop_filename}",
                "dominant_hex": color_info["dominant_hex"],
                "secondary_hex": color_info["secondary_hex"],
                "dominant_name": color_info["dominant_name"],
                "secondary_name": color_info["secondary_name"],
                "dominant_lab": color_info["dominant_lab"],
                "embedding": emb,
            })

        return detections


def main():
    if len(sys.argv) < 5:
        sys.stderr.write("Usage: python wardrobe_worker.py <model_path> <image_path> <crops_dir> <scan_uid>\n")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    crops_dir = sys.argv[3]
    scan_uid = sys.argv[4]

    worker = WardrobeInferenceWorker(model_path)
    res = worker.process_scan(image_path, crops_dir, scan_uid)

    sys.stdout.write("WARDROBE_RESULT_START\n")
    sys.stdout.write(json.dumps(res))
    sys.stdout.write("\nWARDROBE_RESULT_END\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

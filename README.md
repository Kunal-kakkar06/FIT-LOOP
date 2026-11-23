# FitLoop 👗🧥👔
> **AI-Powered Wardrobe Intelligence, Garment Localization, Color Extraction & Duplicate Prevention Engine**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLO26n](https://img.shields.io/badge/Model-YOLO26n--5.4MB-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1--CPU-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

---

## 📌 1. Project Overview & Architecture

FitLoop is an end-to-end computer vision and wardrobe management platform designed to help users digitize their closets, track apparel inventory, and eliminate redundant impulse purchases through real-time **Check Before Buy** duplicate detection.

### Core Processing Pipeline

```
                                 [ User Photo Input ]
                           (Closet, Flat-Lay, Single Garment)
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │   YOLO26n Object Engine   │
                           │  13-Class Garment Bounds  │
                           └─────────────┬─────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
      ┌───────────────────────────┐             ┌───────────────────────────┐
      │  CIE LAB K-Means Cluster  │             │     MobileNetV3-Small     │
      │  60% Core Color Extraction│             │   Visual Feature Backbone │
      │  Dominant & Secondary Hex │             │   576-dim L2 Unit Vector  │
      └─────────────┬─────────────┘             └─────────────┬─────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │     DuplicateDetector     │
                           │ Multi-Attribute Blending: │
                           │  - Category Compatibility │
                           │  - CIE LAB Delta-E Metric │
                           │  - Visual Cosine Match    │
                           │  - Pattern & Secondary    │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │    Decision & Advice:     │
                           │  - DUPLICATE (score>=0.85)│
                           │  - SIMILAR   (0.65 - 0.84)│
                           │  - UNIQUE    (score <0.65)│
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │    FastAPI & SQLite DB    │
                           │ Multi-Tenant Persistence: │
                           │ Users, Scans, Items, Embed│
                           └───────────────────────────┘
```

---

## 🏷️ 2. FitLoop 13-Class Garment Taxonomy

FitLoop maps apparel into 13 mutually exclusive standardized garment categories (adapted from DeepFashion2 and real-world multi-angle consumer wardrobes):

| ID | Class Name | Typical Garments Covered |
| :-: | :--- | :--- |
| **0** | `short sleeve top` | T-shirts, polo shirts, short-sleeve crop tops |
| **1** | `long sleeve top` | Long-sleeve shirts, button-downs, pullovers, crewnecks, hoodies |
| **2** | `short sleeve outwear` | Short-sleeve vests, chore coats, capes |
| **3** | `long sleeve outwear` | Jackets, blazers, trench coats, parkas, denim jackets |
| **4** | `vest` | Sleeveless vests, knit sweater vests, waistcoats |
| **5** | `sling` | Camisoles, spaghetti strap tops, tank tops |
| **6** | `shorts` | Denim shorts, athletic shorts, tailored shorts |
| **7** | `trousers` | Jeans, slacks, chinos, joggers, folded pants, sweatpants |
| **8** | `skirt` | Mini, midi, maxi, and pleated skirts |
| **9** | `short sleeve dress` | T-shirt dresses, short-sleeve midi/maxi dresses |
| **10** | `long sleeve dress` | Long-sleeve evening gowns, knit dresses, maxi dresses |
| **11** | `vest dress` | Sleeveless dresses, pinafores |
| **12** | `sling dress` | Slip dresses, spaghetti strap gowns |

---

## 📊 3. Verified Model Evaluation Scorecard (Phase 6 Production Model)

Evaluated against the **unseen real-world test splits** on an Intel Core i5 CPU without GPU dependency:

### Primary Benchmark: Massive Casual & Retail Apparel Split (`tyaw9` 837 Unseen Images)

| Metric | Measured Score | Evaluation Notes |
| :--- | :---: | :--- |
| **Overall Precision (P)** | **73.07%** | High precision across dense, varied, and real-world casual outfits |
| **Overall Recall (R)** | **67.67%** | Comprehensive item localization across complex layering and occlusions |
| **Overall mAP@50** | **71.25%** | +34.00% absolute increase over Phase 5 baseline (37.25%) |
| **Overall mAP@50-95** | **53.08%** | +27.96% absolute increase over Phase 5 baseline (25.12%) |
| **Inference Latency** | **50.38 ms** | Fast CPU inference (~19.2 FPS on Intel 12th-Gen Core i5) |
| **Model Size** | **5.46 MB** | Lightweight YOLO26n footprint (`wardrobe_ml/models/best.pt`) |

### Itemized Per-Class Performance (`tyaw9` 837 Images, 1,513 Instances)

| Class ID | Garment Category | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| **0** | `short sleeve top` | 233 | **80.71%** | **89.70%** | **92.43%** | **83.37%** |
| **1** | `long sleeve top` | 381 | 66.10% | 63.52% | 64.40% | 43.03% |
| **3** | `long sleeve outwear` | 273 | 74.51% | 55.31% | 59.32% | 42.76% |
| **6** | `shorts` | 114 | 73.86% | 50.00% | 51.28% | 30.10% |
| **7** | `trousers` | 258 | 68.73% | 64.74% | **70.85%** | **56.45%** |
| **8** | `skirt` | 163 | 75.66% | 72.39% | **78.49%** | **55.97%** |
| **9** | `short sleeve dress` | 91 | 71.94% | 78.02% | **81.96%** | **59.91%** |
| **ALL** | **Combined Aggregate** | **1,513** | **73.07%** | **67.67%** | **71.25%** | **53.08%** |

*Note: Classes 2, 4, 5, 10, 11, and 12 had 0 ground-truth occurrences in this split.*

### Complementary Benchmark: Real-World Multi-Garment Split (143 Unseen Images)

| Metric | Measured Score | Evaluation Notes |
| :--- | :---: | :--- |
| **Overall Precision (P)** | **75.54%** | Low false-positive rate across crowded closet environments |
| **Overall Recall (R)** | **73.95%** | Consistent garment identification in multi-garment wardrobe photos |
| **Overall mAP@50** | **77.19%** | Robust IoU overlap across diverse lighting and garment types |
| **Overall mAP@50-95** | **62.95%** | High box localization fidelity |

---

## 🛠️ 4. Quickstart & Installation Guide

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git
- Intel/AMD CPU or Apple Silicon (CUDA optional, not required)

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-org>/Fitloop.git
cd Fitloop
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv fitloop_yolo_env
.\fitloop_yolo_env\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv fitloop_yolo_env
source fitloop_yolo_env/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
pip install -r wardrobe_ml/requirements-yolo.txt
```

---

## 🧪 5. Automated Verification Test Suites

FitLoop includes 5 complete automated verification suites covering all layers of the stack:

```bash
# 1. Verify CIE LAB K-Means color clustering & swatch identification
python wardrobe_ml/src/test_color_extractor.py

# 2. Verify MobileNetV3-Small 576-dim feature extraction & cosine similarity
python wardrobe_ml/src/test_embedding_extractor.py

# 3. Verify CIE LAB color distance & multi-attribute duplicate scoring
python wardrobe_ml/src/test_duplicate_detector.py

# 4. Verify SQLite multi-tenant isolation, ORM relationships & ItemEmbedding
python wardrobe_ml/src/test_database.py

# 5. Run end-to-end FastAPI integration tests (13/13 tests)
python wardrobe_ml/src/test_api.py

# 6. Generate complete Phase 6 Precision/Recall/mAP scorecard on unseen test sets
python wardrobe_ml/src/evaluate_scorecard_phase6.py
```

---

## 🚀 6. Running the Production API Server

Launch the FastAPI application with Uvicorn ASGI server:

```bash
# Run server from repository root
cd wardrobe_ml/src
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: `GET http://localhost:8000/health`

### Key API Endpoints

#### 1. System Health (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```

#### 2. Register User (`POST /api/v1/users`)
```bash
curl -X POST http://localhost:8000/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Fashionista", "email": "alice@fitloop.ai"}'
```

#### 3. Upload Wardrobe Scan (`POST /api/v1/scans`)
Uploads a closet photo or flat-lay, runs YOLO26n garment localization, extracts dominant/secondary colors with 20% core trimming, extracts 576-dim visual embeddings, and persists the items atomically.
```bash
curl -X POST http://localhost:8000/api/v1/scans \
  -F "user_id=1" \
  -F "input_type=flat_lay" \
  -F "file=@path/to/wardrobe_photo.jpg"
```

#### 4. Retrieve Wardrobe Inventory (`GET /api/v1/wardrobe/{user_id}`)
Retrieves all owned items for the tenant with strict user isolation:
```bash
curl -X GET http://localhost:8000/api/v1/wardrobe/1
```

#### 5. Check Before Buy (`POST /api/v1/check-before-buy`)
Prospective purchase verification: extracts candidate category, color, and visual embedding, compares against user wardrobe inventory, and returns a tri-tier purchase decision (`DUPLICATE`, `SIMILAR`, or `UNIQUE`).
```bash
curl -X POST http://localhost:8000/api/v1/check-before-buy \
  -F "user_id=1" \
  -F "file=@path/to/prospective_garment.jpg"
```

---

## 🔒 7. Repository Guidelines & Ignored Artifacts

To prevent repository bloat and ensure fast cloning for all teammates, the following are strictly excluded via `.gitignore`:
- Python environments (`fitloop_yolo_env/`, `.venv/`)
- Raw dataset downloads and extracted image splits (`wardrobe_ml/data/`, `data/`)
- Temporary Kaggle and Colab staging files (`*.zip`, `*.tar.gz`)
- SQLite databases and user upload directories (`*.db`, `uploads/`)
- Archived intermediate phase checkpoints (`best_phase*.pt`, `yolo26n.pt`)

The single authoritative production model tracked in Git is:
👉 **`wardrobe_ml/models/best.pt`** (5.45 MB)

---

## 👥 8. Contribution & Pair Programming Workflow
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Run test suites locally before submitting changes: `python wardrobe_ml/src/test_api.py`
3. Commit clean code preserving documentation integrity and type hints.
4. Open a pull request against `main`.

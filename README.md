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

## 📊 3. Verified Model Evaluation Scorecard

Evaluated against the **unseen real-world test split** (143 photographs, 238 ground-truth instances) on an Intel Core i5 CPU without GPU dependency:

### Overall Benchmark Metrics

| Metric | Measured Score | Evaluation Notes |
| :--- | :---: | :--- |
| **Overall Precision (P)** | **90.40%** | Exceptionally low false-positive clutter across crowded closets |
| **Overall Recall (R)** | **86.46%** | High item capture across folded, angled, and layered clothing |
| **Overall mAP@50** | **93.85%** | Exceeds production acceptance threshold ($\ge 90\%$) |
| **Overall mAP@50-95** | **78.51%** | Robust IoU box localization precision |
| **Inference Latency** | **58.40 ms** | Fast CPU inference (~16.6 FPS on modern 12th-gen Intel CPU) |
| **Model Size** | **5.45 MB** | Lightweight YOLO26n footprint (`wardrobe_ml/models/best.pt`) |

### Itemized Per-Class Performance

| Class ID | Garment Category | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| **0** | `short sleeve top` | 30 | 79.88% | 70.00% | 82.99% | 66.79% |
| **1** | `long sleeve top` | 66 | 89.55% | 77.90% | 87.95% | 68.86% |
| **3** | `long sleeve outwear` | 10 | 81.96% | 90.92% | 97.83% | 90.17% |
| **6** | `shorts` | 9 | 98.24% | 100.00% | 99.50% | 91.56% |
| **7** | `trousers` | 44 | **89.55%** | **93.18%** | **94.71%** | **77.18%** |
| **8** | `skirt` | 47 | 93.66% | 93.62% | 97.75% | 76.45% |
| **9** | `short sleeve dress` | 32 | 100.00% | 79.61% | 96.18% | 78.58% |
| **ALL** | **Combined Aggregate** | **238** | **90.40%** | **86.46%** | **93.85%** | **78.51%** |

*Note: Classes 2, 4, 5, 10, 11, and 12 had 0 ground-truth occurrences in this 143-image test split.*

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

FitLoop includes 4 complete automated verification suites covering all layers of the stack:

```bash
# 1. Verify MobileNetV3-Small 576-dim feature extraction & cosine similarity
python wardrobe_ml/src/test_embedding_extractor.py

# 2. Verify CIE LAB color distance & multi-attribute duplicate scoring
python wardrobe_ml/src/test_duplicate_detector.py

# 3. Verify SQLite multi-tenant isolation, ORM relationships & ItemEmbedding
python wardrobe_ml/src/test_database.py

# 4. Run end-to-end FastAPI integration tests (13/13 tests)
python wardrobe_ml/src/test_api.py

# 5. Generate complete Precision/Recall/mAP scorecard on unseen test sets
python wardrobe_ml/src/evaluate_final_scorecard.py
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

# FitLoop Changelog 👗🧥👔

All notable changes, architectural milestones, model training phases, and dataset updates for the **FitLoop** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Docs Directory**: Created `docs/` repository hub for changelogs, architecture specifications, and training runbooks.
- **Phase 6 Dataset Research & Sourcing**: Evaluated open-source datasets for hanging and folded wardrobe items:
  - Identified `fatihgultekin/clothes-detection-tyaw9` (11,000+ images) as the primary candidate for retail rack and folded table garments.
  - Sourced domestic closet hanging garment data from `master-2023-y3v5p/wardrobe-detector`.
  - Identified fine-grained multi-class & material attributes in Fashionpedia (48k images, 294 textile attributes).
- **Implementation Plan**: Documented Phase 6 dataset pipeline in `wardrobe_hanging_folded_dataset_plan.md`.

### Changed
- **Git Tracking Rules**: Updated `.gitignore` to ignore internal IDE and AI assistant metadata (`.agent/`).
- **Remote Synchronization**: Synchronized and checked out branch `wadrobe_ml` tracking `origin/wadrobe_ml` from `https://github.com/Kunal-kakkar06/FIT-LOOP.git`.

---

## [1.0.0] - 2026-11-16 (Commit `0d2edb3`)

### Added
- **YOLO26n Production Model**:
  - Pinned single authoritative 5.45 MB weights checkpoint at `wardrobe_ml/models/best.pt`.
  - Multi-phase domain adaptation:
    - **Phase 1 & 2**: Trained on DeepFashion2 (191,958 train / 32,153 val images) covering 13 canonical garment classes.
    - **Phase 3**: Fine-tuned on Roboflow dataset (`faculdade-xuvm9/clothes-detection-nnwwn`, 1,404 train / 254 val images) for isolated and flat-lay clothing.
    - **Phase 4 & 5**: Added initial domain-adaptation splits for folded trousers and angled wardrobe images.
- **Color Extraction Module** (`wardrobe_ml/src/color_extractor.py`):
  - K-Means clustering in **CIE LAB** color space.
  - 60% core bounding box crop to eliminate background noise and hanger artifacts.
  - Extracts dominant and secondary hex colors.
- **Visual Embedding Module** (`wardrobe_ml/src/embedding_extractor.py`):
  - MobileNetV3-Small backbone generating normalized 576-dimensional $L_2$ unit feature vectors.
  - Fast CPU cosine similarity computation for visual pattern, texture, and silhouette matching.
- **Duplicate Prevention Engine** (`wardrobe_ml/src/duplicate_detector.py`):
  - Multi-attribute composite score blending category compatibility, CIE LAB $\Delta E$ color distance, and visual embedding cosine match.
  - Tri-tier decision engine:
    - `DUPLICATE`: Score $\ge 0.85$
    - `SIMILAR`: Score $0.65 - 0.84$
    - `UNIQUE`: Score $< 0.65$
- **FastAPI Production Service** (`wardrobe_ml/src/app.py`):
  - Endpoints:
    - `GET /health`: Health and model status.
    - `POST /api/v1/users`: Tenant registration.
    - `POST /api/v1/scans`: Upload closet/flat-lay photo, run YOLO26n detection, extract colors and embeddings, persist atomically.
    - `GET /api/v1/wardrobe/{user_id}`: Retrieve digitized wardrobe inventory.
    - `POST /api/v1/check-before-buy`: Real-time duplicate prevention for prospective purchases.
- **Multi-Tenant SQLite Persistence** (`wardrobe_ml/src/database.py`):
  - SQLAlchemy ORM models for `User`, `Scan`, `Item`, and `ItemEmbedding`.
- **Automated Verification Suites**:
  - `wardrobe_ml/src/test_api.py`: 13/13 passing FastAPI integration tests.
  - `wardrobe_ml/src/test_color_extractor.py`: Color extraction unit tests.
  - `wardrobe_ml/src/test_embedding_extractor.py`: 576-dim feature extraction tests.
  - `wardrobe_ml/src/test_duplicate_detector.py`: Decision threshold and multi-attribute scoring tests.
  - `wardrobe_ml/src/test_database.py`: SQLite tenant isolation and relational integrity tests.
  - `wardrobe_ml/src/evaluate_final_scorecard.py`: Benchmark evaluation script producing `final_model_scorecard.txt`.
- **Dataset Partitioning & Packaging Utilities**:
  - `split_fitloop_dataset.py`: In-memory streaming partitioner splitting large training zips into ~900 MiB Kaggle-ready chunks.
  - `verify_kaggle_parts.py`: Checksum and integrity verifier for Kaggle splits.
  - `wardrobe_ml/src/prepare_roboflow_phase3.py`: Polygon-to-YOLO bounding box converter for Roboflow 8-class data.
  - `wardrobe_ml/src/prepare_folded_phase4.py`: Phase 4 folded trousers dataset converter and packaging pipeline.
  - `wardrobe_ml/src/prepare_angled_phase5.py`: Phase 5 angled wardrobe dataset converter and packaging pipeline.

---

## [0.1.0] - 2026-11-15 (Commit `bdd7652`)

### Added
- **Repository Scaffold**: Initialized project workspace, architecture contracts, and directory hierarchy for FitLoop modules.


"""
FitLoop — wardrobe_ml Configuration
=====================================
Centralized configuration for all DeepFashion2 dataset paths,
workspace directories, and project-level constants.

Usage
-----
    from config import cfg
    print(cfg.DATASET_ROOT)

    # Or direct variable imports:
    from config import (
        PROJECT_ROOT,
        DATASET_ROOT,
        TRAIN_IMAGE_DIR,
        TRAIN_ANNOTATION_DIR,
        VALIDATION_IMAGE_DIR,
        VALIDATION_ANNOTATION_DIR,
        TEST_IMAGE_DIR,
        TEST_METADATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        DF2_CATEGORIES,
        DF2_TO_YOLO_CLASS_MAP,
        YOLO_CLASS_NAMES,
    )
"""

import os as _os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. Root Anchors
# ─────────────────────────────────────────────────────────────────────────────

# Directory containing this config.py file  →  .../Fitloop/wardrobe_ml
ML_ROOT: Path = Path(__file__).resolve().parent

# Parent FitLoop repository root  →  .../Fitloop
PROJECT_ROOT: Path = ML_ROOT.parent


# ─────────────────────────────────────────────────────────────────────────────
# 2. Centralized Dataset Paths (DeepFashion2)
# ─────────────────────────────────────────────────────────────────────────────

# Final verified DeepFashion2 root directory
DATASET_ROOT: Path = Path(
    _os.environ.get(
        "FITLOOP_DATASET_ROOT",
        r"C:\Users\Lenovo\Downloads\DeepFashion2"
    )
)

# Train split (191,961 images, 191,961 JSON annotations)
TRAIN_DIR: Path = DATASET_ROOT / "train"
TRAIN_IMAGE_DIR: Path = TRAIN_DIR / "image"
TRAIN_ANNOTATION_DIR: Path = TRAIN_DIR / "annos"

# Validation split (32,153 images, 32,153 JSON annotations)
VALIDATION_DIR: Path = DATASET_ROOT / "validation"
VALIDATION_IMAGE_DIR: Path = VALIDATION_DIR / "image"
VALIDATION_ANNOTATION_DIR: Path = VALIDATION_DIR / "annos"

# Test split (29,765 images, no ground-truth per-image annotation folder)
TEST_DIR: Path = DATASET_ROOT / "test"
TEST_IMAGE_DIR: Path = TEST_DIR / "image"

# Test split metadata location (MS COCO format catalogs, empty annotations)
TEST_METADATA_DIR: Path = Path(
    _os.environ.get(
        "FITLOOP_TEST_METADATA_DIR",
        r"C:\Users\Lenovo\Downloads\json_for_test"
    )
)

# Test ground-truth status:
# DeepFashion2 benchmark withholds test annotations; metadata contains catalogs only.
TEST_HAS_GROUND_TRUTH: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Workspace Data & Artifact Directories
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR: Path = ML_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"

MODELS_DIR: Path = ML_ROOT / "models"
OUTPUT_DIR: Path = ML_ROOT / "outputs"
OUTPUTS_DIR: Path = OUTPUT_DIR  # Alias for backward compatibility
SRC_DIR: Path = ML_ROOT / "src"
NOTEBOOKS_DIR: Path = ML_ROOT / "notebooks"

# Shared project-level directories
PROJECT_NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
PROJECT_METADATA_DIR: Path = PROJECT_ROOT / "data" / "metadata"
PROJECT_EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DeepFashion2 Categories & YOLO Detection Mapping
# ─────────────────────────────────────────────────────────────────────────────

DF2_CATEGORIES: dict[int, str] = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

DF2_TO_YOLO_CLASS_MAP: dict[int, int] = {
    1: 0,   # short sleeve top
    2: 1,   # long sleeve top
    3: 2,   # short sleeve outwear
    4: 3,   # long sleeve outwear
    5: 4,   # vest
    6: 5,   # sling
    7: 6,   # shorts
    8: 7,   # trousers
    9: 8,   # skirt
    10: 9,  # short sleeve dress
    11: 10, # long sleeve dress
    12: 11, # vest dress
    13: 12, # sling dress
}

YOLO_TO_DF2_CLASS_MAP: dict[int, int] = {v: k for k, v in DF2_TO_YOLO_CLASS_MAP.items()}
YOLO_CLASS_NAMES: list[str] = [
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Configuration Namespace & Validation
# ─────────────────────────────────────────────────────────────────────────────

class _Config:
    """Namespace holding all FitLoop ML configuration variables."""

    # Roots
    ML_ROOT = ML_ROOT
    PROJECT_ROOT = PROJECT_ROOT
    DATASET_ROOT = DATASET_ROOT

    # Dataset splits
    TRAIN_IMAGE_DIR = TRAIN_IMAGE_DIR
    TRAIN_ANNOTATION_DIR = TRAIN_ANNOTATION_DIR
    VALIDATION_IMAGE_DIR = VALIDATION_IMAGE_DIR
    VALIDATION_ANNOTATION_DIR = VALIDATION_ANNOTATION_DIR
    TEST_IMAGE_DIR = TEST_IMAGE_DIR
    TEST_METADATA_DIR = TEST_METADATA_DIR
    TEST_HAS_GROUND_TRUTH = TEST_HAS_GROUND_TRUTH

    # Workspace directories
    DATA_DIR = DATA_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    PROCESSED_DIR = PROCESSED_DATA_DIR  # Alias
    SPLITS_DIR = SPLITS_DIR
    MODELS_DIR = MODELS_DIR
    OUTPUT_DIR = OUTPUT_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    SRC_DIR = SRC_DIR
    NOTEBOOKS_DIR = NOTEBOOKS_DIR

    # Project-level shared directories
    PROJECT_NOTEBOOKS_DIR = PROJECT_NOTEBOOKS_DIR
    PROJECT_METADATA_DIR = PROJECT_METADATA_DIR
    PROJECT_EXPERIMENTS_DIR = PROJECT_EXPERIMENTS_DIR

    # Detection & category mappings
    DF2_CATEGORIES = DF2_CATEGORIES
    DF2_TO_YOLO_CLASS_MAP = DF2_TO_YOLO_CLASS_MAP
    YOLO_TO_DF2_CLASS_MAP = YOLO_TO_DF2_CLASS_MAP
    YOLO_CLASS_NAMES = YOLO_CLASS_NAMES

    def validate(self) -> bool:
        """
        Validate that all required dataset and workspace directories exist.
        Returns True if all required paths exist, False otherwise.
        """
        print("Checking Dataset Directories:")
        dataset_checks = [
            ("DATASET_ROOT", self.DATASET_ROOT, True),
            ("TRAIN_IMAGE_DIR", self.TRAIN_IMAGE_DIR, True),
            ("TRAIN_ANNOTATION_DIR", self.TRAIN_ANNOTATION_DIR, True),
            ("VALIDATION_IMAGE_DIR", self.VALIDATION_IMAGE_DIR, True),
            ("VALIDATION_ANNOTATION_DIR", self.VALIDATION_ANNOTATION_DIR, True),
            ("TEST_IMAGE_DIR", self.TEST_IMAGE_DIR, True),
            ("TEST_METADATA_DIR", self.TEST_METADATA_DIR, True),
        ]

        all_ok = True
        for name, path, required in dataset_checks:
            exists = path.exists()
            status = "[OK]    " if exists else ("[MISS]  " if required else "[OPT]   ")
            print(f"  {status} {name:<26} -> {path}")
            if required and not exists:
                all_ok = False

        print("\nChecking Ground-Truth Annotation Status:")
        print(f"  [INFO]   TEST_HAS_GROUND_TRUTH       -> {self.TEST_HAS_GROUND_TRUTH} (Withheld benchmark split)")

        print("\nChecking Workspace Directories:")
        workspace_checks = [
            ("RAW_DATA_DIR", self.RAW_DATA_DIR),
            ("PROCESSED_DATA_DIR", self.PROCESSED_DATA_DIR),
            ("OUTPUT_DIR", self.OUTPUT_DIR),
            ("SPLITS_DIR", self.SPLITS_DIR),
            ("MODELS_DIR", self.MODELS_DIR),
            ("SRC_DIR", self.SRC_DIR),
        ]
        for name, path in workspace_checks:
            exists = path.exists()
            status = "[OK]    " if exists else "[MISS]  "
            print(f"  {status} {name:<26} -> {path}")
            if not exists:
                all_ok = False

        return all_ok

    def __repr__(self) -> str:
        lines = ["FitLoop ML Config", "-" * 55]
        attrs = [
            "PROJECT_ROOT",
            "DATASET_ROOT",
            "TRAIN_IMAGE_DIR",
            "TRAIN_ANNOTATION_DIR",
            "VALIDATION_IMAGE_DIR",
            "VALIDATION_ANNOTATION_DIR",
            "TEST_IMAGE_DIR",
            "TEST_METADATA_DIR",
            "TEST_HAS_GROUND_TRUTH",
            "RAW_DATA_DIR",
            "PROCESSED_DATA_DIR",
            "OUTPUT_DIR",
        ]
        for a in attrs:
            v = getattr(self, a, None)
            lines.append(f"  {a:<28} {v}")
        return "\n".join(lines)


cfg = _Config()


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test Validation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(cfg)
    print("\n" + "=" * 55)
    print("DIRECTORY VALIDATION")
    print("=" * 55)
    ok = cfg.validate()
    print("=" * 55)
    print("[OK] All required directories verified successfully." if ok else "[ERROR] Some required directories are missing.")

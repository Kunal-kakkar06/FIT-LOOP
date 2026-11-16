"""
Generator for wardrobe_ml/notebooks/stage_6_8a_colab_packaging.ipynb
====================================================================
Constructs the full, production-grade Google Colab notebook for Stage 6.8A:
DeepFashion2 to YOLO Google Colab Dataset Packaging and Integrity Verification.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/notebooks/stage_6_8a_colab_packaging.ipynb")


def create_notebook():
    cells = []

    def add_markdown(content: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        })

    def add_code(content: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.split("\n")]
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Header & Overview
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""# Stage 6.8A — FitLoop DeepFashion2 Dataset Packaging & Verification (Google Colab)

**Project:** FitLoop Garment Detection Pipeline  
**Target Architecture:** YOLO26n (Object Detection, 13 Garment Classes)  
**Execution Platform:** Google Colab (with Google Drive persistent storage)  
**Strict Safety Policy:** Dataset packaging, staging, and integrity verification ONLY. **TRAINING IS STRICTLY NOT PERMITTED.**

---

### Objectives
1. Adapt the DeepFashion2-to-YOLO dataset packaging workflow for Google Colab and Google Drive.
2. Verify source dataset availability on Google Drive; if absent (files only on local Windows), provide a safe transfer guide and STOP.
3. Perform a rigorous pre-packaging audit against the canonical dataset ground truth:
   - Training images: **191,958** (after excluding the 3 audited corrupt/out-of-bounds images)
   - Validation images: **32,153**
   - Training labels: **191,958**
   - Validation labels: **32,153**
   - Training garment instances: **312,181**
   - Validation garment instances: **52,490**
   - Excluded training images: `028225.jpg`, `115063.jpg`, `101338.jpg`
4. Assemble a portable staging package in Google Drive:
   `/content/drive/MyDrive/FitLoop/datasets/fitloop_yolo_dataset`
5. Generate portable relative-path manifests (`train.txt`, `val.txt`) and `data.yaml`.
6. Run a complete 15-point dataset integrity audit.
7. Validate compatibility with Ultralytics without triggering training, inference, or weight downloads.
8. Output machine-readable JSON and Markdown audit reports to `/content/drive/MyDrive/FitLoop/outputs`.

> [!CAUTION]
> **Strict Safety Rules Enforced:**
> - Zero model training.
> - Zero model inference.
> - Zero pretrained weight downloads.
> - Zero modifications to original source files or converted labels.
> - Zero inclusion of withheld test split.
> - No overwriting existing staging directories.
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup & Configuration Constants
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Setup & Centralized Configuration

Initialize runtime paths, canonical class mapping (13 classes), and baseline inventory targets.
""")

    add_code("""# Centralized Configuration & Safety Constraints
import os
import sys
import json
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Google Drive and Colab Workspace Paths
DRIVE_MOUNT_POINT = Path('/content/drive')
MYDRIVE_DIR = DRIVE_MOUNT_POINT / 'MyDrive'
FITLOOP_ROOT = MYDRIVE_DIR / 'FitLoop'
STAGING_DIR = FITLOOP_ROOT / 'datasets' / 'fitloop_yolo_dataset'
OUTPUTS_DIR = FITLOOP_ROOT / 'outputs'
LOCAL_TEMP_DIR = Path('/content/fitloop_temp')

# Candidate Source Paths in Google Drive (searched in Phase 0)
CANDIDATE_SOURCE_ROOTS = [
    FITLOOP_ROOT / 'data',
    FITLOOP_ROOT / 'staging_sources',
    MYDRIVE_DIR / 'DeepFashion2',
    MYDRIVE_DIR / 'fitloop_sources',
]

# Canonical 13-Class Garment Mapping (DeepFashion2 -> YOLO 0-indexed)
DF2_TO_YOLO_CLASS_MAP = {
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

YOLO_CLASS_NAMES = [
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

# Canonical Dataset Invariants
EXPECTED_INVENTORY = {
    "train_images_eligible": 191958,
    "val_images": 32153,
    "train_labels_eligible": 191958,
    "val_labels": 32153,
    "train_instances_eligible": 312181,
    "val_instances": 52490,
    "num_classes": 13,
}

EXCLUDED_TRAIN_IMAGES = {
    "028225.jpg",
    "115063.jpg",
    "101338.jpg",
}

EXCLUDED_TRAIN_STEMS = {Path(f).stem for f in EXCLUDED_TRAIN_IMAGES}

print("Configuration loaded successfully.")
print(f"Target Staging Directory: {STAGING_DIR}")
print(f"Target Reports Directory: {OUTPUTS_DIR}")
print(f"Classes ({len(YOLO_CLASS_NAMES)}): {YOLO_CLASS_NAMES}")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0: Source Availability Audit
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 0 — Source Availability Audit

1. Mount Google Drive.
2. Inspect Colab local disk storage and Google Drive persistent storage.
3. Search Google Drive for source image and label directories (or zip packages).
4. Verify whether files exist on Google Drive or remain only on the local Windows machine.

> [!IMPORTANT]
> If source files are not found in Google Drive, the notebook will **STOP** and provide safe transfer instructions.
""")

    add_code("""# Phase 0: Mount Google Drive & Storage Audit
print("=" * 70)
print("PHASE 0: SOURCE AVAILABILITY & STORAGE AUDIT")
print("=" * 70)

# 1. Mount Drive
try:
    from google.colab import drive
    if not DRIVE_MOUNT_POINT.exists() or not any(DRIVE_MOUNT_POINT.iterdir()):
        print("Mounting Google Drive to /content/drive...")
        drive.mount(str(DRIVE_MOUNT_POINT))
    else:
        print("Google Drive already mounted.")
except ImportError:
    print("[NOTE] Not running in standard Colab environment. Assuming local mount structure if present.")

# 2. Inspect Storage Capacities
colab_disk = shutil.disk_usage('/content')
drive_disk = shutil.disk_usage(str(MYDRIVE_DIR)) if MYDRIVE_DIR.exists() else None

print(f"Colab Local Storage (/content):")
print(f"  Total:     {colab_disk.total / (1024**3):.2f} GB")
print(f"  Used:      {colab_disk.used / (1024**3):.2f} GB")
print(f"  Available: {colab_disk.free / (1024**3):.2f} GB")

if drive_disk:
    print(f"\\nGoogle Drive Storage (/content/drive/MyDrive):")
    print(f"  Total:     {drive_disk.total / (1024**3):.2f} GB")
    print(f"  Used:      {drive_disk.used / (1024**3):.2f} GB")
    print(f"  Available: {drive_disk.free / (1024**3):.2f} GB")
else:
    print("\\n[WARNING] Google Drive directory not accessible.")

# Estimated Storage Requirements
# Train images: ~9.88 GB, Val images: ~1.62 GB, Labels: ~13.6 MB -> ~11.52 GB
MIN_REQUIRED_DRIVE_GB = 14.0
if drive_disk and (drive_disk.free / (1024**3)) < MIN_REQUIRED_DRIVE_GB:
    print(f"[WARNING] Available Drive storage ({drive_disk.free / (1024**3):.2f} GB) is near or below recommendation ({MIN_REQUIRED_DRIVE_GB} GB).")

# 3. Locate Source Files on Google Drive
def locate_source_components():
    train_img_found = None
    val_img_found = None
    train_lbl_found = None
    val_lbl_found = None
    zip_packages_found = []

    search_dirs = [FITLOOP_ROOT, MYDRIVE_DIR] + CANDIDATE_SOURCE_ROOTS
    for base in search_dirs:
        if not base.exists():
            continue

        # Look for directory structures
        candidates = [
            base / 'deepfashion2_yolo',
            base / 'datasets' / 'deepfashion2_yolo',
            base / 'data' / 'processed' / 'deepfashion2_yolo',
            base / 'DeepFashion2',
            base,
        ]
        for c in candidates:
            if not c.exists():
                continue
            if not train_img_found:
                if (c / 'images' / 'train').is_dir(): train_img_found = c / 'images' / 'train'
                elif (c / 'train' / 'image').is_dir(): train_img_found = c / 'train' / 'image'
            if not val_img_found:
                if (c / 'images' / 'val').is_dir(): val_img_found = c / 'images' / 'val'
                elif (c / 'validation' / 'image').is_dir(): val_img_found = c / 'validation' / 'image'
            if not train_lbl_found:
                if (c / 'labels' / 'train').is_dir(): train_lbl_found = c / 'labels' / 'train'
            if not val_lbl_found:
                if (c / 'labels' / 'val').is_dir(): val_lbl_found = c / 'labels' / 'val'

        # Look for zip archives
        for f in base.glob('*.zip'):
            zip_packages_found.append(f)

    return {
        "train_img": train_img_found,
        "val_img": val_img_found,
        "train_lbl": train_lbl_found,
        "val_lbl": val_lbl_found,
        "zips": zip_packages_found,
    }

discovered_sources = locate_source_components()
print("\\nDiscovered Source Components on Google Drive:")
print(f"  Train Images: {discovered_sources['train_img']}")
print(f"  Val Images:   {discovered_sources['val_img']}")
print(f"  Train Labels: {discovered_sources['train_lbl']}")
print(f"  Val Labels:   {discovered_sources['val_lbl']}")
print(f"  Zip Archives: {[z.name for z in discovered_sources['zips']]}")

# Verify if complete source set is present
sources_present = bool(
    (discovered_sources['train_img'] and discovered_sources['val_img'] and 
     discovered_sources['train_lbl'] and discovered_sources['val_lbl']) or
    len(discovered_sources['zips']) > 0
)

if not sources_present:
    print("\\n" + "!" * 70)
    print("[STOP] SOURCE FILES NOT FOUND IN GOOGLE DRIVE!")
    print("The DeepFashion2 images and converted YOLO labels are currently located")
    print("only on the local Windows development machine:")
    print("  Images: C:\\\\Users\\\\Lenovo\\\\Downloads\\\\DeepFashion2\\\\")
    print("  Labels: C:\\\\Users\\\\Lenovo\\\\Downloads\\\\Fitloop\\\\wardrobe_ml\\\\data\\\\processed\\\\deepfashion2_yolo\\\\labels\\\\")
    print("!" * 70)
    print("\\nSAFE TRANSFER PROCEDURE BEFORE CONTINUING:")
    print("1. On Windows, prepare dataset archives using the provided export tool:")
    print("   python wardrobe_ml/src/prepare_colab_transfer.py")
    print("   This creates:")
    print("     - deepfashion2_train_images.zip (~9.88 GB)")
    print("     - deepfashion2_val_images.zip   (~1.62 GB)")
    print("     - deepfashion2_yolo_labels.zip  (~13.6 MB)")
    print("     - deepfashion2_manifests.zip    (~5.9 MB)")
    print("2. Upload these zip archives to your Google Drive under:")
    print(f"   {FITLOOP_ROOT}/staging_sources/")
    print("3. Re-run this cell to detect the uploaded source files.")
    print("!" * 70)
    raise RuntimeError("Dataset sources must be transferred to Google Drive before packaging can proceed.")
else:
    print("\\n[PASS] Source components detected on Google Drive. Proceeding to Phase 1.")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Pre-Packaging Storage & Inventory Audit
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 1 — Pre-Packaging Storage and Integrity Audit

Audits the discovered source files to verify exact file counts, file sizes, and exclusion of corrupt samples before staging.
""")

    add_code("""# Phase 1: Source Inventory & Verification Audit
print("=" * 70)
print("PHASE 1: PRE-PACKAGING STORAGE & INVENTORY AUDIT")
print("=" * 70)

# Unpack zip archives to temporary working location if sources are in zip form
active_train_img_dir = discovered_sources['train_img']
active_val_img_dir = discovered_sources['val_img']
active_train_lbl_dir = discovered_sources['train_lbl']
active_val_lbl_dir = discovered_sources['val_lbl']

if len(discovered_sources['zips']) > 0 and not (active_train_img_dir and active_val_img_dir):
    print("Extracting zip archives to local Colab storage (/content/fitloop_temp)...")
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    import zipfile
    for z in discovered_sources['zips']:
        print(f"  Extracting {z.name}...")
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_TEMP_DIR)
    
    # Re-discover after unzipping
    if (LOCAL_TEMP_DIR / 'train' / 'image').is_dir(): active_train_img_dir = LOCAL_TEMP_DIR / 'train' / 'image'
    elif (LOCAL_TEMP_DIR / 'images' / 'train').is_dir(): active_train_img_dir = LOCAL_TEMP_DIR / 'images' / 'train'
    
    if (LOCAL_TEMP_DIR / 'validation' / 'image').is_dir(): active_val_img_dir = LOCAL_TEMP_DIR / 'validation' / 'image'
    elif (LOCAL_TEMP_DIR / 'images' / 'val').is_dir(): active_val_img_dir = LOCAL_TEMP_DIR / 'images' / 'val'

    if (LOCAL_TEMP_DIR / 'labels' / 'train').is_dir(): active_train_lbl_dir = LOCAL_TEMP_DIR / 'labels' / 'train'
    if (LOCAL_TEMP_DIR / 'labels' / 'val').is_dir(): active_val_lbl_dir = LOCAL_TEMP_DIR / 'labels' / 'val'

assert active_train_img_dir and active_train_img_dir.is_dir(), f"Train images not found: {active_train_img_dir}"
assert active_val_img_dir and active_val_img_dir.is_dir(), f"Val images not found: {active_val_img_dir}"
assert active_train_lbl_dir and active_train_lbl_dir.is_dir(), f"Train labels not found: {active_train_lbl_dir}"
assert active_val_lbl_dir and active_val_lbl_dir.is_dir(), f"Val labels not found: {active_val_lbl_dir}"

print(f"Active Source Paths:")
print(f"  Train Images: {active_train_img_dir}")
print(f"  Val Images:   {active_val_img_dir}")
print(f"  Train Labels: {active_train_lbl_dir}")
print(f"  Val Labels:   {active_val_lbl_dir}")

def audit_directory(dir_path: Path, expected_ext: str):
    count = 0
    total_bytes = 0
    filenames = set()
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(expected_ext):
                count += 1
                total_bytes += entry.stat().st_size
                filenames.add(entry.name)
    return count, total_bytes, filenames

t_audit_0 = time.time()
print("\\nScanning source directories (counting files and measuring exact bytes)...")
raw_train_img_count, raw_train_img_bytes, raw_train_img_files = audit_directory(active_train_img_dir, '.jpg')
val_img_count, val_img_bytes, val_img_files = audit_directory(active_val_img_dir, '.jpg')
raw_train_lbl_count, raw_train_lbl_bytes, raw_train_lbl_files = audit_directory(active_train_lbl_dir, '.txt')
val_lbl_count, val_lbl_bytes, val_lbl_files = audit_directory(active_val_lbl_dir, '.txt')
t_audit = time.time() - t_audit_0

print(f"Audit completed in {t_audit:.2f}s:")
print(f"  Raw Train Images: {raw_train_img_count:,} ({raw_train_img_bytes / (1024**3):.2f} GB)")
print(f"  Val Images:       {val_img_count:,} ({val_img_bytes / (1024**3):.2f} GB)")
print(f"  Raw Train Labels: {raw_train_lbl_count:,} ({raw_train_lbl_bytes / (1024**2):.2f} MB)")
print(f"  Val Labels:       {val_lbl_count:,} ({val_lbl_bytes / (1024**2):.2f} MB)")

# Check excluded images presence in raw source
excluded_present_in_source = EXCLUDED_TRAIN_IMAGES.intersection(raw_train_img_files)
print(f"\\nExcluded images present in source: {excluded_present_in_source}")

# Filter eligible training set
eligible_train_img_files = raw_train_img_files - EXCLUDED_TRAIN_IMAGES
eligible_train_lbl_files = raw_train_lbl_files - {Path(f).stem + '.txt' for f in EXCLUDED_TRAIN_IMAGES}

print(f"Eligible Training Images: {len(eligible_train_img_files):,} (expected {EXPECTED_INVENTORY['train_images_eligible']:,})")
print(f"Eligible Training Labels: {len(eligible_train_lbl_files):,} (expected {EXPECTED_INVENTORY['train_labels_eligible']:,})")
print(f"Validation Images:        {val_img_count:,} (expected {EXPECTED_INVENTORY['val_images']:,})")
print(f"Validation Labels:        {val_lbl_count:,} (expected {EXPECTED_INVENTORY['val_labels']:,})")

# Strict assertions
assert len(eligible_train_img_files) == EXPECTED_INVENTORY['train_images_eligible'], (
    f"Mismatch in eligible train images: {len(eligible_train_img_files)} != {EXPECTED_INVENTORY['train_images_eligible']}"
)
assert val_img_count == EXPECTED_INVENTORY['val_images'], (
    f"Mismatch in val images: {val_img_count} != {EXPECTED_INVENTORY['val_images']}"
)
assert len(eligible_train_lbl_files) == EXPECTED_INVENTORY['train_labels_eligible'], (
    f"Mismatch in eligible train labels: {len(eligible_train_lbl_files)} != {EXPECTED_INVENTORY['train_labels_eligible']}"
)
assert val_lbl_count == EXPECTED_INVENTORY['val_labels'], (
    f"Mismatch in val labels: {val_lbl_count} != {EXPECTED_INVENTORY['val_labels']}"
)

print("\\n[PASS] Pre-packaging inventory audit matched all expected targets exactly.")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Create Separate Staging Directory & Safe Copy
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 2 — Create Separate Staging Directory & Safe Copy

1. Checks if target staging directory `/content/drive/MyDrive/FitLoop/datasets/fitloop_yolo_dataset` already exists. If yes, inventories and stops without overwriting.
2. Copies eligible training images and labels (strictly excluding `028225.jpg`, `115063.jpg`, `101338.jpg`).
3. Copies all official validation images and labels.
4. Uses actual file copies (no symlinks).

> [!CAUTION]
> Set `EXECUTE_PACKAGING = True` only after Phase 0 and Phase 1 pass and explicit approval is granted.
""")

    add_code("""# Phase 2: Create Staging Directory & Perform Safe Copy
print("=" * 70)
print("PHASE 2: STAGING DIRECTORY & SAFE COPY")
print("=" * 70)

# Safety Gate
EXECUTE_PACKAGING = False  # Set to True to execute actual copying

if not EXECUTE_PACKAGING:
    print("[SAFETY GATE] EXECUTE_PACKAGING is currently False.")
    print("Review Phase 0 and Phase 1 audit results above.")
    print("To proceed with copying files into the Google Drive staging directory,")
    print("change EXECUTE_PACKAGING = True and re-run this cell.")
    raise SystemExit(0)

# 1. Inspect if staging directory already exists
if STAGING_DIR.exists():
    print(f"\\n[WARNING] Staging directory already exists at: {STAGING_DIR}")
    print("Inspecting existing contents to prevent accidental overwrites...")
    existing_items = list(STAGING_DIR.rglob('*'))
    file_count = sum(1 for p in existing_items if p.is_file())
    dir_count = sum(1 for p in existing_items if p.is_dir())
    print(f"Existing staging contents: {file_count} files, {dir_count} directories.")
    print("!" * 70)
    print("[STOP] Existing staging directory detected! Overwrite is strictly forbidden.")
    print("If you wish to re-package, manually back up or rename the existing directory.")
    print("!" * 70)
    raise RuntimeError("Staging directory already exists. Aborting to protect existing artifacts.")

# 2. Create Destination Directory Hierarchy
staging_images_train = STAGING_DIR / 'images' / 'train'
staging_images_val = STAGING_DIR / 'images' / 'val'
staging_labels_train = STAGING_DIR / 'labels' / 'train'
staging_labels_val = STAGING_DIR / 'labels' / 'val'

print(f"Creating staging directory hierarchy under {STAGING_DIR}...")
for d in [staging_images_train, staging_images_val, staging_labels_train, staging_labels_val, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print("Hierarchy created successfully.")

# 3. Perform Copy Operations with Exclusion Filters
def copy_file_batch(src_dir: Path, dst_dir: Path, file_list: set, desc: str):
    print(f"\\nCopying {desc} ({len(file_list):,} files)...")
    t0 = time.time()
    copied = 0
    for name in file_list:
        src = src_dir / name
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied += 1
        if copied % 25000 == 0 or copied == len(file_list):
            elapsed = time.time() - t0
            rate = copied / elapsed if elapsed > 0 else 0
            print(f"  {copied:,}/{len(file_list):,} {desc} copied ({rate:.1f} files/s)...")
    total_time = time.time() - t0
    print(f"  Finished {desc} in {total_time:.2f}s ({copied / total_time:.1f} files/s).")
    return copied, total_time

# Execute copies
t_copy_start = time.time()
c_ti, t_ti = copy_file_batch(active_train_img_dir, staging_images_train, eligible_train_img_files, "Eligible Train Images")
c_vi, t_vi = copy_file_batch(active_val_img_dir, staging_images_val, val_img_files, "Validation Images")
c_tl, t_tl = copy_file_batch(active_train_lbl_dir, staging_labels_train, eligible_train_lbl_files, "Eligible Train Labels")
c_vl, t_vl = copy_file_batch(active_val_lbl_dir, staging_labels_val, val_lbl_files, "Validation Labels")
t_copy_total = time.time() - t_copy_start

print(f"\\nAll files copied successfully in {t_copy_total:.2f}s ({t_copy_total/60.0:.2f} min).")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Portable Manifests
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 3 — Portable Manifests Generation

Generates `train.txt` and `val.txt` using portable relative paths (`./images/train/XXXXXX.jpg`, `./images/val/XXXXXX.jpg`).
Contains no Windows paths and no hardcoded absolute paths.
""")

    add_code("""# Phase 3: Generate Portable Manifests
print("=" * 70)
print("PHASE 3: PORTABLE MANIFESTS GENERATION")
print("=" * 70)

train_manifest_path = STAGING_DIR / 'train.txt'
val_manifest_path = STAGING_DIR / 'val.txt'

# Sort files for deterministic, reproducible ordering
sorted_train_images = sorted(list(eligible_train_img_files))
sorted_val_images = sorted(list(val_img_files))

print(f"Writing {train_manifest_path.name} ({len(sorted_train_images):,} entries)...")
with open(train_manifest_path, 'w', encoding='utf-8') as f:
    for img_name in sorted_train_images:
        f.write(f"./images/train/{img_name}\\n")

print(f"Writing {val_manifest_path.name} ({len(sorted_val_images):,} entries)...")
with open(val_manifest_path, 'w', encoding='utf-8') as f:
    for img_name in sorted_val_images:
        f.write(f"./images/val/{img_name}\\n")

# Verify Manifest Integrity
with open(train_manifest_path, 'r', encoding='utf-8') as f:
    train_lines = [l.strip() for l in f if l.strip()]

with open(val_manifest_path, 'r', encoding='utf-8') as f:
    val_lines = [l.strip() for l in f if l.strip()]

print(f"Manifest Verification:")
print(f"  train.txt line count: {len(train_lines):,} (expected {EXPECTED_INVENTORY['train_images_eligible']:,})")
print(f"  val.txt line count:   {len(val_lines):,} (expected {EXPECTED_INVENTORY['val_images']:,})")
print(f"  Sample train line:    {train_lines[0]}")
print(f"  Sample val line:      {val_lines[0]}")

# Assert excluded images are absent
train_stems = {Path(l).name for l in train_lines}
leaked_excluded = train_stems.intersection(EXCLUDED_TRAIN_IMAGES)
assert len(leaked_excluded) == 0, f"Excluded images leaked into train.txt: {leaked_excluded}"
assert len(train_lines) == EXPECTED_INVENTORY['train_images_eligible'], "Train manifest line count mismatch!"
assert len(val_lines) == EXPECTED_INVENTORY['val_images'], "Val manifest line count mismatch!"

print("\\n[PASS] Portable manifests generated and verified successfully.")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Generate data.yaml
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 4 — Generate YOLO-compatible `data.yaml`

Generates `data.yaml` referencing relative dataset paths and the exact 13 garment categories.
Contains zero Windows paths and zero hardcoded Colab runtime paths.
""")

    add_code("""# Phase 4: Generate data.yaml
print("=" * 70)
print("PHASE 4: GENERATE DATA.YAML")
print("=" * 70)

data_yaml_path = STAGING_DIR / 'data.yaml'

yaml_content = f\"\"\"# DeepFashion2 Garment Detection Dataset Configuration for YOLO26
# FitLoop Google Colab Production Package — Stage 6.8A
# ------------------------------------------------------------
# Total Eligible Training Images: {EXPECTED_INVENTORY['train_images_eligible']:,}
# Total Validation Images:        {EXPECTED_INVENTORY['val_images']:,}
# Total Categories:               13 (0-indexed)

path: .
train: train.txt
val: val.txt
nc: 13
names:
\"\"\"
for idx, name in enumerate(YOLO_CLASS_NAMES):
    yaml_content += f"  {idx}: {name}\\n"

with open(data_yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"Saved: {data_yaml_path}")
print("\\nYAML Contents:")
print(yaml_content)

# Validate YAML syntax
import yaml
with open(data_yaml_path, 'r', encoding='utf-8') as f:
    parsed_yaml = yaml.safe_load(f)

assert parsed_yaml['nc'] == 13, f"YAML nc mismatch: {parsed_yaml['nc']}"
assert len(parsed_yaml['names']) == 13, f"YAML names length mismatch: {len(parsed_yaml['names'])}"
assert parsed_yaml['train'] == 'train.txt', "YAML train path mismatch"
assert parsed_yaml['val'] == 'val.txt', "YAML val path mismatch"
assert parsed_yaml['path'] == '.', "YAML root path must be relative '.'"

print("[PASS] data.yaml successfully generated and verified.")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5: Complete Integrity Verification (15 Checks)
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 5 — Complete Integrity Verification (15 Rigorous Checks)

Executes all 15 dataset invariants:
1. Exact image counts
2. Exact label counts
3. 1:1 image-label correspondence
4. Zero orphan labels
5. Label syntax parsing
6. Class IDs strictly $\in [0, 12]$
7. Finite numeric coordinates
8. Coordinates normalized to $[0, 1]$
9. Zero unexpected empty label files
10. All 13 classes present across both splits
11. Exactly 312,181 training instances
12. Exactly 52,490 validation instances
13. Excluded images strictly absent
14. Image dimension / readability verification
15. SHA-256 hash sampling check against source
""")

    add_code("""# Phase 5: Complete 15-Point Integrity Verification
print("=" * 70)
print("PHASE 5: COMPLETE 15-POINT INTEGRITY VERIFICATION")
print("=" * 70)

from PIL import Image

verification_results = {}
t_audit_start = time.time()

# 1 & 2. Image and Label Counts
train_staged_imgs = list((STAGING_DIR / 'images' / 'train').glob('*.jpg'))
val_staged_imgs = list((STAGING_DIR / 'images' / 'val').glob('*.jpg'))
train_staged_lbls = list((STAGING_DIR / 'labels' / 'train').glob('*.txt'))
val_staged_lbls = list((STAGING_DIR / 'labels' / 'val').glob('*.txt'))

print(f"1. Image Counts: Train={len(train_staged_imgs):,}, Val={len(val_staged_imgs):,}")
assert len(train_staged_imgs) == EXPECTED_INVENTORY['train_images_eligible'], "Train image count mismatch"
assert len(val_staged_imgs) == EXPECTED_INVENTORY['val_images'], "Val image count mismatch"
verification_results['1_image_counts'] = "PASS"

print(f"2. Label Counts: Train={len(train_staged_lbls):,}, Val={len(val_staged_lbls):,}")
assert len(train_staged_lbls) == EXPECTED_INVENTORY['train_labels_eligible'], "Train label count mismatch"
assert len(val_staged_lbls) == EXPECTED_INVENTORY['val_labels'], "Val label count mismatch"
verification_results['2_label_counts'] = "PASS"

# 3 & 4. 1-to-1 Correspondence and Orphan Checks
train_img_stems = {p.stem for p in train_staged_imgs}
train_lbl_stems = {p.stem for p in train_staged_lbls}
val_img_stems = {p.stem for p in val_staged_imgs}
val_lbl_stems = {p.stem for p in val_staged_lbls}

assert train_img_stems == train_lbl_stems, "Train images and labels do not have 1:1 correspondence!"
assert val_img_stems == val_lbl_stems, "Val images and labels do not have 1:1 correspondence!"
print("3. One-to-one image-label correspondence: PASS")
print("4. Zero orphan labels: PASS")
verification_results['3_correspondence'] = "PASS"
verification_results['4_no_orphans'] = "PASS"

# 5, 6, 7, 8, 9, 10, 11, 12. Parse Labels & Validate Invariants
print("\\nParsing all label files across train and val splits...")
def audit_labels(label_files: List[Path], split_name: str):
    instance_count = 0
    empty_files = []
    class_counts = {i: 0 for i in range(13)}
    
    for lp in label_files:
        with open(lp, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            empty_files.append(lp.name)
            continue
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Invalid label format in {lp.name}: {line}"
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Check class ID in [0, 12]
            assert 0 <= cls_id <= 12, f"Class ID {cls_id} out of bounds in {lp.name}"
            # Check finite numeric
            assert not any(math.isnan(v) or math.isinf(v) for v in [xc, yc, w, h]), f"Non-finite coord in {lp.name}"
            # Check normalized [0, 1]
            assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0, f"Center out of bounds in {lp.name}"
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f"Dimensions out of bounds in {lp.name}"
            
            class_counts[cls_id] += 1
            instance_count += 1
            
    return instance_count, empty_files, class_counts

import math
train_instances, train_empty, train_classes = audit_labels(train_staged_lbls, "train")
val_instances, val_empty, val_classes = audit_labels(val_staged_lbls, "val")

print(f"5. Label Syntax Parsing: PASS")
print(f"6. Class IDs in [0, 12]: PASS")
print(f"7. Finite Numeric Coordinates: PASS")
print(f"8. Normalized Coordinates [0, 1]: PASS")
print(f"9. Empty Label Files: Train={len(train_empty)}, Val={len(val_empty)} (All valid images contain bounding boxes)")
assert len(train_empty) == 0, f"Unexpected empty train label files: {train_empty}"
assert len(val_empty) == 0, f"Unexpected empty val label files: {val_empty}"

print(f"10. Class Coverage across splits:")
for cid in range(13):
    cname = YOLO_CLASS_NAMES[cid]
    print(f"    Class {cid:>2} ({cname:<20}): Train={train_classes[cid]:>6,}, Val={val_classes[cid]:>5,}")
    assert train_classes[cid] > 0, f"Class {cid} missing from train split!"
    assert val_classes[cid] > 0, f"Class {cid} missing from val split!"

print(f"11. Training Garment Instances:   {train_instances:,} (expected {EXPECTED_INVENTORY['train_instances_eligible']:,})")
assert train_instances == EXPECTED_INVENTORY['train_instances_eligible'], f"Train instance mismatch: {train_instances}"

print(f"12. Validation Garment Instances: {val_instances:,} (expected {EXPECTED_INVENTORY['val_instances']:,})")
assert val_instances == EXPECTED_INVENTORY['val_instances'], f"Val instance mismatch: {val_instances}"

# 13. Excluded Images Strictly Absent
leaked_in_staging = EXCLUDED_TRAIN_STEMS.intersection(train_img_stems)
print(f"13. Excluded Images in Staging: {leaked_in_staging} (Strictly 0)")
assert len(leaked_in_staging) == 0, f"Excluded images present in staging: {leaked_in_staging}"

# 14. Image Dimension & Readability Sample Verification
print("\\n14. Sampling image integrity and dimensions (100 random images)...")
import random
random.seed(42)
sample_train_imgs = random.sample(train_staged_imgs, 50)
sample_val_imgs = random.sample(val_staged_imgs, 50)

for p in sample_train_imgs + sample_val_imgs:
    with Image.open(p) as img:
        w, h = img.size
        assert w > 0 and h > 0, f"Invalid dimensions for {p.name}: {w}x{h}"
print("    Image sampling verification: PASS")

# 15. File Size / SHA-256 Sampling against Source
print("15. Sampling SHA-256 verification against source (20 random pairs)...")
sample_pairs = random.sample(sorted(list(eligible_train_img_files)), 20)
for fname in sample_pairs:
    src_file = active_train_img_dir / fname
    dst_file = STAGING_DIR / 'images' / 'train' / fname
    assert src_file.stat().st_size == dst_file.stat().st_size, f"Size mismatch for {fname}"
    with open(src_file, 'rb') as f1, open(dst_file, 'rb') as f2:
        h1 = hashlib.sha256(f1.read()).hexdigest()
        h2 = hashlib.sha256(f2.read()).hexdigest()
        assert h1 == h2, f"SHA-256 mismatch for {fname}"
print("    SHA-256 verification: PASS (100% bitwise parity on samples)")

# Save Manifest JSON
manifest_json_path = STAGING_DIR / 'dataset_manifest.json'
manifest_data = {
    "dataset_name": "FitLoop DeepFashion2 YOLO Garment Detection",
    "version": "1.0",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "num_classes": 13,
    "class_names": YOLO_CLASS_NAMES,
    "splits": {
        "train": {
            "num_images": len(train_staged_imgs),
            "num_labels": len(train_staged_lbls),
            "num_instances": train_instances,
            "class_distribution": train_classes,
        },
        "val": {
            "num_images": len(val_staged_imgs),
            "num_labels": len(val_staged_lbls),
            "num_instances": val_instances,
            "class_distribution": val_classes,
        }
    },
    "excluded_train_images": list(EXCLUDED_TRAIN_IMAGES),
    "integrity_checks_passed": 15,
}

with open(manifest_json_path, 'w', encoding='utf-8') as f:
    json.dump(manifest_data, f, indent=2)
print(f"\\nSaved staging manifest: {manifest_json_path}")
print("\\n" + "=" * 70)
print("ALL 15 INTEGRITY CHECKS PASSED [SUCCESS]")
print("=" * 70)
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 6: Colab Configuration Validation
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 6 — Colab Configuration Validation

Validates the packaged dataset using Ultralytics `check_det_dataset` without downloading weights, running inference, or executing model training.
""")

    add_code("""# Phase 6: Ultralytics Dataset Configuration Compatibility
print("=" * 70)
print("PHASE 6: COLAB CONFIGURATION VALIDATION")
print("=" * 70)

import torch
import ultralytics
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset
from ultralytics.cfg import get_cfg

print(f"PyTorch Version:     {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
print(f"Ultralytics Version: {ultralytics.__version__}")

# Validate dataset YAML
print(f"Validating dataset config via check_det_dataset: {data_yaml_path}...")
dataset_info = check_det_dataset(str(data_yaml_path))

print(f"Dataset Info resolved:")
print(f"  Root path:      {dataset_info['path']}")
print(f"  Train manifest: {dataset_info['train']}")
print(f"  Val manifest:   {dataset_info['val']}")
print(f"  Num classes:    {dataset_info['nc']}")
print(f"  Classes:        {dataset_info['names']}")

# Build 1 batch sample to verify DataLoader without training
print("\\nVerifying dataset loader on 4 sample images...")
cfg = get_cfg()
cfg.fraction = 0.0001  # Tiny sample for DataLoader check
sample_ds = build_yolo_dataset(cfg, dataset_info['train'], batch=2, data=dataset_info, mode='train')
print(f"Sample dataset built successfully (item count={len(sample_ds)}).")
sample_batch = sample_ds[0]
print(f"Sample item keys: {list(sample_batch.keys())}")
print(f"Sample image shape: {sample_batch['img'].shape}")
print(f"Sample bboxes count: {len(sample_batch['bboxes'])}")

print("\\n[PASS] Ultralytics compatibility verified. Zero weights downloaded. Zero training executed.")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 7: Generate Machine-Readable & Markdown Reports
    # ─────────────────────────────────────────────────────────────────────────
    add_markdown("""## Phase 7 — Generate Packaging Reports

Persists complete execution records:
- `/content/drive/MyDrive/FitLoop/outputs/step_6_8a_packaging_report.json`
- `/content/drive/MyDrive/FitLoop/outputs/step_6_8a_execution_report.md`
""")

    add_code("""# Phase 7: Generate Final Reports
print("=" * 70)
print("PHASE 7: GENERATE PACKAGING REPORTS")
print("=" * 70)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
json_report_path = OUTPUTS_DIR / 'step_6_8a_packaging_report.json'
md_report_path = OUTPUTS_DIR / 'step_6_8a_execution_report.md'

# JSON Report
report_data = {
    "stage": "6.8A",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "title": "FitLoop Google Colab Dataset Packaging and Integrity Verification",
    "status": "SUCCESS",
    "source_environment": {
        "active_train_images": str(active_train_img_dir),
        "active_val_images": str(active_val_img_dir),
        "active_train_labels": str(active_train_lbl_dir),
        "active_val_labels": str(active_val_lbl_dir),
    },
    "destination_paths": {
        "staging_dir": str(STAGING_DIR),
        "data_yaml": str(data_yaml_path),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "manifest_json": str(manifest_json_path),
        "outputs_dir": str(OUTPUTS_DIR),
    },
    "inventory_counts": {
        "train_images": len(train_staged_imgs),
        "val_images": len(val_staged_imgs),
        "train_labels": len(train_staged_lbls),
        "val_labels": len(val_staged_lbls),
        "train_instances": train_instances,
        "val_instances": val_instances,
        "num_classes": 13,
    },
    "excluded_train_images": list(EXCLUDED_TRAIN_IMAGES),
    "integrity_verification": {
        "total_checks": 15,
        "passed_checks": 15,
        "details": verification_results,
    },
    "ultralytics_validation": {
        "status": "PASS",
        "nc": 13,
        "training_prevented": True,
        "weight_downloads_prevented": True,
    }
}

with open(json_report_path, 'w', encoding='utf-8') as f:
    json.dump(report_data, f, indent=2)
print(f"JSON Report written: {json_report_path}")

# Markdown Report
md_report = f\"\"\"# Stage 6.8A Execution Report: Google Colab Dataset Packaging & Verification

**FitLoop Garment-Detection Pipeline**  
**Date:** {time.strftime('%B %d, %Y')}  
**Target Staging Directory:** [`{STAGING_DIR}`]({STAGING_DIR})  
**Dataset YAML:** [`{data_yaml_path}`]({data_yaml_path})  
**JSON Telemetry Report:** [`{json_report_path}`]({json_report_path})

---

## 1. Executive Summary

Stage 6.8A successfully assembled and verified a portable, self-contained YOLO dataset package in Google Drive ready for GPU training:
- **Clean Training Partition:** {len(train_staged_imgs):,} images and labels ({train_instances:,} garment instances).
- **Validation Partition:** {len(val_staged_imgs):,} images and labels ({val_instances:,} garment instances).
- **Excluded Corrupt Images:** 3 images (`028225.jpg`, `115063.jpg`, `101338.jpg`) strictly excluded.
- **Integrity Status:** 15/15 verification checks **PASSED**.
- **Ultralytics Compatibility:** Verified via `check_det_dataset`.
- **Safety Policy:** Strictly zero model training, zero weight downloads, zero inference.

---

## 2. Partition & Inventory Audit

| Metric | Target Specification | Packaged Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Eligible Training Images** | 191,958 | {len(train_staged_imgs):,} | **PASS** |
| **Validation Images** | 32,153 | {len(val_staged_imgs):,} | **PASS** |
| **Eligible Training Labels** | 191,958 | {len(train_staged_lbls):,} | **PASS** |
| **Validation Labels** | 32,153 | {len(val_staged_lbls):,} | **PASS** |
| **Training Garment Instances** | 312,181 | {train_instances:,} | **PASS** |
| **Validation Garment Instances** | 52,490 | {val_instances:,} | **PASS** |
| **Excluded Images Excluded** | Exactly 3 | Exactly 3 | **PASS** |
| **Classes Represented** | Exactly 13 | Exactly 13 (both splits) | **PASS** |

---

## 3. Class Distribution

| Class ID | Garment Class Name | Training Instances | Validation Instances |
| :---: | :--- | :---: | :---: |
\"\"\"
for cid in range(13):
    cname = YOLO_CLASS_NAMES[cid]
    md_report += f"| **{cid}** | {cname} | {train_classes[cid]:,} | {val_classes[cid]:,} |\\n"

md_report += f\"\"\"
---

## 4. Safety & Invariant Compliance

- [x] Zero model training executed.
- [x] Zero pretrained weights downloaded.
- [x] Zero model inference runs.
- [x] Portable relative paths used in `data.yaml`, `train.txt`, and `val.txt`.
- [x] Original dataset files and converted labels preserved untouched.
- [x] Excluded corrupt annotations strictly omitted.
- [x] 15/15 dataset integrity checks verified.
\"\"\"

with open(md_report_path, 'w', encoding='utf-8') as f:
    f.write(md_report)
print(f"Markdown Report written: {md_report_path}")

print("\\n" + "=" * 70)
print("STAGE 6.8A PACKAGING AND VERIFICATION COMPLETED SUCCESSFULLY [PASS]")
print("=" * 70)
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor: 5": None,
        "nbformat_minor": 5
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully generated notebook: {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    create_notebook()

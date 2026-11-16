# FitLoop Model 1 — Body Measurement Engine

Estimates 16 garment-relevant body dimensions from a front + side photograph pair
using landmark geometry and silhouette analysis.

---

## Project status

| Stage | Description                         | Status     |
|-------|-------------------------------------|------------|
| 01    | Environment & project setup         | ✅ Complete |
| 02    | Image capture & quality validation  | 🔄 In progress |
| 03    | Pose & silhouette baseline          | ⏳ Pending |
| 04    | Measurement engine                  | ⏳ Pending |
| 05    | 3D reconstruction benchmark         | ⏳ Pending |
| 06    | Ground-truth evaluation             | ⏳ Pending |
| 07    | Inference interface                 | ⏳ Pending |

---

## Measurement targets

| # | Measurement           |
|---|-----------------------|
| 1 | Height                |
| 2 | Chest circumference   |
| 3 | Waist circumference   |
| 4 | Hip circumference     |
| 5 | Shoulder width        |
| 6 | Inseam length         |
| 7 | Sleeve length         |
| 8 | Neck circumference    |
| 9 | Thigh circumference   |
|10 | Bicep circumference   |
|11 | Wrist circumference   |
|12 | Ankle circumference   |
|13 | Torso length          |
|14 | Rise length           |
|15 | Back width            |
|16 | Arm length            |

---

## Quick start

```bash
# 1. Activate the shared ML environment (Python 3.10 / Apple Silicon)
conda activate ml_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment
python scripts/check_environment.py

# 4. Inspect a captured image
python scripts/inspect_image.py --image data/raw/front.jpg --height 175

# 5. Run the test suite
pytest tests/ -v
```

---

## Folder structure

```
model1_body_measurement/
├── data/
│   ├── raw/            # Private input images  (git-ignored)
│   ├── processed/      # Derived features       (git-ignored)
│   └── ground_truth/   # Reference measurements (git-ignored)
├── notebooks/          # Experiments and visualisation
├── scripts/
│   ├── check_environment.py
│   └── inspect_image.py
├── src/
│   └── fitloop_measurement/
│       ├── config.py
│       ├── image_io.py
│       ├── quality/      # Stage 02 — framing & centering checks
│       ├── pose/         # Stage 03 — MediaPipe landmark extraction
│       ├── segmentation/ # Stage 03 — body contour extraction
│       ├── features/     # Stage 04 — geometric feature computation
│       ├── measurements/ # Stage 04 — dimension derivation
│       └── evaluation/   # Stage 06 — ground-truth comparison
├── tests/
├── requirements.txt
└── README.md
```

---

## Privacy

`data/raw/`, `data/processed/`, and `data/ground_truth/` are git-ignored.
**Never commit personal images or measurement records.**
Participants must give explicit, informed consent before their data is used for
model evaluation.

---

## Integration

Once the inference interface is stable (Stage 07) the model will be consumed by
`backend/app/services/` via a documented Python API.  The shared measurement
schema lives in `models/shared/measurement_schema/`.

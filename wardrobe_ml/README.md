# wardrobe_ml — FitLoop ML Workspace

This sub-workspace contains the machine learning pipeline for **FitLoop**, an AI-powered wardrobe scanning and clothing recognition system.

> **Phase 1 — Workspace Initialization only.**  
> No models are trained here yet. This README documents the structure, environment, and how to get started.

---

## 🗂️ Directory Structure

```
wardrobe_ml/
├── data/
│   ├── raw/          ← Symlinks or manifests pointing into the main dataset
│   ├── processed/    ← Cleaned, resized, normalized images ready for training
│   └── splits/       ← train.csv / val.csv / test.csv manifest files
├── notebooks/        ← Experiment notebooks (import config.py for paths)
├── src/              ← Reusable Python modules (loaders, transforms, metrics)
├── models/           ← Saved model checkpoints and exported weights
├── outputs/          ← Training logs, evaluation results, figures
├── config.py         ← Central path configuration (edit DATASET_ROOT here)
├── requirements.txt  ← Pinned dependencies for this workspace
└── README.md         ← This file
```

> The parent `notebooks/01_dataset_exploration.ipynb` lives in `../notebooks/` and is not affected by this workspace.

---

## ⚙️ Environment

The shared virtual environment lives at `../. venv/` (project root).

| Item | Value |
|------|-------|
| Python | 3.11.6 |
| Environment | `C:\Users\Lenovo\Downloads\Fitloop\.venv` |
| Jupyter kernel | `Python 3 (FitLoop)` |

**All required Phase 1 packages are already installed** — no additional installation needed.

---

## 🚀 Activate & Launch

```powershell
# 1. Navigate to the project root
cd C:\Users\Lenovo\Downloads\Fitloop

# 2. Activate the shared virtual environment
.venv\Scripts\Activate.ps1

# 3. Launch JupyterLab
jupyter lab

# 4. Select kernel: "Python 3 (FitLoop)"
```

---

## ⚙️ Configure Your Dataset Path

Open [`config.py`](./config.py) and set the `DATASET_ROOT` to wherever DeepFashion2 is stored:

```python
DATASET_ROOT: Path = Path(r"D:\datasets\DeepFashion2")  # ← Edit this
```

Or set it via environment variable (no file edit needed):

```powershell
$env:FITLOOP_DATASET_ROOT = "D:\datasets\DeepFashion2"
jupyter lab
```

Verify config resolves correctly:

```powershell
.venv\Scripts\python.exe wardrobe_ml\config.py
```

---

## 📦 Planned ML Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — Workspace Init** | Directories, config, environment | ✅ Complete |
| **2 — Preprocessing** | Subset selection, dedup, normalization | Upcoming |
| **3 — Model Training** | Detection + classification models | Upcoming |
| **4 — Inference Pipeline** | Latency optimization, ONNX export | Upcoming |

---

## 🚫 Restrictions (Phase 1)

- No model training.
- No dataset downloads.
- No modification of original dataset files.
- No PyTorch / TensorFlow / YOLO installation yet.

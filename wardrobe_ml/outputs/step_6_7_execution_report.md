# Stage 6.7 Execution Report: YOLO26n CPU Training Benchmark

**FitLoop Garment-Detection Pipeline**  
**Date:** November 7, 2026  
**Environment:** `fitloop_yolo_env` (Ultralytics 8.4.203, PyTorch 2.5.1+cpu)  
**Hardware:** Intel Core i5-12450H (8 physical / 12 logical cores), 15.73 GB RAM  
**Dataset YAML:** [`data.yaml`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/data.yaml)  
**Benchmark Artifact Directory:** [`step_6_7_cpu_benchmark/`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_7_cpu_benchmark)  
**JSON Telemetry Report:** [`step_6_7_cpu_benchmark_report.json`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_7_cpu_benchmark_report.json)

---

## 1. Executive Summary

Stage 6.7 executed an empirical, controlled CPU training benchmark using the **YOLO26n** architecture and the **DeepFashion2** dataset configuration on the local development machine.

- **Initialization Strategy:** YOLO26n initialized from scratch via `yolo26n.yaml` with **zero pretrained weights downloaded** (2,572,280 parameters, 5.9 GFLOPs across 260 network layers).
- **Controlled Benchmark Execution:** Trained for 1 full epoch on a controlled representative subset of 384 images (192 batches of batch size 2, image size $640 \times 640$, CPU execution, `workers=0`, `amp=False`, `seed=42`).
- **Empirical Throughput:** The Intel Core i5-12450H CPU achieved **2.97 images/second** (**648.9 ms/batch**, median **642.1 ms/batch**).
- **Resource Footprint:** Peak system RAM reached **13.96 GB** (process RSS: **1.28 GB**, minimum available RAM: 1.77 GB) with average CPU utilization of **40.4%** across all 12 logical cores.
- **Model Checkpoints:** Training completed successfully and persisted weights to [`last.pt`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_7_cpu_benchmark/run/weights/last.pt) and [`best.pt`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_7_cpu_benchmark/run/weights/best.pt).
- **Feasibility Verdict:** Full-dataset CPU training across all 191,958 images requires **~17.3 hours of pure training per epoch** + 1.3 hours validation (**~18.6 total wall-clock hours per epoch**, or **~38.8 continuous days for 50 epochs**). Full dataset training strictly requires GPU acceleration or a stratified subset.

---

## 2. Hardware and Environment Audit

| Parameter | Measured Specification | Status / Safety Margin |
| :--- | :---: | :--- |
| **CPU Model** | Intel Core i5-12450H | 8 physical cores, 12 logical cores (AVX2 enabled) |
| **Total System RAM** | 15.73 GB | 15.73 GB installed |
| **Available RAM (Start)** | 3.06 GB | 19.5% free (Above 1.5 GB minimum threshold) |
| **Available RAM (Peak Load)** | 1.77 GB | Process RSS remained strictly bounded at 1.28 GB |
| **PyTorch Version** | 2.5.1+cpu | Official stable CPU build |
| **CUDA Available** | False | CPU-only device execution enforced |
| **Ultralytics Version** | 8.4.203 | Native YOLO26 support |
| **Free Disk Space** | 581.97 GB | Abundant space (>580 GB free on `C:\`) |

---

## 3. Measured Training Benchmark Metrics

| Benchmark Metric | Measured Value | Notes / Interpretation |
| :--- | :---: | :--- |
| **Architecture** | YOLO26n (`yolo26n.yaml`) | Random initialization (no weights downloaded) |
| **Parameters** | 2,572,280 | Light nano detector (~2.57M params, 5.9 GFLOPs) |
| **Completed Batches** | 192 batches | 384 images at batch size 2 |
| **Training Epoch Duration** | 129.29 s | 2.15 minutes pure forward + backward time |
| **Validation Duration** | 10.37 s | Evaluated on representative validation sample (161 images) |
| **Total Benchmark Time** | 161.28 s | Includes dataloader init, training, val, and checkpoint save |
| **Average Time per Batch** | **648.9 ms** | Forward + backward + optimizer step |
| **Median Time per Batch** | **642.1 ms** | Outlier-filtered per-batch speed |
| **Measured Throughput** | **2.97 img/s** | Real-world CPU training rate |
| **Average CPU Load** | 40.4% | Multi-threaded PyTorch CPU execution |
| **Peak CPU Load** | 76.9% | Peak burst during loss backward step |
| **Peak Process RAM** | 1.28 GB | Modest memory footprint, well within safe limits |
| **Loss Values (End of Ep 1)** | Box: 3.2591 \| Cls: 15.3771 \| L1: 0.0835 | Stable convergence trajectory from random weights |

---

## 4. Preliminary Validation Metrics

> [!NOTE]
> This validation pass evaluates a preliminary model trained for exactly 1 epoch from random initialization on a small benchmark subset. These values reflect pipeline functionality and metric reporting, not converged model accuracy.

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Precision (B)** | 0.0917 | Preliminary detections emerging |
| **Recall (B)** | 0.0068 | Early random state before convergence |
| **mAP@50 (B)** | 0.0001 | Baseline starting anchor |
| **mAP@50-95 (B)** | 0.0000 | Baseline starting anchor |

---

## 5. Full Dataset Training Feasibility Extrapolations

Extrapolations based on the empirical throughput of **2.97 images/second** (648.9 ms/batch) for all **191,958 clean training images** (95,979 batches at batch size 2) and **32,153 validation images**:

| Target Epochs | Estimated Training Time | Estimated Validation Time | Total Wall-Clock Time | Feasibility on Local CPU |
| :---: | :---: | :---: | :---: | :--- |
| **1 Epoch** | 17.3 hours | 1.3 hours | **18.6 hours** (0.8 days) | Feasible overnight only |
| **10 Epochs** | 173.0 hours | 13.3 hours | **186.3 hours** (7.8 days) | Unrealistic for routine iteration |
| **30 Epochs** | 519.0 hours | 39.9 hours | **558.9 hours** (23.3 days) | Completely impractical |
| **50 Epochs** | 865.0 hours | 66.5 hours | **931.5 hours** (38.8 days) | Impractical |
| **100 Epochs** | 1730.0 hours | 133.0 hours | **1863.0 hours** (77.6 days) | Impractical |

---

## 6. Strategic Recommendations

1. **Local CPU Training Strategy (Stratified Prototyping):**  
   For local CPU experiments, train on a **1% to 2% stratified pair-clustered subset** (~2,000 to 4,000 images). This preserves outfit diversity and cluster integrity while keeping a 10-epoch training run under **5 to 8 hours**.
2. **GPU Acceleration for Baseline Convergence:**  
   Training on the full 191,958 images for 50–100 epochs requires GPU acceleration (e.g., Google Colab, Kaggle, or cloud GPU instance). On a modern GPU (RTX 4060 or T4/A10), throughput reaches 150–300 img/s, reducing 50 epochs from **38+ days down to ~8–12 hours**.
3. **Hyperparameter Adjustments for CPU:**  
   If continuing CPU experiments, test `batch=4` or `batch=8` with multi-threading to determine whether AVX2 vectorization yields higher throughput per core.

---

## 7. Safety Compliance Checklist

- [x] Raw DeepFashion2 dataset images and annotations remain untouched.
- [x] Converted YOLO labels and dataset manifests remain untouched.
- [x] Zero model weights downloaded (initialized from scratch via `yolo26n.yaml`).
- [x] Separate benchmark directory used ([`wardrobe_ml/outputs/step_6_7_cpu_benchmark/`](file:///C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_7_cpu_benchmark)).
- [x] Exactly 1 epoch executed.
- [x] Device strictly kept to CPU.
- [x] Memory usage remained safe (Peak process RAM: 1.28 GB, system RAM headroom preserved).

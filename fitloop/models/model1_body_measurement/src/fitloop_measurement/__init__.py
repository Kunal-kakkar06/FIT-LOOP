"""
fitloop_measurement
===================
Body measurement engine for FitLoop Model 1.

Stages:
    image_io        – Load and validate image files.
    quality         – Framing, centering, and obstruction checks.
    pose            – MediaPipe landmark extraction.
    segmentation    – Body silhouette / contour extraction.
    features        – Geometric feature computation from landmarks + contour.
    measurements    – Derive body dimensions from features.
    evaluation      – Compare estimates against ground-truth references.
"""

__version__ = "0.1.0"

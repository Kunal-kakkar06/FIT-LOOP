#!/usr/bin/env python3
"""
Unit & Integration Test Suite for GarmentEmbeddingExtractor
============================================================
Tests:
1. Feature dimension (576-dim) and strict L2 unit normalization (norm == 1.0).
2. Identical garment crops yield cosine similarity >= 0.99.
3. Solid navy vs striped navy yields distinct visual similarity (< 0.70).
4. Highly distinct colors/patterns yield low visual similarity (< 0.50).
5. Mathematical invariant checks (all similarity values strictly within [0.0, 1.0]).
6. Generates wardrobe_ml/outputs/embedding_extractor_test_report.txt.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure wardrobe_ml is on sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from embedding_extractor import GarmentEmbeddingExtractor
from duplicate_detector import DuplicateDetector

OUTPUT_REPORT_PATH = PROJECT_DIR / "outputs" / "embedding_extractor_test_report.txt"
OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_solid_image(bgr_color: tuple, size: int = 224) -> np.ndarray:
    """Create a synthetic solid colored image."""
    return np.full((size, size, 3), bgr_color, dtype=np.uint8)


def create_striped_image(bgr_color1: tuple, bgr_color2: tuple, size: int = 224, stripe_width: int = 14) -> np.ndarray:
    """Create a synthetic striped image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        if (y // stripe_width) % 2 == 0:
            img[y, :] = bgr_color1
        else:
            img[y, :] = bgr_color2
    return img


def create_checkerboard_image(size: int = 224, block_size: int = 16) -> np.ndarray:
    """Create a high-contrast checkerboard image (distinct pattern and structure)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = (255, 255, 255)
            else:
                img[i, j] = (0, 0, 0)
    return img


def run_tests() -> bool:
    print("=" * 75)
    print("FitLoop GarmentEmbeddingExtractor Verification Test Suite")
    print("=" * 75)

    t0 = time.perf_counter()
    all_passed = True
    report_lines: List[str] = [
        "FITLOOP GARMENT EMBEDDING EXTRACTOR VERIFICATION REPORT",
        "=" * 75,
        f"Execution Timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Python Environment  : {sys.executable}",
        "",
        "SECTION 1: BACKBONE ARCHITECTURE & NORMALIZATION",
        "-" * 75,
    ]

    extractor = GarmentEmbeddingExtractor(device="cpu")
    print(f"Extractor initialized on device: {extractor.device}")

    # 1. Feature Dimension and L2 Normalization Test
    navy_bgr = (59, 38, 27)  # Navy Blue #1B263B
    solid_navy_img = create_solid_image(navy_bgr)

    emb_navy_1 = extractor.extract(solid_navy_img)
    vec_len = len(emb_navy_1)
    l2_norm = float(np.linalg.norm(emb_navy_1))
    all_finite = all(np.isfinite(x) for x in emb_navy_1)

    t1_pass = (vec_len == 576 and abs(l2_norm - 1.0) < 1e-4 and all_finite)
    if not t1_pass:
        all_passed = False

    t1_status = "PASS" if t1_pass else "FAIL"
    print(f"[{t1_status}] Test 1: Feature Dimension & Unit Norm (Dim: {vec_len}, Norm: {l2_norm:.6f})")
    report_lines.append(f"Test 1: Vector Dimension={vec_len} (exp: 576), L2 Norm={l2_norm:.6f} (exp: 1.000000) -> [{t1_status}]")

    # 2. Identical Garment Swatches
    emb_navy_2 = extractor.extract(solid_navy_img)
    sim_identical = GarmentEmbeddingExtractor.compute_cosine_similarity(emb_navy_1, emb_navy_2)

    t2_pass = (sim_identical >= 0.99 and sim_identical <= 1.0001)
    if not t2_pass:
        all_passed = False

    t2_status = "PASS" if t2_pass else "FAIL"
    print(f"[{t2_status}] Test 2: Identical Swatch Similarity (Sim: {sim_identical:.4f}, exp: >= 0.99)")
    report_lines.append(f"Test 2: Identical Swatch Cosine Similarity={sim_identical:.4f} (exp: >= 0.99) -> [{t2_status}]")

    # 3. Solid Navy vs Striped Navy
    striped_navy_img = create_striped_image(navy_bgr, (255, 255, 255))
    emb_striped = extractor.extract(striped_navy_img)
    sim_striped = GarmentEmbeddingExtractor.compute_cosine_similarity(emb_navy_1, emb_striped)

    t3_pass = (sim_striped < 0.70 and sim_striped >= 0.0)
    if not t3_pass:
        all_passed = False

    t3_status = "PASS" if t3_pass else "FAIL"
    print(f"[{t3_status}] Test 3: Solid Navy vs Striped Navy (Sim: {sim_striped:.4f}, exp: < 0.70)")
    report_lines.append(f"Test 3: Solid Navy vs Striped Navy Cosine Similarity={sim_striped:.4f} (exp: < 0.70) -> [{t3_status}]")

    # 4. Solid Navy vs High-Contrast Pattern / Checkerboard
    checkerboard_img = create_checkerboard_image()
    emb_checker = extractor.extract(checkerboard_img)
    sim_checker = GarmentEmbeddingExtractor.compute_cosine_similarity(emb_navy_1, emb_checker)

    t4_pass = (sim_checker < 0.20 and sim_checker >= 0.0)
    if not t4_pass:
        all_passed = False

    t4_status = "PASS" if t4_pass else "FAIL"
    print(f"[{t4_status}] Test 4: Solid Navy vs Checkerboard Pattern (Sim: {sim_checker:.4f}, exp: < 0.20)")
    report_lines.append(f"Test 4: Solid Navy vs Checkerboard Cosine Similarity={sim_checker:.4f} (exp: < 0.20) -> [{t4_status}]")

    # 5. DuplicateDetector Visual Similarity Method & Invariant Bounds
    sim_dd = DuplicateDetector.compute_visual_similarity(emb_navy_1, emb_navy_2)
    sim_none_1 = DuplicateDetector.compute_visual_similarity(None, emb_navy_1)
    sim_none_2 = DuplicateDetector.compute_visual_similarity(emb_navy_1, None)
    sim_empty = DuplicateDetector.compute_visual_similarity([], emb_navy_1)

    t5_pass = (
        sim_dd is not None and sim_dd >= 0.99
        and sim_none_1 is None
        and sim_none_2 is None
        and sim_empty is None
    )
    if not t5_pass:
        all_passed = False

    t5_status = "PASS" if t5_pass else "FAIL"
    print(f"[{t5_status}] Test 5: DuplicateDetector Interoperability & None Handling")
    report_lines.append(f"Test 5: DuplicateDetector Interoperability: sim={sim_dd}, handles None/empty cleanly -> [{t5_status}]")

    total_time = (time.perf_counter() - t0) * 1000
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"

    report_lines.extend([
        "",
        "=" * 75,
        f"Total Runtime       : {total_time:.2f} ms",
        f"Overall Status      : {overall_status}",
        "=" * 75,
    ])

    report_text = "\n".join(report_lines)
    OUTPUT_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to: {OUTPUT_REPORT_PATH.name}")
    print(f"OVERALL STATUS: [{overall_status}]")
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

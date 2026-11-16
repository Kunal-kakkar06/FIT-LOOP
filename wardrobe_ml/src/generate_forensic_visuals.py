"""
Stage 6.4C.1: Forensic Visualization Generator
==============================================
Generates diagnostic forensic figures for:
1. 028225 (out-of-bounds trousers, y2=626 > 624)
2. 115063 (out-of-bounds short sleeve top, y2=829 > 702)
3. 101338 (zero-area short sleeve top, [0,0,0,0])
4. Multi-case summary contact sheet

Saves all figures to:
- wardrobe_ml/outputs/step_6_4c1_bbox_forensic/
- <artifact_dir> for embedding in markdown reports.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PROJECT_ROOT = Path("C:/Users/Lenovo/Downloads/Fitloop")
DEEPFASHION2_ROOT = Path("C:/Users/Lenovo/Downloads/DeepFashion2")
TRAIN_IMG_DIR = DEEPFASHION2_ROOT / "train" / "image"
TRAIN_ANNO_DIR = DEEPFASHION2_ROOT / "train" / "annos"
OUTPUT_DIR = PROJECT_ROOT / "wardrobe_ml" / "outputs" / "step_6_4c1_bbox_forensic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("C:/Users/Lenovo/.gemini/antigravity/brain/57b5c7be-d326-40ec-a129-e53bc9ecb41d")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def generate_028225_diagnostic():
    image_id = "028225"
    img_path = TRAIN_IMG_DIR / f"{image_id}.jpg"
    json_path = TRAIN_ANNO_DIR / f"{image_id}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size  # 468, 624
    img_np = np.array(img)

    # item1: [40, 0, 399, 381] - valid
    # item2: [1, 198, 422, 626] - invalid (y2=626 > 624)
    item1_box = anno["item1"]["bounding_box"]
    item2_box = anno["item2"]["bounding_box"]
    item2_seg = anno["item2"]["segmentation"]

    # Canvas padding: extend bottom by 60 pixels to display y2=626 without clipping
    pad_bottom = 60
    canvas_h = h + pad_bottom
    canvas_w = w

    canvas = np.full((canvas_h, canvas_w, 3), 220, dtype=np.uint8)
    # Paint image onto top of canvas
    canvas[0:h, 0:w] = img_np
    # Paint extended region with a subtle striped / tinted warning pattern
    for y in range(h, canvas_h):
        for x in range(0, canvas_w):
            if (x + y) % 16 < 8:
                canvas[y, x] = [255, 200, 200]
            else:
                canvas[y, x] = [240, 180, 180]

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2])

    # Subplot 1: Full Canvas Overview
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(canvas)
    ax1.set_title(f"Image {image_id}.jpg: Full Canvas with Extended Margin\n(Dimensions: W={w}, H={h} | Canvas H={canvas_h})", fontsize=11, fontweight="bold")

    # Draw true image boundary
    rect_img = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor="#00FFCC", facecolor="none", linestyle="-")
    ax1.add_patch(rect_img)
    ax1.axhline(h, color="#FF0000", linestyle="--", linewidth=1.5, label=f"True Image Bottom Boundary (y = {h})")

    # Item 1 valid box
    x1, y1, x2, y2 = item1_box
    rect_item1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="#00CC44", facecolor="none")
    ax1.add_patch(rect_item1)
    ax1.text(x1 + 4, y1 + 18, f"item1: short sleeve top (VALID)\n[{x1}, {y1}, {x2}, {y2}]",
             color="#00CC44", fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Item 2 invalid box
    bx1, by1, bx2, by2 = item2_box
    rect_item2 = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2, edgecolor="#FF2222", facecolor="none")
    ax1.add_patch(rect_item2)
    ax1.text(bx1 + 4, by1 + 18, f"item2: trousers (REJECTED: y2={by2} > {h})\n[{bx1}, {by1}, {bx2}, {by2}]",
             color="#FF4444", fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Exterior region label
    ax1.text(w // 2, h + 30, f"<< OUTSIDE IMAGE BOUNDARY (y > {h}) >>\nBottom padding: +{pad_bottom}px",
             color="#990000", fontsize=9, fontweight="bold", ha="center", va="center")

    ax1.set_xlim(-10, w + 10)
    ax1.set_ylim(canvas_h + 5, -5)
    ax1.legend(loc="upper left", fontsize=8)

    # Subplot 2: High-Resolution Zoom on Bottom Boundary
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(canvas)
    ax2.set_title(f"Forensic Detail: Bottom Edge Violation Region\n(Zoom: y in [580, {canvas_h}], x in [0, {w}])", fontsize=11, fontweight="bold")

    # Zoom window
    zoom_y_min, zoom_y_max = 580, canvas_h
    zoom_x_min, zoom_x_max = 0, w

    # Boundary line
    ax2.axhline(h, color="#00FFFF", linestyle="-", linewidth=2.5, label=f"Image Bottom Edge: y = {h} px")
    # Box bottom line
    ax2.axhline(by2, color="#FF0000", linestyle="--", linewidth=2.5, label=f"Annotated Box Bottom: y2 = {by2} px (+2px violation)")

    # Plot item2 bbox boundary
    rect_zoom = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.5, edgecolor="#FF2222", facecolor="none")
    ax2.add_patch(rect_zoom)

    # Plot segmentation polygons in the zoom region
    for poly in item2_seg:
        if isinstance(poly, list):
            pts = np.array(poly).reshape(-1, 2)
            poly_patch = patches.Polygon(pts, linewidth=1.5, edgecolor="#FFD700", facecolor="none", linestyle=":")
            ax2.add_patch(poly_patch)
            # Find max Y point in segmentation
            max_pt = pts[np.argmax(pts[:, 1])]
            ax2.plot(max_pt[0], max_pt[1], "o", color="#FFD700", markersize=6)
            ax2.text(max_pt[0] + 5, max_pt[1] - 3, f"Max Seg Y: {max_pt[1]}px", color="#FFD700", fontsize=8, fontweight="bold")

    # Annotation arrow pointing to the violation
    ax2.annotate(
        f"VIOLATION: y2 = {by2} px\nExceeds H={h} by 2 px\nStatus: REJECTED",
        xy=(200, by2), xytext=(220, 600),
        arrowprops=dict(facecolor="#FF0000", shrink=0.05, width=2, headwidth=8),
        fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#CC0000", alpha=0.9)
    )

    ax2.set_xlim(zoom_x_min, zoom_x_max)
    ax2.set_ylim(zoom_y_max, zoom_y_min)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "forensic_028225_out_of_bounds.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    shutil.copy(out_path, ARTIFACT_DIR / "forensic_028225_out_of_bounds.png")


def generate_115063_diagnostic():
    image_id = "115063"
    img_path = TRAIN_IMG_DIR / f"{image_id}.jpg"
    json_path = TRAIN_ANNO_DIR / f"{image_id}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size  # 468, 702
    img_np = np.array(img)

    # item1: [15, 195, 466, 829] - invalid (y2=829 > 702, +127px violation!)
    item1_box = anno["item1"]["bounding_box"]
    item1_seg = anno["item1"]["segmentation"]
    bx1, by1, bx2, by2 = item1_box

    # Canvas padding: extend bottom by 160 pixels to display y2=829 without clipping
    pad_bottom = 160
    canvas_h = h + pad_bottom
    canvas_w = w

    canvas = np.full((canvas_h, canvas_w, 3), 220, dtype=np.uint8)
    canvas[0:h, 0:w] = img_np
    for y in range(h, canvas_h):
        for x in range(0, canvas_w):
            if (x + y) % 20 < 10:
                canvas[y, x] = [255, 190, 190]
            else:
                canvas[y, x] = [240, 170, 170]

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2])

    # Subplot 1: Full Canvas
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(canvas)
    ax1.set_title(f"Image {image_id}.jpg: Full Canvas with +160px Exterior Padding\n(Image: W={w}, H={h} | BBox: y2={by2} [+127px Out-of-Bounds])", fontsize=11, fontweight="bold")

    # True image boundary
    rect_img = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor="#00FFCC", facecolor="none")
    ax1.add_patch(rect_img)
    ax1.axhline(h, color="#00FFFF", linestyle="-", linewidth=2, label=f"Image Boundary (y = {h})")

    # Unclipped item1 bounding box
    rect_box = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.5, edgecolor="#FF1111", facecolor="none")
    ax1.add_patch(rect_box)
    ax1.text(bx1 + 4, by1 + 25, f"item1: short sleeve top (REJECTED)\nRaw: [{bx1}, {by1}, {bx2}, {by2}]\ny2 = {by2} exceeds H={h} by 127px!",
             color="white", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#CC0000", alpha=0.85))

    # Plot segmentation polygon
    for poly in item1_seg:
        if isinstance(poly, list):
            pts = np.array(poly).reshape(-1, 2)
            poly_patch = patches.Polygon(pts, linewidth=2, edgecolor="#FFD700", facecolor="none", linestyle="--")
            ax1.add_patch(poly_patch)

    ax1.text(w // 2, h + 70, f"<< OUTSIDE IMAGE BOUNDARY (H > {h}) >>\nReported Bounding Box Extends +127px Into Void!\n(y2 = {by2})",
             color="#880000", fontsize=9.5, fontweight="bold", ha="center", va="center")

    ax1.set_xlim(-15, w + 15)
    ax1.set_ylim(canvas_h + 10, -10)
    ax1.legend(loc="upper left", fontsize=8)

    # Subplot 2: Detailed Forensic Diagnostic
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(canvas)
    ax2.set_title("Forensic Analysis: Garment Segmentation vs Bogus BBox Protrusion", fontsize=11, fontweight="bold")

    # Draw lines
    ax2.axhline(h, color="#00FFFF", linestyle="-", linewidth=2.5, label=f"True Image Bottom: y = {h}")
    ax2.axhline(by2, color="#FF0000", linestyle="--", linewidth=2.5, label=f"Bogus Box Bottom: y2 = {by2} (+127px)")

    # Segmentation bounds
    seg_ys = []
    for poly in item1_seg:
        if isinstance(poly, list):
            pts = np.array(poly).reshape(-1, 2)
            seg_ys.extend(pts[:, 1])
            poly_patch = patches.Polygon(pts, linewidth=2.2, edgecolor="#FFD700", facecolor="#FFD700", alpha=0.25)
            ax2.add_patch(poly_patch)
            ax2.plot(pts[:, 0], pts[:, 1], "o", color="#FFD700", markersize=3)

    max_seg_y = max(seg_ys) if seg_ys else 0
    ax2.axhline(max_seg_y, color="#FFD700", linestyle=":", linewidth=2, label=f"Actual Garment Hem (Seg max Y = {max_seg_y})")

    # Unclipped item1 bounding box
    rect_box2 = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.5, edgecolor="#FF1111", facecolor="none")
    ax2.add_patch(rect_box2)

    # Shaded violation region
    violation_rect = patches.Rectangle((bx1, h), bx2 - bx1, by2 - h, linewidth=1, edgecolor="#FF0000", facecolor="#FF0000", alpha=0.35)
    ax2.add_patch(violation_rect)

    # Annotations
    ax2.annotate(
        f"TRUE GARMENT ENDS HERE\n(y = {max_seg_y} px)\nInside image bounds",
        xy=(200, max_seg_y), xytext=(220, 560),
        arrowprops=dict(facecolor="#FFD700", shrink=0.05, width=2, headwidth=7),
        fontsize=9, fontweight="bold", color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFD700", alpha=0.9)
    )

    ax2.annotate(
        f"MASSIVE UPSTREAM ERROR\nBox extends +127 px past image edge!\nTotal height = {by2 - by1} px\nTrue garment height = {max_seg_y - min(seg_ys)} px",
        xy=((bx1 + bx2) // 2, (h + by2) // 2), xytext=(60, 780),
        arrowprops=dict(facecolor="#FF0000", shrink=0.05, width=2, headwidth=7),
        fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#990000", alpha=0.95)
    )

    ax2.set_xlim(0, w)
    ax2.set_ylim(canvas_h, 450)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "forensic_115063_out_of_bounds.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    shutil.copy(out_path, ARTIFACT_DIR / "forensic_115063_out_of_bounds.png")


def generate_101338_diagnostic():
    image_id = "101338"
    img_path = TRAIN_IMG_DIR / f"{image_id}.jpg"
    json_path = TRAIN_ANNO_DIR / f"{image_id}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size  # 750, 836
    img_np = np.array(img)

    # item1: [307, 437, 432, 597] - valid skirt
    # item2: [0, 0, 0, 0] - zero area short sleeve top (degenerate!)
    item1_box = anno["item1"]["bounding_box"]
    item2_box = anno["item2"]["bounding_box"]
    item2_seg = anno["item2"]["segmentation"]

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.1])

    # Subplot 1: Full Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_np)
    ax1.set_title(f"Image {image_id}.jpg: Degenerate Box vs Valid Annotation\n(Dimensions: W={w}, H={h})", fontsize=11, fontweight="bold")

    # Item 1 valid box
    x1, y1, x2, y2 = item1_box
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#00CC44", facecolor="none")
    ax1.add_patch(rect1)
    ax1.text(x1 + 4, y1 - 10, f"item1: skirt (VALID)\n[{x1}, {y1}, {x2}, {y2}]",
             color="#00CC44", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Item 2 degenerate box at (0, 0)
    ax1.plot(0, 0, "X", color="#FF0000", markersize=14, markeredgewidth=3, label="item2: [0,0,0,0] Degenerate Point (Area=0)")
    ax1.text(10, 30, "item2: [0, 0, 0, 0]\nArea = 0 (REJECTED)",
             color="#FF3333", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))

    # True item2 segmentation location
    for poly in item2_seg:
        if isinstance(poly, list):
            pts = np.array(poly).reshape(-1, 2)
            poly_patch = patches.Polygon(pts, linewidth=2, edgecolor="#FFD700", facecolor="#FFD700", alpha=0.3)
            ax1.add_patch(poly_patch)
            ax1.text(350, 360, "Actual Garment\n(Segmentation exists,\nbut BBox is [0,0,0,0])",
                     color="yellow", fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.legend(loc="lower right", fontsize=8)

    # Subplot 2: Close-up on item2 Garment & Origin Point
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(img_np)
    ax2.set_title("Forensic Detail: Item 2 True Garment Location vs Box Origin", fontsize=11, fontweight="bold")

    for poly in item2_seg:
        if isinstance(poly, list):
            pts = np.array(poly).reshape(-1, 2)
            poly_patch = patches.Polygon(pts, linewidth=2.5, edgecolor="#FFD700", facecolor="#FFD700", alpha=0.35)
            ax2.add_patch(poly_patch)

    # Item 1 top portion
    rect1_close = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#00CC44", facecolor="none")
    ax2.add_patch(rect1_close)

    ax2.annotate(
        "item2 true garment location\nCategory: short sleeve top\nValid segmentation & landmarks exist!",
        xy=(390, 390), xytext=(220, 240),
        arrowprops=dict(facecolor="#FFD700", shrink=0.05, width=2, headwidth=7),
        fontsize=9, fontweight="bold", color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFD700", alpha=0.9)
    )

    ax2.annotate(
        "Reported Bounding Box:\n[0, 0, 0, 0] at top-left corner!\ndx=0, dy=0, area=0\nStatus: REJECTED",
        xy=(50, 50), xytext=(80, 140),
        arrowprops=dict(facecolor="#FF0000", shrink=0.05, width=2, headwidth=7),
        fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#990000", alpha=0.95)
    )

    ax2.set_xlim(0, w)
    ax2.set_ylim(650, 0)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "forensic_101338_zero_area.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    shutil.copy(out_path, ARTIFACT_DIR / "forensic_101338_zero_area.png")


def generate_summary_contact_sheet():
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=150)
    fig.suptitle("FitLoop Stage 6.4C.1: Forensic Verification of All 3 Rejected Annotations", fontsize=14, fontweight="bold", y=0.98)

    # Case 1: 028225
    img1 = Image.open(OUTPUT_DIR / "forensic_028225_out_of_bounds.png")
    axes[0].imshow(img1)
    axes[0].set_title("Case 1: 028225 (item2)\nReason: out_of_bounds (y2=626 > 624, +2px)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Case 2: 115063
    img2 = Image.open(OUTPUT_DIR / "forensic_115063_out_of_bounds.png")
    axes[1].imshow(img2)
    axes[1].set_title("Case 2: 115063 (item1)\nReason: out_of_bounds (y2=829 > 702, +127px)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Case 3: 101338
    img3 = Image.open(OUTPUT_DIR / "forensic_101338_zero_area.png")
    axes[2].imshow(img3)
    axes[2].set_title("Case 3: 101338 (item2)\nReason: zero_or_negative_area ([0,0,0,0])", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "forensic_summary_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    shutil.copy(out_path, ARTIFACT_DIR / "forensic_summary_comparison.png")


def main():
    print("Generating forensic diagnostic visualizations...")
    generate_028225_diagnostic()
    generate_115063_diagnostic()
    generate_101338_diagnostic()
    generate_summary_contact_sheet()
    print("All forensic visualizations generated and copied to artifact directory.")


if __name__ == "__main__":
    main()

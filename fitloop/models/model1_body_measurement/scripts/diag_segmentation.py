"""
diag_segmentation.py
--------------------
Minimal isolated script that reproduces the segmentation crash scenario
and reports which step succeeds or fails.

Run as:
    PYTHONPATH=src /opt/anaconda3/envs/ml_env/bin/python scripts/diag_segmentation.py
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

IMG = "/Users/kunalkakkar/Desktop/IMG_5391.jpg"
PROJECT_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(PROJECT_SRC))

def main():
    print("[DIAG-01] imports starting", flush=True)
    import cv2
    import numpy as np
    print("[DIAG-02] cv2 + numpy imported", flush=True)

    from fitloop_measurement.segmentation.segmenter import BodySegmenter, _MP_MAX_SIDE
    from fitloop_measurement.segmentation._worker import run_segmentation_in_subprocess
    print("[DIAG-03] BodySegmenter + worker imported", flush=True)

    # ── Step 1: load image ──────────────────────────────────────────────────────
    img = cv2.imread(IMG)
    if img is None:
        print(f"[DIAG-FAIL] Could not load {IMG}", flush=True)
        sys.exit(1)
    orig_h, orig_w = img.shape[:2]
    print(f"[DIAG-04] Image loaded: {orig_w}x{orig_h} (longest={max(orig_h,orig_w)})", flush=True)

    # ── Step 2: confirm segmenter.py has NO direct numpy_view() call ────────────
    segmenter_src = os.path.join(PROJECT_SRC,
        "fitloop_measurement/segmentation/segmenter.py")
    with open(segmenter_src) as f:
        seg_text = f.read()
    if "numpy_view" in seg_text:
        print("[DIAG-WARN] segmenter.py still contains 'numpy_view' — check the file!", flush=True)
    else:
        print("[DIAG-05] segmenter.py: no numpy_view call present ✓", flush=True)

    worker_src = os.path.join(PROJECT_SRC,
        "fitloop_measurement/segmentation/_worker.py")
    with open(worker_src) as f:
        worker_text = f.read()
    nv_in_worker = "numpy_view" in worker_text
    print(f"[DIAG-06] _worker.py has numpy_view: {nv_in_worker} "
          f"(expected: True — worker uses it inside child process where SIGABRT is contained)", flush=True)

    # ── Step 3: dispatch to subprocess ─────────────────────────────────────────
    print("[DIAG-07] dispatching to subprocess...", flush=True)
    from fitloop_measurement.config import DEFAULT_MODEL_PATH
    result = run_segmentation_in_subprocess(
        bgr_image=img,
        model_path_str=str(DEFAULT_MODEL_PATH),
        mp_max_side=_MP_MAX_SIDE,
        timeout=90.0,
    )
    print(f"[DIAG-08] subprocess returned: type={type(result).__name__}", flush=True)

    if isinstance(result, str):
        print(f"[DIAG-FAIL] Subprocess error: {result}", flush=True)
        sys.exit(1)

    mask = result["confidence_mask"]
    h, w = result["h"], result["w"]
    print(f"[DIAG-09] mask shape={mask.shape}  dtype={mask.dtype}  "
          f"inference_dims={w}x{h}  orig_dims={orig_w}x{orig_h}", flush=True)
    print(f"[DIAG-10] mask value range: min={mask.min():.4f} max={mask.max():.4f} "
          f"mean={mask.mean():.4f}", flush=True)

    # ── Step 4: full BodySegmenter.segment() call ───────────────────────────────
    print("[DIAG-11] calling BodySegmenter.segment() (full pipeline)...", flush=True)
    with BodySegmenter() as seg:
        seg_result = seg.segment(img, view_type="front")

    print(f"[DIAG-12] status={seg_result.status}", flush=True)
    print(f"[DIAG-13] fg_ratio={seg_result.foreground_ratio:.4f}", flush=True)
    print(f"[DIAG-14] mask_shape={(seg_result.mask.shape if seg_result.mask is not None else None)}", flush=True)
    print(f"[DIAG-15] contour={seg_result.principal_contour is not None}", flush=True)
    print(f"[DIAG-16] metadata={seg_result.metadata}", flush=True)
    print(f"[DIAG-17] warnings={seg_result.warnings}", flush=True)

    # ── Step 5: coordinate-system check ─────────────────────────────────────────
    print("[DIAG-18] coordinate-system alignment check:", flush=True)
    print(f"          returned image_width={seg_result.image_width}  orig_w={orig_w}  match={seg_result.image_width == orig_w}", flush=True)
    print(f"          returned image_height={seg_result.image_height}  orig_h={orig_h}  match={seg_result.image_height == orig_h}", flush=True)
    if seg_result.mask is not None:
        mh, mw = seg_result.mask.shape[:2]
        print(f"          mask dims={mw}x{mh}  match image dims={mw==orig_w and mh==orig_h}", flush=True)
    if seg_result.principal_contour is not None:
        bb = seg_result.principal_contour.bounding_box
        print(f"          contour bbox x={bb.x} y={bb.y} w={bb.width} h={bb.height}", flush=True)
        in_bounds = (bb.x >= 0 and bb.y >= 0 and
                     bb.x + bb.width <= orig_w and
                     bb.y + bb.height <= orig_h)
        print(f"          contour bbox in original image bounds: {in_bounds}", flush=True)

    print("[DIAG-PASS] Parent process survived — subprocess isolation working correctly", flush=True)

if __name__ == "__main__":
    main()

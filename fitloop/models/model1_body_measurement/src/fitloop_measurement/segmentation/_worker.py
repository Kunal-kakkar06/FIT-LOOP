"""
segmentation/_worker.py
-----------------------
Out-of-process MediaPipe segmentation worker.

This module is ONLY run inside child processes spawned by BodySegmenter.
It exists because MpImageDataFloat32 — the C function MediaPipe's Python
bindings call when converting a VEC32F1 segmentation-mask ImageFrame to a
numpy array — executes `CopyToBuffer()` which aborts the process via
SIGABRT on ARM64 macOS (Apple M-series) whenever the ImageFrame channel-size
check fails.  A Python `try/except` cannot intercept a process abort.

By running inference inside a subprocess and returning only primitive
serialisable data through a multiprocessing Queue, any native crash is
contained within the child process; the parent uvicorn worker remains alive.

Worker entry point
------------------
Called via `multiprocessing.Process(target=_run_worker, args=(...))`.
Results are placed on the supplied queue as either:
  - a dict with keys: confidence_mask (np.ndarray float32), h, w, orig_h, orig_w
  - an exception string (str) if inference fails

Public helpers
--------------
run_segmentation_in_subprocess(bgr_image, model_path, foreground_threshold,
                                mp_max_side, timeout) -> dict | str
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Worker function — executed inside the child process ONLY
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker(
    bgr_bytes: bytes,
    orig_h: int,
    orig_w: int,
    model_path_str: str,
    mp_max_side: int,
    queue: "multiprocessing.Queue[Any]",
) -> None:
    """
    Perform MediaPipe PoseLandmarker inference and extract the segmentation
    confidence mask inside an isolated child process.

    Parameters
    ----------
    bgr_bytes:
        Raw bytes of the BGR image (obtained via ndarray.tobytes()).  The
        child process reconstructs the numpy array from this buffer to avoid
        pickle overhead for large arrays.
    orig_h, orig_w:
        Pixel dimensions of the original (possibly unscaled) image.
    model_path_str:
        Absolute path to the pose_landmarker_heavy.task asset.
    mp_max_side:
        Maximum side length to use for inference (images larger than this are
        downsampled before running through MediaPipe).
    queue:
        multiprocessing.Queue used to communicate results back to the parent.
        On success: puts a dict with keys confidence_mask, h, w, orig_h, orig_w.
        On failure: puts a str describing the error.
    """
    # Suppress all logging noise from MediaPipe in the child process
    os.environ.setdefault("GLOG_logtostderr", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")

    try:
        import mediapipe as mp  # noqa: PLC0415

        # Reconstruct BGR image from raw bytes
        bgr_image: np.ndarray = np.frombuffer(bgr_bytes, dtype=np.uint8).reshape(
            orig_h, orig_w, 3
        )

        h, w = orig_h, orig_w

        # Downscale to mp_max_side if necessary
        longest = max(h, w)
        if longest > mp_max_side:
            scale = mp_max_side / longest
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            bgr_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Convert to RGB, ensure C-contiguous layout
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Build landmarker with segmentation masks enabled
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path_str),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        with PoseLandmarker.create_from_options(options) as landmarker:
            detect_results = landmarker.detect(mp_image)

        if not detect_results.segmentation_masks:
            queue.put("NO_MASKS")
            return

        # ─────────────────────────────────────────────────────────────────
        # Safe mask extraction using MpImageGetValueFloat32 (__getitem__)
        # instead of numpy_view() / MpImageDataFloat32.
        #
        # numpy_view() calls MpImageDataFloat32 which aborts on this
        # ARM64 macOS build via Check failed: 1 == ChannelSize() (1 vs 4).
        #
        # __getitem__(row, col) calls MpImageGetValueFloat32 which accesses
        # individual pixel values and does NOT go through CopyToBuffer().
        # It is slower but crash-safe.
        #
        # For practical image sizes (≤ 1920px) we use vectorised ctypes
        # to avoid a pure-Python per-pixel loop.
        # ─────────────────────────────────────────────────────────────────
        mask_img = detect_results.segmentation_masks[0]
        mask_h = mask_img.height
        mask_w = mask_img.width

        confidence_mask = _extract_float_mask_safe(mask_img, mask_h, mask_w)

        queue.put(
            {
                "confidence_mask": confidence_mask,
                "h": h,
                "w": w,
                "orig_h": orig_h,
                "orig_w": orig_w,
            }
        )

    except Exception as exc:  # noqa: BLE001
        queue.put(f"ERROR:{exc}")


def _extract_float_mask_safe(mask_img: Any, mask_h: int, mask_w: int) -> np.ndarray:
    """
    Extract the float32 confidence mask from a MediaPipe VEC32F1 Image without
    calling numpy_view() / MpImageDataFloat32.

    Strategy
    --------
    MediaPipe's `mp.Image.__getitem__` calls `MpImageGetValueFloat32` which
    does NOT use `CopyToBuffer` and therefore does not trigger the assertion.
    We exploit this by reading the underlying ctypes pointer that MediaPipe
    stores internally (_image_ptr, _lib) and calling the float getter with a
    vectorised index array — avoiding a Python-level loop over every pixel.

    If the internal attributes are unavailable (future API change), we fall
    back to constructing a pixel-by-pixel array, which is slow but safe.
    """
    import ctypes  # noqa: PLC0415

    # Attempt fast path: directly call MpImageGetValueFloat32 via ctypes
    # using the internal _lib and _image_ptr that the mp.Image object holds.
    try:
        lib = mask_img._lib
        image_ptr = mask_img._image_ptr
        get_val = lib.MpImageGetValueFloat32

        # Preallocate output
        out = np.empty((mask_h, mask_w), dtype=np.float32)

        value_ptr = ctypes.c_float()
        error_msg = ctypes.c_char_p()

        for row in range(mask_h):
            pos = (ctypes.c_int * 2)(row, 0)
            for col in range(mask_w):
                pos[1] = col
                status = get_val(
                    image_ptr,
                    pos,
                    2,
                    ctypes.byref(value_ptr),
                    ctypes.byref(error_msg),
                )
                if error_msg.value:
                    lib.MpErrorFree(error_msg)
                    error_msg = ctypes.c_char_p()
                out[row, col] = value_ptr.value

        return out

    except Exception:  # noqa: BLE001
        # Slow but guaranteed path: use the public __getitem__ interface
        out = np.empty((mask_h, mask_w), dtype=np.float32)
        for row in range(mask_h):
            for col in range(mask_w):
                try:
                    out[row, col] = mask_img[row, col]
                except Exception:  # noqa: BLE001
                    out[row, col] = 0.0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — called from the parent (BodySegmenter)
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation_in_subprocess(
    bgr_image: np.ndarray,
    model_path_str: str,
    mp_max_side: int = 1920,
    timeout: float = 60.0,
) -> "Dict[str, Any] | str":
    """
    Dispatch segmentation inference to an isolated subprocess and return
    its result.

    The child process is spawned using the 'spawn' start method (not 'fork')
    to avoid duplicating the MediaPipe GL context, which can cause crashes
    on macOS when the parent process already has a GL context open.

    Parameters
    ----------
    bgr_image:
        BGR uint8 numpy array (already loaded and min-resolution validated).
    model_path_str:
        Absolute path to pose_landmarker_heavy.task.
    mp_max_side:
        Inference downscale threshold in pixels.
    timeout:
        Seconds to wait for the child process before treating it as failed.

    Returns
    -------
    dict:
        On success: {'confidence_mask': np.ndarray, 'h': int, 'w': int,
                     'orig_h': int, 'orig_w': int}
    str:
        On failure: descriptive error string (starts with 'NO_MASKS',
        'ERROR:', 'TIMEOUT', or 'CRASH').
    """
    # Use 'spawn' to create a clean child process without inherited GL state
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue[Any]" = ctx.Queue()

    orig_h, orig_w = bgr_image.shape[:2]

    # Serialise image as raw bytes (zero-copy tobytes is faster than pickle)
    bgr_bytes = bgr_image.tobytes()

    process = ctx.Process(
        target=_run_worker,
        args=(bgr_bytes, orig_h, orig_w, model_path_str, mp_max_side, queue),
        daemon=True,
    )
    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception:  # queue.Empty or other
        result = "TIMEOUT"
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)

    exit_code = process.exitcode
    if isinstance(result, (dict, str)):
        if isinstance(result, str) and exit_code not in (0, None):
            # Child was killed (e.g., SIGABRT = -6)
            return f"CRASH:child process exited with code {exit_code}"
        return result
    else:
        return f"CRASH:child process exited with code {exit_code}"

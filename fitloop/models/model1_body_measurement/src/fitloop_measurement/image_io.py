"""
image_io.py
-----------
Load, validate and normalise input images.

Public API
~~~~~~~~~~
    load_image(path)  -> np.ndarray          BGR uint8
    ImageLoadError                           raised on unrecoverable problems
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or fails basic sanity checks."""


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from *path* and return a BGR uint8 numpy array.

    Parameters
    ----------
    path:
        Filesystem path to the image file.

    Returns
    -------
    np.ndarray
        BGR uint8 array with shape ``(H, W, 3)``.

    Raises
    ------
    ImageLoadError
        If the file is missing, unsupported, unreadable, or below the minimum
        resolution required for reliable landmark detection.
    """
    path = Path(path)

    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported extension '{path.suffix}'. "
            f"Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    img = cv2.imread(str(path))
    if img is None:
        raise ImageLoadError(f"OpenCV could not decode the file: {path}")

    h, w = img.shape[:2]
    logger.debug("Loaded '%s'  shape=%dx%d", path.name, w, h)

    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        raise ImageLoadError(
            f"Image too small ({w}×{h} px). "
            f"Minimum required: {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT} px."
        )

    return img


def image_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 array to RGB uint8 (for MediaPipe / matplotlib)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_for_display(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Proportionally resize *img* so its longest side ≤ *max_side* pixels.

    The original array is returned unchanged if it already fits.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

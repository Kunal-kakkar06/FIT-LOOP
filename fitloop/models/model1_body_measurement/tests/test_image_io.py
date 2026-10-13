"""
tests/test_image_io.py
----------------------
Unit tests for fitloop_measurement.image_io.

Run with:  pytest tests/test_image_io.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.image_io import (
    ImageLoadError,
    image_to_rgb,
    load_image,
    resize_for_display,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(tmp_path: Path, w: int, h: int, name: str = "img.png") -> Path:
    """Create a minimal solid-colour PNG and return its path."""
    import cv2

    p = tmp_path / name
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return p


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_loads_valid_png(self, tmp_path):
        p = _write_png(tmp_path, 640, 960)
        img = load_image(p)
        assert isinstance(img, np.ndarray)
        assert img.shape == (960, 640, 3)

    def test_raises_on_missing_file(self):
        with pytest.raises(ImageLoadError, match="not found"):
            load_image("/nonexistent/path/image.jpg")

    def test_raises_on_unsupported_extension(self, tmp_path):
        p = tmp_path / "file.bmp"
        p.write_bytes(b"fakecontent")
        with pytest.raises(ImageLoadError, match="Unsupported extension"):
            load_image(p)

    def test_raises_on_too_small(self, tmp_path):
        # 100×100 is below the 480×640 minimum
        p = _write_png(tmp_path, 100, 100)
        with pytest.raises(ImageLoadError, match="too small"):
            load_image(p)

    def test_accepts_string_path(self, tmp_path):
        p = _write_png(tmp_path, 640, 960)
        img = load_image(str(p))  # str, not Path
        assert img is not None


# ---------------------------------------------------------------------------
# image_to_rgb
# ---------------------------------------------------------------------------

class TestImageToRgb:
    def test_channel_order_swapped(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel
        rgb = image_to_rgb(bgr)
        assert rgb[:, :, 2].mean() == 255   # should now be in Red channel
        assert rgb[:, :, 0].mean() == 0


# ---------------------------------------------------------------------------
# resize_for_display
# ---------------------------------------------------------------------------

class TestResizeForDisplay:
    def test_already_small_unchanged(self):
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        out = resize_for_display(img, max_side=800)
        assert out.shape == img.shape

    def test_large_image_resized(self):
        img = np.zeros((2000, 1200, 3), dtype=np.uint8)
        out = resize_for_display(img, max_side=800)
        assert max(out.shape[:2]) <= 800

    def test_aspect_ratio_preserved(self):
        img = np.zeros((1600, 800, 3), dtype=np.uint8)
        out = resize_for_display(img, max_side=800)
        original_ratio = 1600 / 800
        out_ratio = out.shape[0] / out.shape[1]
        assert abs(out_ratio - original_ratio) < 0.05

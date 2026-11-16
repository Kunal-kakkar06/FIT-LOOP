"""
FitLoop Garment Color Extraction Engine
=======================================
Production-grade OpenCV + K-Means color extractor that:
1. Isolates garment pixels by filtering background borders (white/black/transparent).
2. Converts pixels to CIE LAB color space for perceptually uniform clustering.
3. Performs K-Means clustering to extract dominant and secondary color centroids.
4. Computes percentage prominence of each color.
5. Maps centroids to human-readable fashion color names via Delta-E nearest-neighbor
   search against a comprehensive fashion palette.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Standard fashion color dictionary with representative HEX color codes.
# Supports multiple representative tones per fashion color category for robust matching.
DEFAULT_FASHION_PALETTE: Dict[str, List[str]] = {
    "Black": [
        "#000000",
        "#111111",
        "#1A1A1A",
        "#141414",
        "#1C1D20",
        "#222326",
        "#282A2E",
        "#2A2C30",
        "#2B2D32",
    ],
    "Charcoal": [
        "#45484C",
        "#4A4D52",
        "#505358",
        "#55585C",
        "#3F4246",
    ],
    "Slate Gray": ["#708090", "#6A7B82", "#5A6772"],
    "Light Gray": [
        "#C8C8C8",
        "#B8BEC5",
        "#C0C0C0",
        "#BBBFC5",
        "#B5B9BF",
    ],
    "White": [
        "#FFFFFF",
        "#FAFAFA",
        "#F8F9FA",
        "#F2F3F5",
        "#EDF0F3",
        "#E8ECF0",
        "#DDE3EA",
        "#D9E0E8",
    ],
    "Off-White": ["#FAF9F6", "#F5F5F0", "#F0EFEA"],
    "Ivory": ["#FFFFF0", "#FDFBF7", "#FFFFEE"],
    "Cream": ["#FFFDD0", "#FDFD96", "#FAF0E6"],
    "Beige": ["#F5F5DC", "#E8D8C8", "#E6D7B9", "#DDCBA4", "#E4D5B7"],
    "Khaki": ["#C3B091", "#BDB76B", "#C0B283", "#A89F91", "#9F8D69"],
    "Tan": ["#D2B48C", "#C8AD7F", "#B5996E", "#A07855"],
    "Camel": ["#C19A6B", "#C29B38", "#A67B5B", "#967144"],
    "Brown": ["#8B4513", "#654321", "#5C4033", "#4A2F1B", "#795548"],
    "Chocolate Brown": ["#3D2314", "#2B1700", "#3E2723", "#2E1503"],
    "Coffee": ["#4B3621", "#6F4E37", "#43302B"],
    "Rust": ["#B7410E", "#8B3103", "#A03C1B", "#8C2D19"],
    "Terracotta": ["#E2725B", "#CC4E3D", "#D9684C", "#D25D48"],
    "Burgundy": ["#800020", "#6B1D2F", "#58111A", "#4A0E17", "#641C2C"],
    "Maroon": ["#800000", "#8B0000", "#5C0606"],
    "Wine": ["#722F37", "#6A1A24", "#5E2129", "#4E1820"],
    "Red": ["#FF0000", "#EE1C25", "#E60000", "#D32F2F"],
    "Crimson": ["#DC143C", "#9E1B32", "#B80F0A"],
    "Scarlet": ["#FF2400", "#FF3419", "#E32636"],
    "Coral": ["#FF7F50", "#F88379", "#E06D53"],
    "Peach": ["#FFDAB9", "#FBC4AB", "#F7B299", "#FFE5B4"],
    "Pink": ["#FFC0CB", "#FFB6C1", "#FAAFBE"],
    "Blush": ["#DE5D83", "#E3A857", "#D49E9E", "#F4C2C2"],
    "Dusty Rose": ["#DCAE96", "#C59B8B", "#B76E79", "#D8A39D"],
    "Rose": ["#FF007F", "#E0115F", "#C21E56"],
    "Magenta": ["#FF00FF", "#CA1F7B", "#D10068"],
    "Fuchsia": ["#C154C1", "#A80077", "#FF1493"],
    "Purple": ["#800080", "#6A0DAD", "#4B0082", "#7B1FA2"],
    "Plum": ["#8E4585", "#701C46", "#5A1B43", "#673147"],
    "Violet": ["#8A2BE2", "#7F00FF", "#673AB7"],
    "Lavender": ["#E6E6FA", "#BDB5D5", "#C4AEAD", "#D1C4E9"],
    "Lilac": ["#C8A2C8", "#B695C0", "#BDA0CB"],
    "Navy Blue": [
        "#000080",
        "#1B263B",
        "#0A192F",
        "#0B1D3A",
        "#001F3F",
        "#1D2D44",
        "#0D1B2A",
        "#24305E",
    ],
    "Midnight Blue": ["#191970", "#0B1021", "#101820"],
    "Royal Blue": ["#4169E1", "#2B50EC", "#1E40AF", "#1565C0"],
    "Cobalt Blue": ["#0047AB", "#0020C2", "#003399"],
    "Denim Blue": [
        "#1560BD",
        "#2243B6",
        "#4D79A8",
        "#3B6998",
        "#4A6B82",
        "#506D84",
        "#2E536F",
    ],
    "Dark Denim": [
        "#3B4A5E",
        "#3C4651",
        "#2B3A4E",
        "#344557",
        "#2C3E50",
        "#4A5B6E",
    ],
    "Sky Blue": ["#87CEEB", "#6CA0DC", "#89CFF0", "#4FC3F7"],
    "Baby Blue": ["#89CFF0", "#A1CAF1", "#B0E0E6"],
    "Light Blue": [
        "#7E96B4",
        "#8AA3BC",
        "#91A9C8",
        "#8899B3",
        "#7B899E",
        "#859AB0",
    ],
    "Teal": ["#008080", "#005E5D", "#116D6E", "#006666"],
    "Turquoise": ["#40E0D0", "#30D5C8", "#20B2AA"],
    "Cyan": ["#00FFFF", "#00E5EE", "#26C6DA"],
    "Olive Green": [
        "#556B2F",
        "#6B8E23",
        "#4B5320",
        "#555D50",
        "#4D5645",
        "#5C6353",
    ],
    "Army Green": ["#4B5320", "#454B1B", "#3B443B", "#3A4B3A"],
    "Forest Green": ["#228B22", "#1E4620", "#194D33", "#2E7D32"],
    "Emerald Green": ["#50C878", "#097969", "#046307"],
    "Dark Green": ["#006400", "#013220", "#0B3B0B", "#1B5E20"],
    "Sage Green": ["#9CAF88", "#879782", "#7A8B7B", "#A3B18A"],
    "Mint Green": ["#98FF98", "#A2E8DD", "#90EE90", "#80CBC4"],
    "Lime Green": ["#32CD32", "#7FFF00", "#76FF03"],
    "Yellow": ["#FFFF00", "#FFEF00", "#FFEB3B"],
    "Mustard Yellow": ["#FFDB58", "#E1AD01", "#D4AF37", "#C69214", "#CCA01D"],
    "Gold": ["#FFD700", "#CFB53B", "#D4AF37"],
    "Ochre": ["#CC7722", "#B87333", "#C68642"],
    "Orange": ["#FFA500", "#FF8C00", "#E67E22", "#FF9800"],
    "Dark Orange": ["#FF8C00", "#D35400", "#E65100"],
    "Rust Orange": ["#C45508", "#C35214", "#B84A0E"],
}


class GarmentColorExtractor:
    """
    Color extraction engine using OpenCV + LAB space + K-Means clustering.
    Extracts dominant and secondary colors and maps them to standard fashion color names.
    """

    def __init__(
        self,
        color_palette: Optional[Dict[str, List[str]]] = None,
        max_processing_dim: int = 200,
    ) -> None:
        """
        Initialize the GarmentColorExtractor.

        Args:
            color_palette: Dictionary of color name -> list of representative HEX strings.
                           If None, uses DEFAULT_FASHION_PALETTE.
            max_processing_dim: Downscaled maximum image dimension for high-speed clustering.
        """
        self.palette = color_palette or DEFAULT_FASHION_PALETTE
        self.max_processing_dim = max_processing_dim
        self._precomputed_model: Dict[str, Tuple[List[np.ndarray], str]] = {}
        self._precompute_lab_palette()

    def _precompute_lab_palette(self) -> None:
        """Precompute CIE LAB float coordinates for all palette colors."""
        for name, hexes in self.palette.items():
            lab_vectors = []
            for hx in hexes:
                r, g, b = self.hex_to_rgb(hx)
                bgr_f = np.float32([[[b / 255.0, g / 255.0, r / 255.0]]])
                lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB)[0, 0]
                lab_vectors.append(lab)
            # Store lab vectors and primary representative hex
            self._precomputed_model[name] = (lab_vectors, hexes[0])

    @staticmethod
    def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """Convert a HEX string (#RRGGBB) to (R, G, B) tuple."""
        clean = hex_code.lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", clean):
            raise ValueError(f"Invalid 6-character hex color code: '{hex_code}'")
        return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)

    @staticmethod
    def rgb_to_hex(rgb: Union[Tuple[int, int, int], List[int], np.ndarray]) -> str:
        """Convert (R, G, B) to uppercase '#RRGGBB' string."""
        r, g, b = int(round(rgb[0])), int(round(rgb[1])), int(round(rgb[2]))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02X}{g:02X}{b:02X}"

    def map_color_name(
        self,
        rgb: Union[Tuple[int, int, int], List[int], np.ndarray],
    ) -> Tuple[str, str, float]:
        """
        Map an RGB color to the closest human-readable fashion color name using CIE Delta-E.

        Returns:
            Tuple of (color_name, reference_hex, delta_e_distance).
        """
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        target_bgr_f = np.float32([[[b, g, r]]])
        target_lab = cv2.cvtColor(target_bgr_f, cv2.COLOR_BGR2LAB)[0, 0]

        best_name = "Unknown"
        best_hex = "#000000"
        min_dist = float("inf")

        for name, (lab_vectors, ref_hex) in self._precomputed_model.items():
            for ref_lab in lab_vectors:
                # Euclidean distance in CIE LAB space corresponds to Delta-E 76
                dist = float(np.linalg.norm(target_lab - ref_lab))
                if dist < min_dist:
                    min_dist = dist
                    best_name = name
                    best_hex = ref_hex

        return best_name, best_hex, round(min_dist, 2)

    def _load_image(
        self,
        image_input: Union[np.ndarray, str, Path],
    ) -> np.ndarray:
        """Load and validate an input image."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image file does not exist: {p}")
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {p}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if img.size == 0 or img.ndim < 2:
            raise ValueError("Input image array is empty or has invalid dimensions.")

        return img

    def _apply_bbox(
        self,
        img: np.ndarray,
        bbox: Optional[Union[Tuple, List]],
    ) -> np.ndarray:
        """Crop image to bounding box if provided."""
        if bbox is None:
            return img

        h, w = img.shape[:2]

        # Handle YOLO normalized format (x_c, y_c, bw, bh) if values are all <= 1.0
        if len(bbox) == 4 and all(0.0 <= float(v) <= 1.0 for v in bbox):
            xc, yc, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x1 = int(round((xc - bw / 2.0) * w))
            y1 = int(round((yc - bh / 2.0) * h))
            x2 = int(round((xc + bw / 2.0) * w))
            y2 = int(round((yc + bh / 2.0) * h))
        elif len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected 4 coordinates.")

        # Clamp to image boundaries
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            raise ValueError(f"Cropped bbox region is empty: ({x1}, {y1}, {x2}, {y2})")

        return cropped

    def _filter_garment_pixels(
        self,
        img_bgr: np.ndarray,
        alpha_mask: Optional[np.ndarray],
        ignore_background: bool,
    ) -> Tuple[np.ndarray, int]:
        """
        Filter out background border pixels (pure white, pure black padding, transparent).
        Returns filtered BGR float32 array and total valid pixel count.
        """
        bgr_f = img_bgr.astype(np.float32) / 255.0
        total_pixels = bgr_f.shape[0] * bgr_f.shape[1]

        if not ignore_background:
            if alpha_mask is not None:
                mask = alpha_mask > 0.2
            else:
                mask = np.ones((bgr_f.shape[0], bgr_f.shape[1]), dtype=bool)
            filtered = bgr_f[mask]
            return filtered, total_pixels

        # Build composite mask
        # 1. Ignore pure white background (all channels > 0.94 / RGB > 240)
        is_white = (bgr_f[:, :, 0] > 0.94) & (bgr_f[:, :, 1] > 0.94) & (bgr_f[:, :, 2] > 0.94)

        # 2. Ignore pure black background padding (all channels < 0.05 / RGB < 13)
        is_black = (bgr_f[:, :, 0] < 0.05) & (bgr_f[:, :, 1] < 0.05) & (bgr_f[:, :, 2] < 0.05)

        mask = ~(is_white | is_black)

        if alpha_mask is not None:
            mask = mask & (alpha_mask > 0.2)

        # Safety Fallback: If filtering discarded more than 96% of pixels (e.g. solid white/black garment)
        # fallback to using all non-transparent pixels so we never return an empty set.
        if np.sum(mask) < 0.04 * total_pixels:
            if alpha_mask is not None:
                mask = alpha_mask > 0.2
            else:
                mask = np.ones((bgr_f.shape[0], bgr_f.shape[1]), dtype=bool)

        filtered = bgr_f[mask]
        return filtered, len(filtered)

    def extract(
        self,
        image: Union[np.ndarray, str, Path],
        bbox: Optional[Union[Tuple, List]] = None,
        k: int = 5,
        ignore_background: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract dominant and secondary colors from a garment image.

        Args:
            image: Image as file path (str/Path) or NumPy array (BGR/BGRA).
            bbox: Optional garment bounding box (x1, y1, x2, y2) or normalized YOLO (xc, yc, w, h).
            k: Number of K-Means clusters (clamped between 2 and 8).
            ignore_background: Whether to filter out white/black/transparent background.

        Returns:
            Structured dictionary with 'dominant_color', 'secondary_color', and full 'palette'.
        """
        img = self._load_image(image)
        img = self._apply_bbox(img, bbox)

        # Extract alpha channel if present
        alpha_mask = None
        if img.ndim == 3 and img.shape[2] == 4:
            alpha_mask = (img[:, :, 3].astype(np.float32) / 255.0)
            img_bgr = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 3:
            img_bgr = img
        elif img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # Downscale for performance if larger than max_processing_dim
        h, w = img_bgr.shape[:2]
        if max(h, w) > self.max_processing_dim:
            scale = self.max_processing_dim / float(max(h, w))
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if alpha_mask is not None:
                alpha_mask = cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Filter background
        filtered_bgr, total_garment_pixels = self._filter_garment_pixels(
            img_bgr, alpha_mask, ignore_background
        )

        if len(filtered_bgr) == 0:
            raise ValueError("No garment pixels remained after background filtering.")

        # Convert pixels to LAB space
        # Reshape to (1, N, 3) for cv2.cvtColor
        filtered_bgr_2d = filtered_bgr.reshape((1, -1, 3))
        filtered_lab = cv2.cvtColor(filtered_bgr_2d, cv2.COLOR_BGR2LAB)[0]

        # Clamp k to available pixels and bounds
        num_pixels = len(filtered_lab)
        effective_k = max(2, min(k, 8, num_pixels))

        # Perform K-Means clustering in LAB space
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(
            filtered_lab.astype(np.float32),
            effective_k,
            None,
            criteria,
            10,
            flags,
        )

        # Count pixel prominence per cluster
        counts = np.bincount(labels.flatten(), minlength=effective_k)
        sorted_indices = np.argsort(counts)[::-1]

        palette: List[Dict[str, Any]] = []

        for rank, idx in enumerate(sorted_indices, start=1):
            cluster_count = int(counts[idx])
            if cluster_count == 0:
                continue

            pct = round((cluster_count / float(num_pixels)) * 100.0, 2)
            center_lab = centers[idx]

            # Convert LAB centroid back to BGR float -> RGB uint8
            center_bgr_f = cv2.cvtColor(np.float32([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
            center_bgr_clipped = np.clip(center_bgr_f * 255.0, 0, 255).round().astype(np.uint8)

            r = int(center_bgr_clipped[2])
            g = int(center_bgr_clipped[1])
            b = int(center_bgr_clipped[0])
            hex_code = self.rgb_to_hex((r, g, b))

            # Map to human-readable fashion color name
            name, ref_hex, delta_e = self.map_color_name((r, g, b))

            palette.append({
                "rank": rank,
                "name": name,
                "hex": hex_code,
                "percentage": pct,
                "rgb": [r, g, b],
                "delta_e": delta_e,
                "pixel_count": cluster_count,
            })

        if not palette:
            raise RuntimeError("Clustering failed to produce any valid color entries.")

        dominant = palette[0]
        secondary = palette[1] if len(palette) > 1 else palette[0]

        return {
            "dominant_color": {
                "name": dominant["name"],
                "hex": dominant["hex"],
                "percentage": dominant["percentage"],
                "rgb": dominant["rgb"],
            },
            "secondary_color": {
                "name": secondary["name"],
                "hex": secondary["hex"],
                "percentage": secondary["percentage"],
                "rgb": secondary["rgb"],
            },
            "palette": palette,
            "total_garment_pixels": total_garment_pixels,
        }

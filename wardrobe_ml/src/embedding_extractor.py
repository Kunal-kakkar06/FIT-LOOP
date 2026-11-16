"""
FitLoop Visual Similarity Embedding Extractor
==============================================
Uses torchvision MobileNetV3-Small (pretrained ImageNet backbone with classification
head removed) to extract compact, L2-normalized 576-dimensional feature representations
from garment bounding-box crops for visual similarity matching and duplicate detection.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.models as models

logger = logging.getLogger("fitloop.embedding_extractor")


class GarmentEmbeddingExtractor:
    """
    Extracts deep visual feature embeddings from garment image crops using
    MobileNetV3-Small backbone on CPU.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize MobileNetV3-Small feature extractor.

        Args:
            device: Compute device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model_version = "mobilenet_v3_small"
        self.embedding_dim = 576

        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            self.model = models.mobilenet_v3_small(pretrained=True)

        # Replace classification head with Identity to extract pooled features
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        # ImageNet normalization parameters
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(3, 1, 1)

    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]] = None,
    ) -> List[float]:
        """
        Extract an L2-normalized 576-dimensional embedding from a garment image or crop.

        Args:
            image: Full image or crop in BGR/RGB uint8 format.
            bbox: Optional [x1, y1, x2, y2] bounding box coordinates.

        Returns:
            List of 576 floats normalized to unit L2 length (||v|| = 1.0).
        """
        if image is None or image.size == 0:
            return [0.0] * self.embedding_dim

        # Crop if bbox is specified
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            cx1 = max(0, min(w - 1, int(x1)))
            cy1 = max(0, min(h - 1, int(y1)))
            cx2 = max(cx1 + 1, min(w, int(x2)))
            cy2 = max(cy1 + 1, min(h, int(y2)))
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                image = crop

        # Color space conversion to RGB
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to standard MobileNet input resolution (224x224)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert to tensor: (H, W, C) -> (C, H, W) normalized to [0, 1]
        t = torch.from_numpy(resized).permute(2, 0, 1).float().to(self.device) / 255.0

        # Apply ImageNet normalization
        t = ((t - self._mean) / self._std).unsqueeze(0)

        with torch.no_grad():
            features = self.model(t).squeeze(0).cpu().numpy()

        # L2-normalize feature vector
        norm = float(np.linalg.norm(features))
        if norm < 1e-12:
            return [0.0] * self.embedding_dim

        normalized = (features / norm).astype(float)
        return normalized.tolist()

    @staticmethod
    def compute_cosine_similarity(
        emb1: Optional[Union[List[float], np.ndarray]],
        emb2: Optional[Union[List[float], np.ndarray]],
    ) -> float:
        """
        Compute cosine similarity between two feature embeddings,
        strictly bounded within [0.0, 1.0].

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity score rounded to 4 decimals in [0.0, 1.0].
        """
        if emb1 is None or emb2 is None:
            return 0.0

        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)

        if v1.size == 0 or v2.size == 0:
            return 0.0

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0

        dot_val = float(np.dot(v1, v2) / (norm1 * norm2))
        return round(float(np.clip(dot_val, 0.0, 1.0)), 4)

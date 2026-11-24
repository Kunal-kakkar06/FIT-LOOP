"""
FitLoop Duplicate Detection & Wardrobe Similarity Engine
=========================================================
Computes multi-attribute similarity scores (0.0 to 1.0) between a candidate garment
and a user's existing wardrobe inventory using:
1. Strict and group-based category matching.
2. Perceptually uniform CIE LAB Delta-E color distance + semantic color family matching.
3. Secondary color and pattern alignment (with dynamic weight rebalancing).
4. Tri-tier verdict categorization:
   - DUPLICATE (score >= 0.85)
   - SIMILAR   (score 0.65 - 0.84)
   - UNIQUE    (score < 0.65)
5. Explanatory human-readable recommendation messages.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Map fashion colors to broader color families for semantic similarity
COLOR_FAMILIES: Dict[str, str] = {
    "Black": "black_gray",
    "Charcoal": "black_gray",
    "Slate Gray": "black_gray",
    "Light Gray": "black_gray",
    "White": "white_cream",
    "Off-White": "white_cream",
    "Ivory": "white_cream",
    "Cream": "white_cream",
    "Beige": "beige_brown",
    "Khaki": "beige_brown",
    "Tan": "beige_brown",
    "Camel": "beige_brown",
    "Brown": "beige_brown",
    "Chocolate Brown": "beige_brown",
    "Coffee": "beige_brown",
    "Rust": "orange_rust",
    "Terracotta": "orange_rust",
    "Orange": "orange_rust",
    "Dark Orange": "orange_rust",
    "Rust Orange": "orange_rust",
    "Burgundy": "red_wine",
    "Maroon": "red_wine",
    "Wine": "red_wine",
    "Red": "red_wine",
    "Crimson": "red_wine",
    "Scarlet": "red_wine",
    "Coral": "pink_coral",
    "Peach": "pink_coral",
    "Pink": "pink_coral",
    "Blush": "pink_coral",
    "Dusty Rose": "pink_coral",
    "Rose": "pink_coral",
    "Magenta": "purple_magenta",
    "Fuchsia": "purple_magenta",
    "Purple": "purple_magenta",
    "Plum": "purple_magenta",
    "Violet": "purple_magenta",
    "Lavender": "purple_magenta",
    "Lilac": "purple_magenta",
    "Navy Blue": "blue",
    "Midnight Blue": "blue",
    "Royal Blue": "blue",
    "Cobalt Blue": "blue",
    "Denim Blue": "blue",
    "Sky Blue": "blue",
    "Baby Blue": "blue",
    "Cyan": "blue",
    "Teal": "green",
    "Turquoise": "green",
    "Olive Green": "green",
    "Army Green": "green",
    "Forest Green": "green",
    "Emerald Green": "green",
    "Dark Green": "green",
    "Sage Green": "green",
    "Mint Green": "green",
    "Lime Green": "green",
    "Yellow": "yellow",
    "Mustard Yellow": "yellow",
    "Gold": "yellow",
    "Ochre": "yellow",
}

# Compatibility similarity between related sub-categories
CATEGORY_RELATED_SCORES: Dict[Tuple[str, str], float] = {
    ("short sleeve top", "long sleeve top"): 0.50,
    ("long sleeve top", "short sleeve top"): 0.50,
    ("vest", "short sleeve top"): 0.40,
    ("short sleeve top", "vest"): 0.40,
    ("vest", "long sleeve top"): 0.35,
    ("long sleeve top", "vest"): 0.35,
    ("shorts", "trousers"): 0.40,
    ("trousers", "shorts"): 0.40,
    ("short sleeve outwear", "long sleeve outwear"): 0.50,
    ("long sleeve outwear", "short sleeve outwear"): 0.50,
    ("short sleeve dress", "long sleeve dress"): 0.50,
    ("long sleeve dress", "short sleeve dress"): 0.50,
    ("vest dress", "short sleeve dress"): 0.40,
    ("short sleeve dress", "vest dress"): 0.40,
}


class DuplicateDetector:
    """
    Multi-attribute garment similarity and duplicate detection engine.
    """

    # Scoring thresholds
    DUPLICATE_THRESHOLD: float = 0.85
    SIMILAR_THRESHOLD: float = 0.65

    # Base attribute weights
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "category": 0.50,
        "dominant_color": 0.40,
        "secondary_color": 0.05,
        "pattern": 0.05,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        duplicate_threshold: float = 0.85,
        similar_threshold: float = 0.65,
    ) -> None:
        """
        Initialize the DuplicateDetector.

        Args:
            weights: Optional custom dictionary of weights for attributes.
            duplicate_threshold: Minimum similarity score for 'DUPLICATE' verdict (default: 0.85).
            similar_threshold: Minimum similarity score for 'SIMILAR' verdict (default: 0.65).
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.duplicate_threshold = duplicate_threshold
        self.similar_threshold = similar_threshold

    @staticmethod
    def hex_to_lab(hex_code: str) -> np.ndarray:
        """Convert a HEX string (#RRGGBB) to CIE LAB float32 coordinate."""
        clean = hex_code.lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", clean):
            clean = "000000"
        r = int(clean[0:2], 16) / 255.0
        g = int(clean[2:4], 16) / 255.0
        b = int(clean[4:6], 16) / 255.0
        bgr_f = np.float32([[[b, g, r]]])
        return cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB)[0, 0]

    @staticmethod
    def _normalize_color_input(color_input: Any) -> Tuple[str, str]:
        """
        Extract (hex_code, color_name) from varied color inputs:
        - dict: {"name": "Navy Blue", "hex": "#1B263B"}
        - str: "#1B263B" or "Navy Blue"
        """
        if isinstance(color_input, dict):
            hx = str(color_input.get("hex", "#000000"))
            name = str(color_input.get("name", "Unknown"))
            return hx, name
        elif isinstance(color_input, str):
            if color_input.startswith("#"):
                return color_input, "Unknown"
            else:
                return "#000000", color_input
        return "#000000", "Unknown"

    def compute_category_similarity(self, cat1: str, cat2: str) -> float:
        """
        Compute similarity between two garment categories:
        - Exact match: 1.0
        - Related sub-category (e.g. short sleeve top vs long sleeve top): 0.35 to 0.50
        - Cross-category (e.g. tops vs bottoms, skirt vs dress): 0.0
        """
        c1 = str(cat1).strip().lower()
        c2 = str(cat2).strip().lower()

        if c1 == c2:
            return 1.0

        return CATEGORY_RELATED_SCORES.get((c1, c2), 0.0)

    def compute_color_similarity(
        self,
        color1: Any,
        color2: Any,
    ) -> Tuple[float, float]:
        """
        Compute perceptual + semantic color similarity between two colors.

        Returns:
            Tuple of (color_similarity_score [0.0 to 1.0], delta_e_distance).
        """
        hex1, name1 = self._normalize_color_input(color1)
        hex2, name2 = self._normalize_color_input(color2)

        lab1 = self.hex_to_lab(hex1)
        lab2 = self.hex_to_lab(hex2)
        de = float(np.linalg.norm(lab1 - lab2))

        # Exact match or imperceptible difference
        if de < 2.0 and (name1 == "Unknown" or name2 == "Unknown" or name1.lower() == name2.lower()):
            return 1.0, round(de, 2)

        # Delta-E continuous similarity curve
        # deltaE = 0 -> 1.0; deltaE = 12 -> 0.50; deltaE >= 24 -> 0.0
        metric_sim = max(0.0, 1.0 - (de / 24.0))

        # Semantic name and family matching
        fam1 = COLOR_FAMILIES.get(name1)
        fam2 = COLOR_FAMILIES.get(name2)
        if name1.lower() == name2.lower() and name1 != "Unknown":
            name_sim = 1.0
        elif fam1 and fam2 and fam1 == fam2:
            name_sim = 0.65
        else:
            name_sim = 0.0

        # Weighted combination of metric and semantic alignment
        sim = float(np.clip(0.60 * metric_sim + 0.40 * name_sim, 0.0, 1.0))
        return round(sim, 3), round(de, 2)

    def compute_pattern_similarity(self, pat1: Optional[str], pat2: Optional[str]) -> float:
        """Compute pattern similarity (solid, striped, plaid, etc.)."""
        p1 = str(pat1 or "solid").strip().lower()
        p2 = str(pat2 or "solid").strip().lower()
        if p1 == p2:
            return 1.0
        return 0.30

    @staticmethod
    def compute_visual_similarity(
        emb1: Optional[Union[List[float], np.ndarray]],
        emb2: Optional[Union[List[float], np.ndarray]],
    ) -> Optional[float]:
        """
        Compute cosine similarity between two visual embedding vectors.
        Returns float strictly bounded in [0.0, 1.0], or None if either embedding is missing.
        """
        if emb1 is None or emb2 is None:
            return None
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        if v1.size == 0 or v2.size == 0:
            return None
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
        return round(float(np.clip(cos_sim, 0.0, 1.0)), 4)

    def compute_item_similarity(
        self,
        candidate: Dict[str, Any],
        inventory_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute composite multi-attribute similarity between candidate and an inventory item.
        """
        # 1. Category Similarity
        cat_cand = candidate.get("category", "")
        cat_inv = inventory_item.get("category", "")
        cat_sim = self.compute_category_similarity(cat_cand, cat_inv)

        # 2. Dominant Color Similarity
        dom_sim, de_dom = self.compute_color_similarity(
            candidate.get("dominant_color"),
            inventory_item.get("dominant_color"),
        )

        # 3. Visual Feature Embedding Similarity
        emb_cand = candidate.get("embedding")
        emb_inv = inventory_item.get("embedding")
        vis_sim = self.compute_visual_similarity(emb_cand, emb_inv)

        # Cross-category rule: If items belong to completely different functional categories
        # (e.g. Top vs Trousers, Skirt vs Jacket), they cannot be duplicates or close substitutes.
        if cat_sim == 0.0:
            # Heavily attenuated score ensuring it stays strictly below UNIQUE ceiling (< 0.65)
            if vis_sim is not None:
                composite_score = round(float(np.clip(0.20 * dom_sim + 0.10 * vis_sim, 0.0, 1.0)), 3)
            else:
                composite_score = round(float(np.clip(0.30 * dom_sim, 0.0, 1.0)), 3)

            return {
                "similarity_score": composite_score,
                "category_similarity": 0.0,
                "dominant_color_similarity": dom_sim,
                "secondary_color_similarity": 0.0,
                "pattern_similarity": 0.0,
                "visual_similarity": vis_sim,
                "delta_e": de_dom,
            }

        # Check which optional attributes are active
        sec_sim = 0.0
        pat_sim = 0.0
        has_secondary = bool(candidate.get("secondary_color") and inventory_item.get("secondary_color"))
        has_pattern = bool(candidate.get("pattern") and inventory_item.get("pattern"))

        if has_secondary:
            sec_sim, _ = self.compute_color_similarity(
                candidate.get("secondary_color"),
                inventory_item.get("secondary_color"),
            )

        if has_pattern:
            pat_sim = self.compute_pattern_similarity(
                candidate.get("pattern"),
                inventory_item.get("pattern"),
            )

        # Weight selection: incorporates visual_similarity when available
        if vis_sim is not None:
            if has_secondary and has_pattern:
                active_weights = {
                    "category": 0.35,
                    "dominant_color": 0.25,
                    "visual_similarity": 0.25,
                    "secondary_color": 0.10,
                    "pattern": 0.05,
                }
            elif has_secondary:
                active_weights = {
                    "category": 0.35,
                    "dominant_color": 0.25,
                    "visual_similarity": 0.25,
                    "secondary_color": 0.15,
                }
            elif has_pattern:
                active_weights = {
                    "category": 0.40,
                    "dominant_color": 0.25,
                    "visual_similarity": 0.30,
                    "pattern": 0.05,
                }
            else:
                active_weights = {
                    "category": 0.40,
                    "dominant_color": 0.30,
                    "visual_similarity": 0.30,
                }
        else:
            # Backward-compatible weights when visual embeddings are not present
            if has_secondary and has_pattern:
                active_weights = {
                    "category": 0.45,
                    "dominant_color": 0.40,
                    "secondary_color": 0.10,
                    "pattern": 0.05,
                }
            elif has_secondary:
                active_weights = {
                    "category": 0.45,
                    "dominant_color": 0.40,
                    "secondary_color": 0.15,
                }
            elif has_pattern:
                active_weights = {
                    "category": 0.50,
                    "dominant_color": 0.45,
                    "pattern": 0.05,
                }
            else:
                active_weights = {
                    "category": 0.50,
                    "dominant_color": 0.50,
                }

        # Calculate composite score based on active attributes
        score = (
            active_weights["category"] * cat_sim
            + active_weights["dominant_color"] * dom_sim
            + active_weights.get("visual_similarity", 0.0) * (vis_sim or 0.0)
            + active_weights.get("secondary_color", 0.0) * sec_sim
            + active_weights.get("pattern", 0.0) * pat_sim
        )
        composite_score = round(float(np.clip(score, 0.0, 1.0)), 3)

        return {
            "similarity_score": composite_score,
            "category_similarity": cat_sim,
            "dominant_color_similarity": dom_sim,
            "secondary_color_similarity": sec_sim,
            "pattern_similarity": pat_sim,
            "visual_similarity": vis_sim,
            "delta_e": de_dom,
        }

    def generate_recommendation(
        self,
        verdict: str,
        score: float,
        matched_item: Optional[Dict[str, Any]],
        candidate_item: Dict[str, Any],
    ) -> str:
        """Generate human-readable recommendation message explaining the verdict."""
        cand_name = candidate_item.get("name") or f"{candidate_item.get('category', 'Garment')}"
        if isinstance(candidate_item.get("dominant_color"), dict):
            c_col = candidate_item["dominant_color"].get("name", "")
            if c_col and c_col.lower() not in cand_name.lower():
                cand_name = f"{c_col} {cand_name}".strip()

        if verdict == "DUPLICATE" and matched_item:
            inv_name = matched_item.get("name") or matched_item.get("category", "garment")
            inv_date = matched_item.get("purchase_date")
            date_clause = f" on {inv_date}" if inv_date else ""
            return (
                f"DUPLICATE DETECTED: You already own an identical or near-identical {inv_name}{date_clause} "
                f"(similarity: {score*100:.0f}%). We recommend skipping this purchase or selecting an alternative color."
            )
        elif verdict == "SIMILAR" and matched_item:
            inv_name = matched_item.get("name") or matched_item.get("category", "garment")
            inv_date = matched_item.get("purchase_date")
            date_clause = f" (purchased on {inv_date})" if inv_date else ""
            return (
                f"SIMILAR ITEM OWNED: You already own a very similar {inv_name}{date_clause} "
                f"(similarity: {score*100:.0f}%). This candidate item shares the same color family and category."
            )
        else:
            return (
                f"UNIQUE ADDITION: This {cand_name} is distinct from your existing wardrobe "
                f"(highest match: {score*100:.0f}%). Safe to add to your collection!"
            )

    def check_duplicate(
        self,
        candidate_item: Dict[str, Any],
        wardrobe_inventory: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate item against the user's complete wardrobe inventory.

        Args:
            candidate_item: Dictionary with 'category', 'dominant_color', etc.
            wardrobe_inventory: List of existing owned garment dictionaries.

        Returns:
            Structured dictionary with verdict, similarity score, top match, and breakdown.
        """
        if not wardrobe_inventory:
            return {
                "verdict": "UNIQUE",
                "similarity_score": 0.0,
                "matched_item": None,
                "breakdown": None,
                "recommendation": "UNIQUE ADDITION: Your wardrobe is currently empty. Safe to add!",
                "all_matches": [],
            }

        evaluated_matches = []

        for idx, inv_item in enumerate(wardrobe_inventory):
            sim_info = self.compute_item_similarity(candidate_item, inv_item)
            item_id = inv_item.get("id", f"item_{idx+1:02d}")
            evaluated_matches.append({
                "item_id": item_id,
                "item": inv_item,
                **sim_info,
            })

        # Sort matches by similarity score descending
        evaluated_matches.sort(key=lambda m: m["similarity_score"], reverse=True)
        best_match = evaluated_matches[0]
        max_score = best_match["similarity_score"]

        # Determine verdict
        if max_score >= self.duplicate_threshold:
            verdict = "DUPLICATE"
        elif max_score >= self.similar_threshold:
            verdict = "SIMILAR"
        else:
            verdict = "UNIQUE"

        recommendation = self.generate_recommendation(
            verdict=verdict,
            score=max_score,
            matched_item=best_match["item"],
            candidate_item=candidate_item,
        )

        return {
            "verdict": verdict,
            "similarity_score": max_score,
            "matched_item": best_match["item"],
            "breakdown": {
                "category_similarity": best_match["category_similarity"],
                "dominant_color_similarity": best_match["dominant_color_similarity"],
                "secondary_color_similarity": best_match["secondary_color_similarity"],
                "pattern_similarity": best_match["pattern_similarity"],
                "visual_similarity": best_match.get("visual_similarity"),
                "delta_e": best_match["delta_e"],
            },
            "recommendation": recommendation,
            "all_matches": evaluated_matches,
        }

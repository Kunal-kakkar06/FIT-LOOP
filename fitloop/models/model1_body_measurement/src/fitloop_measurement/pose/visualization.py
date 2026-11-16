"""
pose/visualization.py
---------------------
Overlay canonical MediaPipe pose landmarks and anatomical connections on an image.
Does not mutate the input image or save files automatically.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

from .schemas import PoseExtractionResult, PoseLandmarkPoint


def draw_pose_landmarks(
    bgr_image: np.ndarray,
    pose_result: PoseExtractionResult,
    min_visibility: float = 0.5,
    point_color: tuple[int, int, int] = (0, 255, 0),        # BGR Green
    connection_color: tuple[int, int, int] = (0, 140, 255), # BGR Orange
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw landmarks and skeleton connections on a non-mutated copy of the input image.

    Parameters
    ----------
    bgr_image:
        Input BGR numpy array.
    pose_result:
        PoseExtractionResult containing the extracted landmarks.
    min_visibility:
        Minimum landmark visibility threshold required to draw a point or connection.
    point_color:
        BGR tuple for landmark circles.
    connection_color:
        BGR tuple for skeleton connection lines.
    point_radius:
        Pixel radius of landmark circles.
    line_thickness:
        Pixel thickness of connection lines.

    Returns
    -------
    np.ndarray
        New BGR image array with overlays.
    """
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Cannot draw pose landmarks on empty or None image.")

    # Never mutate the caller's input array
    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]

    if not pose_result.landmarks or not pose_result.is_successful:
        return annotated

    # Map landmarks by index
    lm_coords: dict[int, tuple[int, int]] = {}
    for lm in pose_result.landmarks:
        if lm.is_usable(min_visibility=min_visibility):
            cx = int(round(lm.x * (w - 1)))
            cy = int(round(lm.y * (h - 1)))
            # Clamp coordinates to visible frame
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            lm_coords[lm.index] = (cx, cy)

    # 1. Draw anatomical skeleton connections
    try:
        connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        for edge in connections:
            start_idx = edge.start
            end_idx = edge.end
            if start_idx in lm_coords and end_idx in lm_coords:
                pt1 = lm_coords[start_idx]
                pt2 = lm_coords[end_idx]
                cv2.line(annotated, pt1, pt2, connection_color, line_thickness, cv2.LINE_AA)
    except Exception:
        # Fallback if connection definition fails
        pass

    # 2. Draw landmark points
    for idx, (cx, cy) in lm_coords.items():
        cv2.circle(annotated, (cx, cy), point_radius, point_color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated

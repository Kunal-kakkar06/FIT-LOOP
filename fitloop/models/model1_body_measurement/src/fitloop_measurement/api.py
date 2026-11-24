"""
api.py
------
FastAPI application for FitLoop Model 1 testing interface.

Provides:
- GET  /health -> Service health and model status
- POST /api/v1/measurements/estimate -> Dual-view circumference inference

Configures CORS strictly for local Vite frontend origin.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fitloop_measurement.config import HEIGHT_MAX_CM, HEIGHT_MIN_CM
from fitloop_measurement.service import MeasurementPipelineService, decode_image_bytes

logger = logging.getLogger("fitloop_api")

# Maximum upload limit per image (15 MB)
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="FitLoop Model 1 Circumference Estimation API",
    version="1.0.0",
    description="Inference API exposing calibrated 2D circumference estimation for Model 1.",
)

# Configure CORS strictly for Vite dev server (default port 5173, fallback 5174)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global or lazily managed pipeline service instance
_pipeline_service: Optional[MeasurementPipelineService] = None


def get_pipeline_service() -> MeasurementPipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = MeasurementPipelineService()
    return _pipeline_service


@app.get("/health", tags=["System"])
async def health_check():
    """Lightweight health status endpoint."""
    return {
        "status": "healthy",
        "service": "fitloop-model1-measurement",
        "version": "1.0.0",
        "model_pipeline": "Stage 4B Calibrated Circumference Estimator",
    }


@app.post(
    "/api/v1/measurements/estimate",
    tags=["Estimation"],
    summary="Estimate circumferences from front and side captures",
)
async def estimate_measurements(
    front_image: UploadFile = File(..., description="Front-facing unclipped photograph"),
    side_image: UploadFile = File(..., description="Side-profile unclipped photograph"),
    standing_height_cm: float = Form(..., description="Barefoot stature in centimetres"),
):
    """Run dual-view calibrated circumference estimation.

    Validates:
    - Required files present
    - Supported content-types (image/jpeg, image/png, image/webp)
    - File size limits (≤ 15MB)
    - Stature bounds [100.0, 250.0] cm
    """
    # 1. Validate height bounds
    if standing_height_cm < HEIGHT_MIN_CM or standing_height_cm > HEIGHT_MAX_CM:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Standing height ({standing_height_cm} cm) is outside valid human stature range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm.",
        )

    # 2. Validate MIME types
    if front_image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Front image content-type '{front_image.content_type}' is unsupported. Allowed: JPEG, PNG, WebP.",
        )
    if side_image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Side image content-type '{side_image.content_type}' is unsupported. Allowed: JPEG, PNG, WebP.",
        )

    # 3. Read image bytes and validate file sizes
    front_bytes = await front_image.read()
    if len(front_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Front image exceeds maximum allowed upload size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
        )

    side_bytes = await side_image.read()
    if len(side_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Side image exceeds maximum allowed upload size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
        )

    # 4. Decode images via OpenCV
    try:
        bgr_front = decode_image_bytes(front_bytes, "front")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    try:
        bgr_side = decode_image_bytes(side_bytes, "side")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # 5. Execute pipeline
    service = get_pipeline_service()
    try:
        result_dict = service.process_pair(
            front_image=bgr_front,
            side_image=bgr_side,
            standing_height_cm=standing_height_cm,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error during pipeline execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )

    return JSONResponse(content=result_dict)

"""
config.py
---------
Central configuration constants for the measurement engine.
All tuneable thresholds live here so no magic numbers appear in logic modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
# Resolved relative to this file so the package is location-independent.
_PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL1_ROOT   = _PACKAGE_ROOT.parents[1]          # model1_body_measurement/

DATA_DIR          = MODEL1_ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
GROUND_TRUTH_DIR  = DATA_DIR / "ground_truth"

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Image quality thresholds
# ---------------------------------------------------------------------------
MIN_IMAGE_WIDTH  = 480    # pixels
MIN_IMAGE_HEIGHT = 640    # pixels

# Fraction of image height that the detected person must occupy.
MIN_BODY_HEIGHT_FRACTION = 0.70

# Maximum horizontal offset of the body midpoint from image centre (fraction).
MAX_CENTER_OFFSET_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------
# The known height of the subject (cm) must be supplied by the caller.
# This constant is a sanity-check range only.
HEIGHT_MIN_CM = 120.0
HEIGHT_MAX_CM = 230.0

# ---------------------------------------------------------------------------
# MediaPipe Pose Task
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = MODEL1_ROOT / "models" / "pose_landmarker_heavy.task"

MEDIAPIPE_MIN_DETECTION_CONF = 0.70
MEDIAPIPE_MIN_PRESENCE_CONF  = 0.70
MEDIAPIPE_MIN_TRACKING_CONF  = 0.70

# Landmark visibility threshold below which a landmark is treated as missing.
LANDMARK_VISIBILITY_THRESHOLD = 0.50

# Landmark indices in MediaPipe 33 keypoints
NOSE_IDX = 0
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

# ---------------------------------------------------------------------------
# Measurement targets (names only — formulae live in measurements/)
# ---------------------------------------------------------------------------
MEASUREMENT_NAMES = [
    "height",
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "shoulder_width",
    "inseam_length",
    "sleeve_length",
    "neck_circumference",
    "thigh_circumference",
    "bicep_circumference",
    "wrist_circumference",
    "ankle_circumference",
    "torso_length",
    "rise_length",
    "back_width",
    "arm_length",
]

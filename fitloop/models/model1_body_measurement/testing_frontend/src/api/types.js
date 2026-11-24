/**
 * api/types.js
 * ------------
 * Type definitions and contracts matching the Model 1 Stage 4B / Stage 7G FastAPI specification.
 *
 * All measurement fields match python schemas in `fitloop_measurement.measurements.schemas`.
 * Quality gate fields match `fitloop_measurement.quality.schemas`.
 */

export const CalibrationStatus = {
  VALID: 'VALID',
  INVALID_HEIGHT: 'INVALID_HEIGHT',
  INVALID_SILHOUETTE: 'INVALID_SILHOUETTE',
  CLIPPED_BODY: 'CLIPPED_BODY',
  FAILED: 'FAILED',
};

export const MeasurementStatus = {
  SUCCESS: 'SUCCESS',
  PARTIAL: 'PARTIAL',
  UNCALIBRATED: 'UNCALIBRATED',
  ANATOMICAL_LEVEL_UNAVAILABLE: 'ANATOMICAL_LEVEL_UNAVAILABLE',
  INVALID_INPUT: 'INVALID_INPUT',
  UNSUPPORTED_LEVEL: 'UNSUPPORTED_LEVEL',
};

export const AnatomicalLevel = {
  CHEST: 'chest',
  WAIST: 'waist',
  ABDOMEN: 'abdomen',
  HIPS: 'hips',
};

/**
 * Capture quality gate status (Stage 7G).
 * - PASS: all checks passed; estimates may be displayed normally.
 * - WARNING: one or more non-critical issues detected; estimates should show a visible warning.
 * - REJECT: one or more critical violations detected; estimates are diagnostic-only.
 * - UNKNOWN: quality_status field was absent or unrecognised in the response.
 */
export const QualityStatus = {
  PASS: 'PASS',
  WARNING: 'WARNING',
  REJECT: 'REJECT',
  UNKNOWN: 'UNKNOWN',
};

/** Severity of an individual quality issue. */
export const QualityIssueSeverity = {
  CRITICAL: 'CRITICAL',
  WARNING: 'WARNING',
};

/** Machine-readable quality issue codes (mirrors QualityIssueCode enum in Python). */
export const QualityIssueCode = {
  MISSING_IMAGE: 'MISSING_IMAGE',
  UNREADABLE_IMAGE: 'UNREADABLE_IMAGE',
  INVALID_IMAGE_TYPE: 'INVALID_IMAGE_TYPE',
  INVALID_HEIGHT: 'INVALID_HEIGHT',
  RESOLUTION_BELOW_MINIMUM: 'RESOLUTION_BELOW_MINIMUM',
  IMAGE_DIMENSION_MISMATCH: 'IMAGE_DIMENSION_MISMATCH',
  ASPECT_RATIO_MISMATCH: 'ASPECT_RATIO_MISMATCH',
  INSUFFICIENT_VERTICAL_MARGIN: 'INSUFFICIENT_VERTICAL_MARGIN',
  SILHOUETTE_CLIPPED: 'SILHOUETTE_CLIPPED',
  NO_PERSON_DETECTED: 'NO_PERSON_DETECTED',
  SILHOUETTE_SEGMENTATION_FAILED: 'SILHOUETTE_SEGMENTATION_FAILED',
  IMPLAUSIBLE_SILHOUETTE_HEIGHT: 'IMPLAUSIBLE_SILHOUETTE_HEIGHT',
  SILHOUETTE_HEIGHT_DISCREPANCY: 'SILHOUETTE_HEIGHT_DISCREPANCY',
  SCALE_DISCREPANCY_EXCEEDED: 'SCALE_DISCREPANCY_EXCEEDED',
  CALIBRATION_UNAVAILABLE: 'CALIBRATION_UNAVAILABLE',
  CROSS_VIEW_ALIGNMENT_UNVERIFIED: 'CROSS_VIEW_ALIGNMENT_UNVERIFIED',
  CROSS_VIEW_ALIGNMENT_REJECTED: 'CROSS_VIEW_ALIGNMENT_REJECTED',
};

/**
 * Normalise quality status from a server response, verifying agreement between top-level
 * `quality_status` and nested `quality_gate.status`.
 *
 * If either status is missing, invalid, or if they disagree with each other, the effective status
 * is normalized to UNKNOWN and a diagnostic warning message is preserved/recorded.
 *
 * @param {string|null|undefined} topLevelStatus Raw top-level quality_status
 * @param {Object|string|null|undefined} [qualityGateOrNestedStatus] Nested quality_gate object or nested status string
 * @returns {{ status: string, diagnosticMessage: string|null }} Normalized status and diagnostic info
 */
export function normaliseQualityStatus(topLevelStatus, qualityGateOrNestedStatus) {
  const validStatuses = new Set([QualityStatus.PASS, QualityStatus.WARNING, QualityStatus.REJECT]);

  const rawTop = topLevelStatus;
  const rawNested = (qualityGateOrNestedStatus && typeof qualityGateOrNestedStatus === 'object')
    ? qualityGateOrNestedStatus.status
    : qualityGateOrNestedStatus;

  // Case 1: If called with 1 argument only, behave as legacy single-status normalizer
  if (arguments.length === 1) {
    if (validStatuses.has(rawTop)) {
      return rawTop;
    }
    return QualityStatus.UNKNOWN;
  }

  // Case 2: Multi-argument call verifying agreement between top-level and nested
  const topValid = validStatuses.has(rawTop);
  const nestedValid = validStatuses.has(rawNested);

  if (!topValid && !nestedValid) {
    return {
      status: QualityStatus.UNKNOWN,
      diagnosticMessage: `Both top-level quality_status ('${rawTop}') and nested quality_gate.status ('${rawNested}') are missing or invalid.`,
    };
  }

  if (!topValid) {
    return {
      status: QualityStatus.UNKNOWN,
      diagnosticMessage: `Top-level quality_status ('${rawTop}') is missing or invalid (nested status was '${rawNested}').`,
    };
  }

  if (!nestedValid) {
    return {
      status: QualityStatus.UNKNOWN,
      diagnosticMessage: `Nested quality_gate.status ('${rawNested}') is missing or invalid (top-level was '${rawTop}').`,
    };
  }

  if (rawTop !== rawNested) {
    return {
      status: QualityStatus.UNKNOWN,
      diagnosticMessage: `Inconsistent quality status: top-level quality_status is '${rawTop}' but nested quality_gate.status is '${rawNested}'.`,
    };
  }

  return {
    status: rawTop,
    diagnosticMessage: null,
  };
}


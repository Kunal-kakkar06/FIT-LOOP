/**
 * api/client.js
 * -------------
 * Abstract API Client adapter for Model 1 circumference estimation.
 * 
 * Supports:
 * - Mock mode: returns deterministic simulated test data clearly flagged with `is_mock: true`.
 * - Real API mode: transmits multipart/form-data to FastAPI `/api/v1/measurements/estimate`.
 * 
 * Switch mode using `VITE_USE_MOCK_API="false"` in .env.local without touching UI components.
 */

import { MeasurementStatus, CalibrationStatus, QualityStatus, normaliseQualityStatus } from './types';

// Height validity bounds matching Model 1 protocol
export const HEIGHT_MIN_CM = 100.0;
export const HEIGHT_MAX_CM = 250.0;

/**
 * Validates request payload before submitting to network or mock adapter.
 * @param {Object} params
 * @param {File|null} params.frontImage
 * @param {File|null} params.sideImage
 * @param {number|string} params.heightCm
 * @returns {Array<string>} List of validation errors, empty if valid
 */
export function validateScanRequest({ frontImage, sideImage, heightCm }) {
  const errors = [];

  const h = parseFloat(heightCm);
  if (!heightCm || isNaN(h)) {
    errors.push('Standing height is required.');
  } else if (h < HEIGHT_MIN_CM || h > HEIGHT_MAX_CM) {
    errors.push(`Standing height must be between ${HEIGHT_MIN_CM} cm and ${HEIGHT_MAX_CM} cm (got ${h} cm).`);
  }

  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];

  if (!frontImage) {
    errors.push('Front-view image is required.');
  } else if (!validTypes.includes(frontImage.type)) {
    errors.push(`Front image type '${frontImage.type || 'unknown'}' is unsupported. Please upload JPG, PNG, or WebP.`);
  }

  if (!sideImage) {
    errors.push('Side-view image is required.');
  } else if (!validTypes.includes(sideImage.type)) {
    errors.push(`Side image type '${sideImage.type || 'unknown'}' is unsupported. Please upload JPG, PNG, or WebP.`);
  }

  return errors;
}

/**
 * Generates an honest simulated response for Phase 1 testing.
 * Every response is prominently tagged with `is_mock: true` and simulation notices.
 *
 * @param {number|string} heightCm
 * @param {string} [mockQualityStatus='PASS'] 'PASS' | 'WARNING' | 'REJECT' scenario selector
 */
export function generateMockResponse(heightCm, mockQualityStatus = QualityStatus.PASS) {
  const h = parseFloat(heightCm) || 175.0;

  // Approximate proportional values based on height for plausible UI rendering
  const estChest = Math.round((h * 0.54) * 10) / 10;
  const estWaist = Math.round((h * 0.44) * 10) / 10;
  const estAbdomen = Math.round((h * 0.47) * 10) / 10;
  const estHips = Math.round((h * 0.56) * 10) / 10;

  let qualityStatus = QualityStatus.PASS;
  let gateStatus = QualityStatus.PASS;
  let hasRejections = false;
  let hasWarnings = false;
  let mockIssues = [];
  let mockWarnings = [
    'Note: This is simulated mock output for frontend design verification. No actual inference performed.',
  ];

  if (mockQualityStatus === QualityStatus.WARNING) {
    qualityStatus = QualityStatus.WARNING;
    gateStatus = QualityStatus.WARNING;
    hasWarnings = true;
    mockIssues = [
      {
        code: 'SCALE_DISCREPANCY_EXCEEDED',
        severity: 'WARNING',
        message: 'Mock warning: Cross-view scale discrepancy is 8.5% (exceeds 5.0% warning threshold).',
        actionable_fix: 'Retake photos standing at exactly the same distance from camera.',
        context: { scale_discrepancy_fraction: 0.085 },
      },
    ];
    mockWarnings.push('CAPTURE QUALITY GATE WARNING: Non-critical capture discrepancies detected.');
  } else if (mockQualityStatus === QualityStatus.REJECT) {
    qualityStatus = QualityStatus.REJECT;
    gateStatus = QualityStatus.REJECT;
    hasRejections = true;
    hasWarnings = true;
    mockIssues = [
      {
        code: 'SILHOUETTE_CLIPPED',
        severity: 'CRITICAL',
        message: 'Mock reject: Subject silhouette is clipped at upper image boundary.',
        actionable_fix: 'Step back to ensure the entire body from head to feet is clearly inside the frame.',
        context: { is_clipped: true },
      },
    ];
    mockWarnings.push(
      'CAPTURE QUALITY GATE REJECT: One or more critical capture quality violations were detected. ' +
      'Returned measurements are diagnostic-only and MUST NOT be treated as quality-approved estimates.'
    );
  }

  return {
    is_mock: true,
    disclaimer: 'SIMULATED DATA: This is a simulated testing fixture. Real model inference was not executed.',
    status: MeasurementStatus.SUCCESS,
    // quality_status and quality_gate are included in the mock so that the frontend
    // shape remains compatible with real API responses (Stage 7G).
    quality_status: qualityStatus,
    quality_gate: {
      status: gateStatus,
      has_rejections: hasRejections,
      has_warnings: hasWarnings,
      scale_discrepancy_fraction: mockQualityStatus === QualityStatus.WARNING ? 0.085 : null,
      discrepancy_threshold_fraction: 0.05,
      scale_rejection_threshold_fraction: 0.15,
      min_margin_threshold_fraction: 0.05,
      issues: mockIssues,
      front_framing: null,
      side_framing: null,
      assumptions_and_disclaimers: [
        'A quality gate PASS verifies geometric and capture constraints, NOT anthropometric accuracy.',
        'Scale calibration assumes unclipped upright stature and pinhole camera perspective.',
        'Independent 2D calibration scales do not correct for pitch, roll, or 3D posture lean.',
        '[MOCK] This quality report is simulated and does not reflect real image analysis.',
      ],
    },
    timestamp: new Date().toISOString(),
    calibration: {
      known_height_cm: h,
      is_valid: true,
      front: {
        view_type: 'front',
        silhouette_height_px: 1240,
        scale_cm_per_pixel: Math.round((h / 1240) * 1000000) / 1000000,
        is_valid: true,
        status: CalibrationStatus.VALID,
        warnings: [],
      },
      side: {
        view_type: 'side',
        silhouette_height_px: 1225,
        scale_cm_per_pixel: Math.round((h / 1225) * 1000000) / 1000000,
        is_valid: true,
        status: CalibrationStatus.VALID,
        warnings: [],
      },
      assumptions: [
        'Assumes standing height represents visible vertical silhouette extent.',
        'Assumes pinhole camera model without lens distortion.',
        'Front and side scales are calculated independently and not averaged.',
      ],
      warnings: [],
    },
    measurements: {
      chest: {
        level: 'chest',
        circumference_cm: estChest,
        status: MeasurementStatus.SUCCESS,
        front_width_px: 260,
        side_depth_px: 180,
        semi_major_cm: Math.round(((260 * (h / 1240)) / 2) * 100) / 100,
        semi_minor_cm: Math.round(((180 * (h / 1225)) / 2) * 100) / 100,
        approximation_method: 'Ramanujan Second Ellipse Approximation',
        caveats: [
          'Assumes planar elliptical cross-section.',
          'Experimental geometric research baseline.',
        ],
        warnings: [],
      },
      waist: {
        level: 'waist',
        circumference_cm: estWaist,
        status: MeasurementStatus.SUCCESS,
        front_width_px: 220,
        side_depth_px: 165,
        semi_major_cm: Math.round(((220 * (h / 1240)) / 2) * 100) / 100,
        semi_minor_cm: Math.round(((165 * (h / 1225)) / 2) * 100) / 100,
        approximation_method: 'Ramanujan Second Ellipse Approximation',
        caveats: [
          'Assumes planar elliptical cross-section.',
          'Experimental geometric research baseline.',
        ],
        warnings: [],
      },
      abdomen: {
        level: 'abdomen',
        circumference_cm: estAbdomen,
        status: MeasurementStatus.SUCCESS,
        front_width_px: 235,
        side_depth_px: 175,
        semi_major_cm: Math.round(((235 * (h / 1240)) / 2) * 100) / 100,
        semi_minor_cm: Math.round(((175 * (h / 1225)) / 2) * 100) / 100,
        approximation_method: 'Ramanujan Second Ellipse Approximation',
        caveats: [
          'Assumes planar elliptical cross-section.',
          'Experimental geometric research baseline.',
        ],
        warnings: [],
      },
      hips: {
        level: 'hips',
        circumference_cm: estHips,
        status: MeasurementStatus.SUCCESS,
        front_width_px: 275,
        side_depth_px: 190,
        semi_major_cm: Math.round(((275 * (h / 1240)) / 2) * 100) / 100,
        semi_minor_cm: Math.round(((190 * (h / 1225)) / 2) * 100) / 100,
        approximation_method: 'Ramanujan Second Ellipse Approximation',
        caveats: [
          'Assumes planar elliptical cross-section.',
          'Experimental geometric research baseline.',
        ],
        warnings: [],
      },
    },
    warnings: [
      'Note: This is simulated mock output for frontend design verification. No actual inference performed.',
    ],
  };
}

/**
 * Execute measurement estimation request via Mock or Real FastAPI endpoint.
 * @param {Object} params
 * @param {File} params.frontImage
 * @param {File} params.sideImage
 * @param {number|string} params.heightCm
 * @param {boolean} [params.simulateFailure=false] Optional flag to test error state handling
 * @param {string} [params.mockQualityStatus] Optional mock scenario ('PASS' | 'WARNING' | 'REJECT'), default 'PASS'
 * @returns {Promise<Object>} API response payload
 */
export async function estimateCircumferences({
  frontImage,
  sideImage,
  heightCm,
  simulateFailure = false,
  mockQualityStatus = QualityStatus.PASS,
}) {
  // 1. Client-side input validation
  const validationErrors = validateScanRequest({ frontImage, sideImage, heightCm });
  if (validationErrors.length > 0) {
    const error = new Error('Input validation failed');
    error.validationErrors = validationErrors;
    throw error;
  }

  // 2. Check environment mode (default is mock for Phase 1)
  const isMock = import.meta.env.VITE_USE_MOCK_API !== 'false';
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '';

  if (isMock) {
    // Simulate network delay for UI state verification (loading indicators)
    await new Promise((resolve) => setTimeout(resolve, 800));

    if (simulateFailure) {
      const err = new Error('Simulated calibration failure: Silhouette height below minimum threshold.');
      err.status = 'FAILED';
      throw err;
    }

    return generateMockResponse(heightCm, mockQualityStatus);
  }

  // 3. Real HTTP FastAPI transmission (Used in Phase 2)
  const formData = new FormData();
  formData.append('front_image', frontImage);
  formData.append('side_image', sideImage);
  formData.append('standing_height_cm', String(heightCm));

  const response = await fetch(`${apiBaseUrl}/api/v1/measurements/estimate`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Backend HTTP error ${response.status}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errorDetail;
    } catch (_) {
      // Use fallback errorDetail
    }
    throw new Error(errorDetail);
  }

  const result = await response.json();
  // Validate agreement between quality_status and quality_gate.status defensively.
  // If either is missing, invalid, or inconsistent, normalises effective status to UNKNOWN
  // and preserves a useful diagnostic message.
  const norm = normaliseQualityStatus(result.quality_status, result.quality_gate);
  result.quality_status = norm.status;
  if (norm.diagnosticMessage) {
    result.quality_status_diagnostic = norm.diagnosticMessage;
    result.warnings = result.warnings || [];
    result.warnings.push(`CONTRACT NOTICE: ${norm.diagnosticMessage}`);
  }
  result.is_mock = false;
  return result;
}


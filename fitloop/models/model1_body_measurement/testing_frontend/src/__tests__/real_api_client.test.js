import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { estimateCircumferences } from '../api/client';
import { QualityStatus, normaliseQualityStatus } from '../api/types';

describe('API Client Real and Mock Pathways', () => {
  const dummyFile = (name = 'test.jpg', type = 'image/jpeg') =>
    new File(['dummy-bytes'], name, { type });

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  // ---------------------------------------------------------------------------
  // 1. Mock mode
  // ---------------------------------------------------------------------------

  it('generates simulated mock response when VITE_USE_MOCK_API is true', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'true');

    const front = dummyFile('front.jpg');
    const side = dummyFile('side.jpg');

    const result = await estimateCircumferences({
      frontImage: front,
      sideImage: side,
      heightCm: 175.0,
    });

    expect(result.is_mock).toBe(true);
    expect(result.disclaimer).toMatch(/SIMULATED DATA/);
    expect(result.measurements).toHaveProperty('chest');
    expect(result.measurements).toHaveProperty('waist');
    expect(result.measurements).toHaveProperty('abdomen');
    expect(result.measurements).toHaveProperty('hips');
    // Mock must now include quality_status and quality_gate (Stage 7G)
    expect(result.quality_status).toBe(QualityStatus.PASS);
    expect(result.quality_gate).toHaveProperty('status', QualityStatus.PASS);
    expect(result.quality_gate).toHaveProperty('issues');
    expect(Array.isArray(result.quality_gate.issues)).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 2. Real API mode — successful response
  // ---------------------------------------------------------------------------

  it('sends multipart FormData to real FastAPI endpoint when VITE_USE_MOCK_API is false', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'false');
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8000');

    const mockBackendResponse = {
      status: 'SUCCESS',
      quality_status: 'PASS',
      quality_gate: {
        status: 'PASS',
        has_rejections: false,
        has_warnings: false,
        issues: [],
        discrepancy_threshold_fraction: 0.05,
        scale_rejection_threshold_fraction: 0.15,
        assumptions_and_disclaimers: [],
      },
      calibration: {
        known_height_cm: 180.0,
        is_valid: true,
        front: { view_type: 'front', silhouette_height_px: 1200, scale_cm_per_pixel: 0.15, is_valid: true, status: 'VALID', warnings: [] },
        side: { view_type: 'side', silhouette_height_px: 1190, scale_cm_per_pixel: 0.151, is_valid: true, status: 'VALID', warnings: [] },
        assumptions: [],
        warnings: [],
      },
      measurements: {
        chest: { level: 'chest', circumference_cm: 98.2, status: 'SUCCESS', semi_major_cm: 18.5, semi_minor_cm: 13.0, warnings: [] },
        waist: { level: 'waist', circumference_cm: 81.0, status: 'SUCCESS', semi_major_cm: 15.2, semi_minor_cm: 11.0, warnings: [] },
      },
      warnings: [],
      is_mock: false,
    };

    let capturedUrl = '';
    let capturedOptions = null;

    global.fetch = vi.fn().mockImplementation((url, options) => {
      capturedUrl = url;
      capturedOptions = options;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockBackendResponse),
      });
    });

    const front = dummyFile('front.png', 'image/png');
    const side = dummyFile('side.png', 'image/png');

    const result = await estimateCircumferences({
      frontImage: front,
      sideImage: side,
      heightCm: 180.0,
    });

    expect(capturedUrl).toBe('http://127.0.0.1:8000/api/v1/measurements/estimate');
    expect(capturedOptions.method).toBe('POST');
    expect(capturedOptions.body).toBeInstanceOf(FormData);
    // Never manually specify Content-Type header so browser sets multipart boundary
    expect(capturedOptions.headers).toBeUndefined();

    // Verify response mapping
    expect(result.is_mock).toBe(false);
    expect(result.measurements.chest.circumference_cm).toBe(98.2);
    // quality_status preserved through normaliseQualityStatus
    expect(result.quality_status).toBe(QualityStatus.PASS);
  });

  // ---------------------------------------------------------------------------
  // 3. Real API mode — REJECT quality_status normalisation
  // ---------------------------------------------------------------------------

  it('normalises REJECT quality_status from real API response', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'false');

    const rejectResponse = {
      status: 'PARTIAL',
      quality_status: 'REJECT',
      quality_gate: {
        status: 'REJECT',
        has_rejections: true,
        has_warnings: true,
        issues: [
          { code: 'SILHOUETTE_CLIPPED', severity: 'CRITICAL', message: 'Clipped.', actionable_fix: 'Step back.' },
        ],
        discrepancy_threshold_fraction: 0.05,
        scale_rejection_threshold_fraction: 0.15,
        assumptions_and_disclaimers: [],
      },
      calibration: { known_height_cm: 175.0, is_valid: false, front: {}, side: {}, assumptions: [], warnings: [] },
      measurements: {},
      warnings: ['CAPTURE QUALITY GATE REJECT: ...'],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(rejectResponse),
    });

    const result = await estimateCircumferences({
      frontImage: dummyFile('f.jpg'),
      sideImage: dummyFile('s.jpg'),
      heightCm: 175.0,
    });

    expect(result.quality_status).toBe(QualityStatus.REJECT);
    // is_mock must be false even for REJECT responses
    expect(result.is_mock).toBe(false);
  });

  // ---------------------------------------------------------------------------
  // 4. Real API mode — missing quality_status becomes UNKNOWN
  // ---------------------------------------------------------------------------

  it('normalises missing quality_status to UNKNOWN — never silently promotes to PASS', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'false');

    const legacyResponse = {
      status: 'SUCCESS',
      // No quality_status field (legacy response from pre-Stage-7G server)
      calibration: { known_height_cm: 175.0, is_valid: true, front: {}, side: {}, assumptions: [], warnings: [] },
      measurements: { chest: { level: 'chest', circumference_cm: 95.0, status: 'SUCCESS', warnings: [] } },
      warnings: [],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(legacyResponse),
    });

    const result = await estimateCircumferences({
      frontImage: dummyFile('f.jpg'),
      sideImage: dummyFile('s.jpg'),
      heightCm: 175.0,
    });

    expect(result.quality_status).toBe(QualityStatus.UNKNOWN);
    expect(result.quality_status).not.toBe(QualityStatus.PASS);
  });

  // ---------------------------------------------------------------------------
  // 5. HTTP error path
  // ---------------------------------------------------------------------------

  it('propagates backend HTTP errors with server detail', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'false');

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({ detail: 'Silhouette height below minimum threshold.' }),
    });

    const front = dummyFile('front.jpg');
    const side = dummyFile('side.jpg');

    await expect(
      estimateCircumferences({
        frontImage: front,
        sideImage: side,
        heightCm: 175.0,
      })
    ).rejects.toThrow('Silhouette height below minimum threshold.');
  });

  // ---------------------------------------------------------------------------
  // 6. Network error path
  // ---------------------------------------------------------------------------

  it('handles backend network connection drop', async () => {
    vi.stubEnv('VITE_USE_MOCK_API', 'false');

    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));

    const front = dummyFile('front.jpg');
    const side = dummyFile('side.jpg');

    await expect(
      estimateCircumferences({
        frontImage: front,
        sideImage: side,
        heightCm: 175.0,
      })
    ).rejects.toThrow('Failed to fetch');
  });
});

// ---------------------------------------------------------------------------
// 7. normaliseQualityStatus unit tests (direct)
// ---------------------------------------------------------------------------

describe('normaliseQualityStatus', () => {
  it('returns PASS for "PASS" (single arg)', () => {
    expect(normaliseQualityStatus('PASS')).toBe(QualityStatus.PASS);
  });
  it('returns WARNING for "WARNING" (single arg)', () => {
    expect(normaliseQualityStatus('WARNING')).toBe(QualityStatus.WARNING);
  });
  it('returns REJECT for "REJECT" (single arg)', () => {
    expect(normaliseQualityStatus('REJECT')).toBe(QualityStatus.REJECT);
  });
  it('returns UNKNOWN for null (single arg)', () => {
    expect(normaliseQualityStatus(null)).toBe(QualityStatus.UNKNOWN);
  });
  it('returns UNKNOWN for undefined (single arg)', () => {
    expect(normaliseQualityStatus(undefined)).toBe(QualityStatus.UNKNOWN);
  });
  it('returns UNKNOWN for empty string (single arg)', () => {
    expect(normaliseQualityStatus('')).toBe(QualityStatus.UNKNOWN);
  });
  it('returns UNKNOWN for unrecognised string (single arg)', () => {
    expect(normaliseQualityStatus('GARBAGE')).toBe(QualityStatus.UNKNOWN);
  });

  // Stage 7G.3: Two-argument contract validation (top-level vs nested quality_gate)
  it('returns status with null diagnosticMessage when top and nested agree', () => {
    const res = normaliseQualityStatus('PASS', { status: 'PASS' });
    expect(res.status).toBe(QualityStatus.PASS);
    expect(res.diagnosticMessage).toBeNull();
  });

  it('normalises to UNKNOWN and provides diagnosticMessage when top and nested disagree', () => {
    const res = normaliseQualityStatus('PASS', { status: 'REJECT' });
    expect(res.status).toBe(QualityStatus.UNKNOWN);
    expect(res.diagnosticMessage).toMatch(/Inconsistent quality status/);
    expect(res.diagnosticMessage).toMatch(/top-level.*PASS.*nested.*REJECT/);
  });

  it('normalises to UNKNOWN when top-level is missing/invalid but nested is valid', () => {
    const res = normaliseQualityStatus(null, { status: 'WARNING' });
    expect(res.status).toBe(QualityStatus.UNKNOWN);
    expect(res.diagnosticMessage).toMatch(/Top-level quality_status.*missing or invalid/);
  });

  it('normalises to UNKNOWN when nested is missing/invalid but top-level is valid', () => {
    const res = normaliseQualityStatus('REJECT', null);
    expect(res.status).toBe(QualityStatus.UNKNOWN);
    expect(res.diagnosticMessage).toMatch(/Nested quality_gate\.status.*missing or invalid/);
  });

  it('normalises to UNKNOWN when both top and nested are invalid', () => {
    const res = normaliseQualityStatus('INVALID1', 'INVALID2');
    expect(res.status).toBe(QualityStatus.UNKNOWN);
    expect(res.diagnosticMessage).toMatch(/Both top-level.*and nested.*missing or invalid/);
  });
});

// ---------------------------------------------------------------------------
// 8. Deterministic Mock Scenarios (Stage 7G.3)
// ---------------------------------------------------------------------------

describe('Deterministic Mock Scenarios', () => {
  const dummyFile = (name = 'test.jpg') =>
    new File(['dummy-bytes'], name, { type: 'image/jpeg' });

  beforeEach(() => {
    vi.stubEnv('VITE_USE_MOCK_API', 'true');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('defaults to PASS mock scenario when mockQualityStatus is omitted', async () => {
    const result = await estimateCircumferences({
      frontImage: dummyFile('f.jpg'),
      sideImage: dummyFile('s.jpg'),
      heightCm: 175.0,
    });
    expect(result.is_mock).toBe(true);
    expect(result.quality_status).toBe(QualityStatus.PASS);
    expect(result.quality_gate.status).toBe(QualityStatus.PASS);
    expect(result.quality_gate.has_rejections).toBe(false);
    expect(result.quality_gate.has_warnings).toBe(false);
    expect(result.quality_gate.issues).toHaveLength(0);
  });

  it('generates WARNING mock scenario with deterministic scale issue', async () => {
    const result = await estimateCircumferences({
      frontImage: dummyFile('f.jpg'),
      sideImage: dummyFile('s.jpg'),
      heightCm: 175.0,
      mockQualityStatus: QualityStatus.WARNING,
    });
    expect(result.is_mock).toBe(true);
    expect(result.quality_status).toBe(QualityStatus.WARNING);
    expect(result.quality_gate.status).toBe(QualityStatus.WARNING);
    expect(result.quality_gate.has_warnings).toBe(true);
    expect(result.quality_gate.has_rejections).toBe(false);
    expect(result.quality_gate.issues).toHaveLength(1);
    expect(result.quality_gate.issues[0].code).toBe('SCALE_DISCREPANCY_EXCEEDED');
    expect(result.quality_gate.issues[0].severity).toBe('WARNING');
    expect(result.quality_gate.issues[0].actionable_fix).toBeTruthy();
  });

  it('generates REJECT mock scenario with deterministic critical clipping issue', async () => {
    const result = await estimateCircumferences({
      frontImage: dummyFile('f.jpg'),
      sideImage: dummyFile('s.jpg'),
      heightCm: 175.0,
      mockQualityStatus: QualityStatus.REJECT,
    });
    expect(result.is_mock).toBe(true);
    expect(result.quality_status).toBe(QualityStatus.REJECT);
    expect(result.quality_gate.status).toBe(QualityStatus.REJECT);
    expect(result.quality_gate.has_rejections).toBe(true);
    expect(result.quality_gate.issues).toHaveLength(1);
    expect(result.quality_gate.issues[0].code).toBe('SILHOUETTE_CLIPPED');
    expect(result.quality_gate.issues[0].severity).toBe('CRITICAL');
    expect(result.quality_gate.issues[0].actionable_fix).toBeTruthy();
    // Diagnostic measurements are preserved
    expect(result.measurements.chest.circumference_cm).toBeGreaterThan(0);
  });
});


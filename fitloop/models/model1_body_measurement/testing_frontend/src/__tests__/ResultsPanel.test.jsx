import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ResultsPanel, { QualityGateBanner } from '../components/ResultsPanel';
import { MeasurementStatus, QualityStatus } from '../api/types';

// ---------------------------------------------------------------------------
// Shared fixture helpers
// ---------------------------------------------------------------------------

function makeResult(overrides = {}) {
  return {
    is_mock: false,
    status: MeasurementStatus.PARTIAL,
    quality_status: QualityStatus.PASS,
    quality_gate: {
      status: QualityStatus.PASS,
      has_rejections: false,
      has_warnings: false,
      issues: [],
      discrepancy_threshold_fraction: 0.05,
      scale_rejection_threshold_fraction: 0.15,
      assumptions_and_disclaimers: [],
    },
    calibration: {
      known_height_cm: 175.0,
      front: { silhouette_height_px: 2562, scale_cm_per_pixel: 0.068306, status: 'VALID' },
      side: { silhouette_height_px: 2828, scale_cm_per_pixel: 0.061881, status: 'VALID' },
      assumptions: ['Front and side scales calculated independently.'],
    },
    measurements: {
      chest: {
        level: 'chest',
        circumference_cm: 103.2,
        status: MeasurementStatus.SUCCESS,
        semi_major_cm: 10.4,
        semi_minor_cm: 21.48,
        warnings: [],
      },
      waist: {
        level: 'waist',
        circumference_cm: 110.55,
        status: MeasurementStatus.SUCCESS,
        semi_major_cm: 11.76,
        semi_minor_cm: 22.58,
        warnings: [],
      },
      abdomen: {
        level: 'abdomen',
        circumference_cm: 111.63,
        status: MeasurementStatus.PARTIAL,
        semi_major_cm: 12.5,
        semi_minor_cm: 22.34,
        warnings: ['[Side] Scanline at y=1300 contains 2 disconnected segments (arm or garment separation).'],
      },
      hips: {
        level: 'hips',
        circumference_cm: 113.0,
        status: MeasurementStatus.SUCCESS,
        semi_major_cm: 12.56,
        semi_minor_cm: 22.68,
        warnings: [],
      },
    },
    warnings: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. QualityGateBanner — isolated unit tests
// ---------------------------------------------------------------------------

describe('QualityGateBanner', () => {
  it('renders PASS banner with accuracy disclaimer', () => {
    render(<QualityGateBanner result={{ quality_status: 'PASS', quality_gate: { issues: [] } }} />);
    expect(screen.getByTestId('quality-gate-pass-banner')).toBeInTheDocument();
    expect(screen.getByText(/capture quality checks passed/i)).toBeInTheDocument();
    expect(screen.getByText(/does not certify anthropometric accuracy/i)).toBeInTheDocument();
  });

  it('renders WARNING banner with issue list', () => {
    const result = {
      quality_status: 'WARNING',
      quality_gate: {
        status: 'WARNING',
        issues: [
          {
            code: 'SCALE_DISCREPANCY_EXCEEDED',
            severity: 'WARNING',
            message: 'Scale discrepancy is 9.1%.',
            actionable_fix: 'Retake photos from the same camera.',
          },
        ],
      },
    };
    render(<QualityGateBanner result={result} />);
    expect(screen.getByTestId('quality-gate-warning-banner')).toBeInTheDocument();
    expect(screen.getByText(/capture quality warning/i)).toBeInTheDocument();
    expect(screen.getByTestId('quality-issue-SCALE_DISCREPANCY_EXCEEDED')).toBeInTheDocument();
    expect(screen.getByText(/scale discrepancy is 9.1%/i)).toBeInTheDocument();
    expect(screen.getByText(/retake photos from the same camera/i)).toBeInTheDocument();
  });

  it('renders REJECT banner with diagnostic-only notice', () => {
    const result = {
      quality_status: 'REJECT',
      quality_gate: {
        status: 'REJECT',
        issues: [
          {
            code: 'SILHOUETTE_CLIPPED',
            severity: 'CRITICAL',
            message: 'Top of head is outside frame.',
            actionable_fix: 'Step back from the camera.',
          },
        ],
      },
    };
    render(<QualityGateBanner result={result} />);
    expect(screen.getByTestId('quality-gate-reject-banner')).toBeInTheDocument();
    expect(screen.getByText(/diagnostic values only/i)).toBeInTheDocument();
    expect(screen.getByText(/must not be used as approved measurements/i)).toBeInTheDocument();
    expect(screen.getByTestId('quality-issue-SILHOUETTE_CLIPPED')).toBeInTheDocument();
    expect(screen.getByText(/step back from the camera/i)).toBeInTheDocument();
  });

  it('renders UNKNOWN banner when quality_status is "UNKNOWN"', () => {
    render(<QualityGateBanner result={{ quality_status: 'UNKNOWN', quality_gate: { issues: [] } }} />);
    expect(screen.getByTestId('quality-gate-unknown-banner')).toBeInTheDocument();
    expect(screen.getByText(/quality status is unknown/i)).toBeInTheDocument();
  });

  it('renders UNKNOWN banner when quality_status is missing (null)', () => {
    render(<QualityGateBanner result={{ quality_status: null }} />);
    expect(screen.getByTestId('quality-gate-unknown-banner')).toBeInTheDocument();
  });

  it('renders UNKNOWN banner when quality_status is undefined', () => {
    render(<QualityGateBanner result={{}} />);
    expect(screen.getByTestId('quality-gate-unknown-banner')).toBeInTheDocument();
  });

  it('renders UNKNOWN banner when result itself is missing quality_gate', () => {
    // quality_status present but quality_gate absent — should still render, issues default to []
    render(<QualityGateBanner result={{ quality_status: 'WARNING' }} />);
    expect(screen.getByTestId('quality-gate-warning-banner')).toBeInTheDocument();
  });

  it('does NOT render PASS banner for an unrecognised quality_status string', () => {
    render(<QualityGateBanner result={{ quality_status: 'GARBAGE' }} />);
    // Should fall through to UNKNOWN
    expect(screen.getByTestId('quality-gate-unknown-banner')).toBeInTheDocument();
    expect(screen.queryByTestId('quality-gate-pass-banner')).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 2. ResultsPanel integration — quality gate within full panel
// ---------------------------------------------------------------------------

describe('ResultsPanel Diagnostic Status and Warning Display', () => {
  it('renders overall PARTIAL status indicator', () => {
    render(<ResultsPanel result={makeResult()} onBackToScan={() => {}} />);
    expect(screen.getByTestId('engine-status-badge')).toHaveTextContent('PARTIAL');
  });

  it('shows quality-gate PASS banner inside ResultsPanel', () => {
    render(<ResultsPanel result={makeResult({ quality_status: 'PASS' })} onBackToScan={() => {}} />);
    expect(screen.getByTestId('quality-gate-pass-banner')).toBeInTheDocument();
  });

  it('shows quality-gate WARNING banner inside ResultsPanel', () => {
    const result = makeResult({
      quality_status: 'WARNING',
      quality_gate: {
        status: 'WARNING',
        has_rejections: false,
        has_warnings: true,
        issues: [
          {
            code: 'SCALE_DISCREPANCY_EXCEEDED',
            severity: 'WARNING',
            message: 'Cross-view scale discrepancy is 9.1%.',
            actionable_fix: 'Retake both photos from the same position.',
          },
        ],
        discrepancy_threshold_fraction: 0.05,
        scale_rejection_threshold_fraction: 0.15,
        assumptions_and_disclaimers: [],
      },
    });
    render(<ResultsPanel result={result} onBackToScan={() => {}} />);
    expect(screen.getByTestId('quality-gate-warning-banner')).toBeInTheDocument();
    // Estimates are still shown
    expect(screen.getByTestId('measurement-card-chest')).toBeInTheDocument();
  });

  it('shows quality-gate REJECT banner and still renders diagnostic measurements', () => {
    const result = makeResult({
      quality_status: 'REJECT',
      quality_gate: {
        status: 'REJECT',
        has_rejections: true,
        has_warnings: true,
        issues: [
          {
            code: 'SILHOUETTE_CLIPPED',
            severity: 'CRITICAL',
            message: 'Silhouette extends beyond the top of the image.',
            actionable_fix: 'Increase distance from camera.',
          },
        ],
        discrepancy_threshold_fraction: 0.05,
        scale_rejection_threshold_fraction: 0.15,
        assumptions_and_disclaimers: [],
      },
    });
    render(<ResultsPanel result={result} onBackToScan={() => {}} />);
    expect(screen.getByTestId('quality-gate-reject-banner')).toBeInTheDocument();
    expect(screen.getByText(/diagnostic values only/i)).toBeInTheDocument();
    // Diagnostic measurement cards still rendered (not hidden)
    expect(screen.getByTestId('measurement-card-chest')).toBeInTheDocument();
    expect(screen.getByTestId('measurement-card-waist')).toBeInTheDocument();
    // Stage 7G.3: Cards prominently display diagnostic-only warning tags
    expect(screen.getByTestId('card-reject-tag-chest')).toBeInTheDocument();
    expect(screen.getByTestId('card-reject-tag-waist')).toBeInTheDocument();
    expect(screen.getAllByText(/diagnostic only — not approved/i).length).toBeGreaterThanOrEqual(1);
  });

  it('shows UNKNOWN banner when quality_status is absent from real API response', () => {
    const result = makeResult();
    delete result.quality_status;
    delete result.quality_gate;
    render(<ResultsPanel result={result} onBackToScan={() => {}} />);
    expect(screen.getByTestId('quality-gate-unknown-banner')).toBeInTheDocument();
    expect(screen.queryByTestId('quality-gate-pass-banner')).not.toBeInTheDocument();
  });

  it('visually distinguishes PARTIAL measurement card with warning and badge', () => {
    render(<ResultsPanel result={makeResult()} onBackToScan={() => {}} />);

    // Abdomen card should have PARTIAL badge
    const abdomenBadge = screen.getByTestId('measurement-status-abdomen');
    expect(abdomenBadge).toHaveTextContent('PARTIAL');

    // Abdomen card warning should be displayed
    expect(screen.getByTestId('card-warning-abdomen')).toHaveTextContent(
      '[Side] Scanline at y=1300 contains 2 disconnected segments'
    );

    // Caveat note for compromised slice
    expect(
      screen.getByText(/Approximation may be compromised by slice irregularities/i)
    ).toBeInTheDocument();
  });

  it('renders SUCCESS badges on clean measurements', () => {
    render(<ResultsPanel result={makeResult()} onBackToScan={() => {}} />);
    expect(screen.getByTestId('measurement-status-chest')).toHaveTextContent('SUCCESS');
    expect(screen.getByTestId('measurement-status-waist')).toHaveTextContent('SUCCESS');
    expect(screen.getByTestId('measurement-status-hips')).toHaveTextContent('SUCCESS');
  });

  it('correctly converts and displays scale in mm/px without factor-of-10 error', () => {
    // 176 cm / 1537 px = 0.114509 cm/px -> 1.1451 mm/px
    const testResult = makeResult({
      calibration: {
        known_height_cm: 176.0,
        front: {
          silhouette_height_px: 1537,
          scale_cm_per_pixel: 176.0 / 1537,
          status: 'VALID',
        },
        side: {
          silhouette_height_px: 1494,
          scale_cm_per_pixel: 176.0 / 1494,
          status: 'VALID',
        },
        assumptions: [],
      },
    });

    render(<ResultsPanel result={testResult} onBackToScan={() => {}} />);
    expect(screen.getByTestId('front-scale-display')).toHaveTextContent('1.1451 mm/px');
    expect(screen.getByTestId('side-scale-display')).toHaveTextContent('1.1780 mm/px');
  });

  it('does not show measurement values when circumference_cm is null', () => {
    const result = makeResult({
      measurements: {
        chest: {
          level: 'chest',
          circumference_cm: null,
          status: MeasurementStatus.UNCALIBRATED,
          semi_major_cm: null,
          semi_minor_cm: null,
          warnings: [],
        },
        waist: { level: 'waist', circumference_cm: 81.0, status: MeasurementStatus.SUCCESS, semi_major_cm: 10, semi_minor_cm: 8, warnings: [] },
        abdomen: { level: 'abdomen', circumference_cm: 83.0, status: MeasurementStatus.SUCCESS, semi_major_cm: 10, semi_minor_cm: 8, warnings: [] },
        hips: { level: 'hips', circumference_cm: 91.0, status: MeasurementStatus.SUCCESS, semi_major_cm: 11, semi_minor_cm: 9, warnings: [] },
      },
    });
    render(<ResultsPanel result={result} onBackToScan={() => {}} />);
    // chest card shows failed state, not a numeric value
    expect(screen.getByTestId('measurement-card-chest')).toHaveTextContent('Failed / Uncalibrated');
    // circumference is NOT present as a number
    expect(screen.queryByText('null cm')).not.toBeInTheDocument();
  });

  it('shows mock warning banner for is_mock responses', () => {
    render(<ResultsPanel result={makeResult({ is_mock: true, quality_status: 'PASS' })} onBackToScan={() => {}} />);
    expect(screen.getByTestId('mock-warning-banner')).toBeInTheDocument();
    expect(screen.getByText(/Simulated Mock Test Data/i)).toBeInTheDocument();
  });
});

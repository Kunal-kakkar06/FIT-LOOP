import React from 'react';
import { AlertTriangle, Info, CheckCircle2, XCircle, Sliders } from 'lucide-react';
import { MeasurementStatus } from '../api/types';

// Dark dusty amber color system
const c = {
  pageBg:    '#0E0C07',
  cardBg:    '#181410',
  subBg:     '#252018',
  subBorder: '#342A1C',
  border:    '#2A2218',
  amber:     '#D97706',
  amberDim:  '#92560A',
  textMain:  '#EDE8DF',
  textMuted: '#7A6A55',
  textFaint: '#3A3020',
};

export default function ResultsPanel({ result, onBackToScan }) {
  if (!result) return null;

  const { is_mock, disclaimer, calibration, measurements, warnings, status } = result;

  const anatomicalOrder = [
    { key: 'chest',    label: 'Chest Circumference',  description: 'Maximal thoracic projection' },
    { key: 'waist',    label: 'Waist Circumference',  description: 'Narrowest abdominal notch' },
    { key: 'abdomen',  label: 'Abdomen Circumference',description: 'Umbilical landmark level' },
    { key: 'hips',     label: 'Hip Circumference',    description: 'Maximal gluteal prominence' },
  ];

  return (
    <div className="space-y-6 animate-fadeIn" data-testid="results-panel">

      {/* 1. Mock / Live inference banner */}
      {is_mock ? (
        <div className="rounded-xl border p-4" data-testid="mock-warning-banner"
          style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A' }}>
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: c.amber }} />
            <div className="space-y-1">
              <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: '#D97706' }}>
                Simulated Mock Test Data — Not Real Inference
              </h4>
              <p className="text-xs leading-relaxed" style={{ color: '#A08860' }}>
                {disclaimer || 'This preview displays mock geometric outputs for UI architecture verification. No machine learning model or inference weights were executed.'}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border p-4" style={{ backgroundColor: '#0F1A10', borderColor: '#1A3A1E' }}>
          <div className="flex items-center gap-2 text-xs font-semibold" style={{ color: '#4ADE80' }}>
            <CheckCircle2 className="w-4 h-4" style={{ color: '#22C55E' }} />
            <span>Executed via Live Model 1 Endpoint</span>
          </div>
        </div>
      )}

      {/* 2. Capture Quality Gate Status Banner */}
      <QualityGateBanner result={result} />

      {/* 3. Top Summary Header */}
      <div className="rounded-xl border p-6 flex flex-col md:flex-row md:items-center justify-between gap-4"
        style={{ backgroundColor: c.cardBg, borderColor: c.border }}>
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: c.textMuted }}>
            Estimation Engine Status
          </span>
          <div className="flex items-center gap-3 mt-1">
            <h2 className="text-lg font-bold" style={{ color: c.textMain }}>
              Circumference Estimates
            </h2>
            <span
              className="text-xs font-semibold px-2.5 py-0.5 rounded-full border"
              style={
                status === MeasurementStatus.SUCCESS
                  ? { backgroundColor: '#0F2010', borderColor: '#1A4A1E', color: '#4ADE80' }
                  : { backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: c.amber }
              }
              data-testid="engine-status-badge"
            >
              {status}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onBackToScan}
          className="px-4 py-2 text-xs font-semibold rounded-lg transition-colors self-start md:self-auto"
          style={{ backgroundColor: c.subBg, color: c.textMuted, border: `1px solid ${c.border}` }}
          onMouseEnter={e => { e.currentTarget.style.backgroundColor = '#342A1C'; e.currentTarget.style.color = c.textMain; }}
          onMouseLeave={e => { e.currentTarget.style.backgroundColor = c.subBg; e.currentTarget.style.color = c.textMuted; }}
          data-testid="re-scan-button"
        >
          New Scan
        </button>
      </div>

      {/* 4. Measurement Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {anatomicalOrder.map(({ key, label, description }) => {
          const m = measurements?.[key];
          const hasVal = m && m.circumference_cm !== null && m.circumference_cm !== undefined;
          const isPartial = m?.status === MeasurementStatus.PARTIAL;
          const hasWarnings = m?.warnings && m.warnings.length > 0;

          const cardBgColor = hasVal
            ? isPartial
              ? '#1F1A0F'
              : c.cardBg
            : '#1F0F0F';
          const cardBorderColor = hasVal
            ? isPartial
              ? '#3A2E1A'
              : c.border
            : '#3A1A1A';

          return (
            <div
              key={key}
              className="rounded-xl border p-5 transition-all flex flex-col justify-between"
              style={{ backgroundColor: cardBgColor, borderColor: cardBorderColor }}
              data-testid={`measurement-card-${key}`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: c.textMuted }}>
                    {key}
                  </span>
                  {hasVal && (
                    <span
                      className="text-[10px] font-semibold px-2 py-0.5 rounded-full border"
                      style={
                        isPartial
                          ? { backgroundColor: '#2A1F0A', borderColor: '#3A2E1A', color: c.amber }
                          : { backgroundColor: '#0F2010', borderColor: '#1A4A1E', color: '#4ADE80' }
                      }
                      data-testid={`measurement-status-${key}`}
                    >
                      {m.status}
                    </span>
                  )}
                </div>
                <h3 className="text-sm font-semibold mt-1" style={{ color: c.textMain }}>{label}</h3>
                <p className="text-[11px] mt-0.5 mb-4" style={{ color: c.textMuted }}>{description}</p>

                {hasVal ? (
                  <div className="my-2">
                    {result?.quality_status === 'REJECT' && (
                      <div className="mb-2 py-1 px-2 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 border"
                        style={{ backgroundColor: '#2A0F0F', borderColor: '#4A1A1A', color: '#F87171' }}
                        data-testid={`card-reject-tag-${key}`}>
                        <AlertTriangle className="w-3 h-3 shrink-0" style={{ color: '#EF4444' }} />
                        <span>Diagnostic Only — Not Approved</span>
                      </div>
                    )}
                    <div className="flex items-baseline gap-1.5">
                      <span
                        className="text-3xl font-extrabold font-mono tracking-tight"
                        style={{
                          color: result?.quality_status === 'REJECT'
                            ? '#EF4444'
                            : c.amber,
                          textDecoration: result?.quality_status === 'REJECT' ? 'line-through' : 'none',
                        }}
                      >
                        {m.circumference_cm.toFixed(1)}
                      </span>
                      <span className="text-sm font-semibold" style={{ color: c.textMuted }}>cm</span>
                    </div>

                    {isPartial && (
                      <div className="mt-2 text-[11px] rounded-lg p-2 border flex items-start gap-1.5"
                        style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: '#B08030' }}>
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color: c.amber }} />
                        <span>Approximation may be compromised by slice irregularities.</span>
                      </div>
                    )}

                    <div className="mt-3 pt-3 border-t space-y-1 text-[11px] font-mono" style={{ borderColor: c.subBorder, color: c.textMuted }}>
                      <div className="flex justify-between">
                        <span>Front axis (2a):</span>
                        <span style={{ color: c.textMain }}>
                          {m.semi_major_cm ? (m.semi_major_cm * 2).toFixed(1) + ' cm' : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Side axis (2b):</span>
                        <span style={{ color: c.textMain }}>
                          {m.semi_minor_cm ? (m.semi_minor_cm * 2).toFixed(1) + ' cm' : '—'}
                        </span>
                      </div>
                    </div>

                    {hasWarnings && (
                      <div className="mt-2 pt-2 border-t text-[10px] space-y-1" style={{ borderColor: c.subBorder, color: '#B08030' }}>
                        {m.warnings.map((w, idx) => (
                          <p key={idx} className="leading-tight" data-testid={`card-warning-${key}`}>
                            ⚠ {w}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="my-4 py-3 px-3 rounded-lg text-xs flex items-center gap-2"
                    style={{ backgroundColor: '#2A0F0F', color: '#F87171' }}>
                    <XCircle className="w-4 h-4 shrink-0" />
                    <span>Failed / Uncalibrated</span>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-2 border-t text-[10px] flex items-center justify-between"
                style={{ borderColor: c.textFaint, color: c.textFaint }}>
                <span>Ramanujan Ellipse</span>
                <span className="capitalize">{m?.status || 'Unavailable'}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* 5. Independent Dual-View Calibration Diagnostics */}
      {calibration && (
        <div className="rounded-xl border p-6" style={{ backgroundColor: c.cardBg, borderColor: c.border }}>
          <div className="flex items-center gap-2 mb-4">
            <Sliders className="w-4 h-4" style={{ color: c.amber }} />
            <h3 className="text-sm font-bold" style={{ color: c.textMain }}>
              Independent Metric Scale Calibration
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { label: 'Front View Scale', data: calibration.front, testId: 'front-scale-display' },
              { label: 'Side View Scale',  data: calibration.side,  testId: 'side-scale-display'  },
            ].map(({ label, data, testId }) => (
              <div key={label} className="p-4 rounded-lg border" style={{ backgroundColor: c.subBg, borderColor: c.subBorder }}>
                <span className="text-xs font-semibold uppercase" style={{ color: c.amber }}>{label}</span>
                <div className="mt-2 text-xs space-y-1" style={{ color: c.textMuted }}>
                  <p>Silhouette Height: <span className="font-mono font-medium" style={{ color: c.textMain }}>{data?.silhouette_height_px || '—'} px</span></p>
                  <p>Scale Ratio: <span className="font-mono font-medium" style={{ color: c.textMain }} data-testid={testId}>{data?.scale_cm_per_pixel ? (data.scale_cm_per_pixel * 10).toFixed(4) + ' mm/px' : '—'}</span></p>
                  <p>Status: <span className="font-medium" style={{ color: '#4ADE80' }}>{data?.status || '—'}</span></p>
                </div>
              </div>
            ))}
          </div>

          {calibration.assumptions && calibration.assumptions.length > 0 && (
            <div className="mt-4 pt-4 border-t" style={{ borderColor: c.subBorder }}>
              <span className="text-[11px] font-semibold uppercase" style={{ color: c.textFaint }}>Assumptions & Caveats</span>
              <ul className="mt-1 list-disc list-inside text-xs space-y-0.5" style={{ color: c.textMuted }}>
                {calibration.assumptions.map((assump, idx) => (
                  <li key={idx}>{assump}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* 6. Pipeline Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="rounded-xl border p-4" style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A' }}>
          <div className="flex items-start gap-2.5">
            <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: c.amber }} />
            <div>
              <h4 className="text-xs font-bold" style={{ color: c.amber }}>Pipeline Quality Notices</h4>
              <ul className="mt-1 list-disc list-inside text-xs space-y-0.5" style={{ color: '#A08840' }}>
                {warnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// QualityGateBanner — separate named export for isolated testing
// ---------------------------------------------------------------------------

export function QualityGateBanner({ result }) {
  const qs = result?.quality_status;
  const gate = result?.quality_gate;
  const issues = gate?.issues ?? [];

  if (qs === 'PASS') {
    return (
      <div
        className="rounded-xl border p-3 flex items-center gap-2"
        style={{ backgroundColor: '#0F2010', borderColor: '#1A4A1E' }}
        data-testid="quality-gate-pass-banner"
      >
        <CheckCircle2 className="w-4 h-4 shrink-0" style={{ color: '#22C55E' }} />
        <p className="text-xs font-medium" style={{ color: '#4ADE80' }}>
          Capture quality checks passed.{' '}
          <span className="font-normal opacity-70">
            A PASS indicates geometric and calibration constraints are met — it does not certify anthropometric accuracy.
          </span>
        </p>
      </div>
    );
  }

  if (qs === 'WARNING') {
    return (
      <div
        className="rounded-xl border p-4"
        style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A' }}
        data-testid="quality-gate-warning-banner"
      >
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: '#D97706' }} />
          <div className="space-y-2 flex-1">
            <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: '#D97706' }}>
              Capture Quality Warning
            </h4>
            <p className="text-xs leading-relaxed" style={{ color: '#A08840' }}>
              One or more non-critical capture issues were detected. Estimates are shown but may be less reliable.
            </p>
            {issues.length > 0 && (
              <ul className="text-[11px] space-y-1 mt-1 list-disc list-inside" style={{ color: '#A08840' }}>
                {issues.map((issue, i) => (
                  <li key={i} data-testid={`quality-issue-${issue.code}`}>
                    <span className="font-semibold" style={{ color: '#D97706' }}>[{issue.code}]</span> {issue.message}
                    {issue.actionable_fix && (
                      <span className="block pl-4 italic" style={{ color: '#7A6A55' }}>Fix: {issue.actionable_fix}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (qs === 'REJECT') {
    return (
      <div
        className="rounded-xl border-2 p-4"
        style={{ backgroundColor: '#1F0F0F', borderColor: '#6B2020' }}
        data-testid="quality-gate-reject-banner"
      >
        <div className="flex items-start gap-3">
          <XCircle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: '#EF4444' }} />
          <div className="space-y-2 flex-1">
            <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: '#F87171' }}>
              ⛔ Capture Quality Gate: REJECTED — Diagnostic Values Only
            </h4>
            <p className="text-xs leading-relaxed font-medium" style={{ color: '#FCA5A5' }}>
              Critical capture quality violations were detected. Estimates below are for diagnostic inspection only and{' '}
              <strong>must not be used as approved measurements.</strong>{' '}
              Recapture using the remediation guidance below.
            </p>
            {issues.length > 0 && (
              <ul className="text-[11px] space-y-1 mt-1 list-disc list-inside" style={{ color: '#FCA5A5' }}>
                {issues.map((issue, i) => (
                  <li key={i} data-testid={`quality-issue-${issue.code}`}>
                    <span
                      className="font-semibold"
                      style={{ color: issue.severity === 'CRITICAL' ? '#F87171' : '#FCA5A5' }}
                    >
                      [{issue.severity}] [{issue.code}]
                    </span>{' '}
                    {issue.message}
                    {issue.actionable_fix && (
                      <span className="block pl-4 italic" style={{ color: '#EF4444' }}>Fix: {issue.actionable_fix}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    );
  }

  // UNKNOWN or missing
  return (
    <div
      className="rounded-xl border p-3 flex items-center gap-2"
      style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A' }}
      data-testid="quality-gate-unknown-banner"
    >
      <Info className="w-4 h-4 shrink-0" style={{ color: '#D97706' }} />
      <p className="text-xs" style={{ color: '#A08840' }}>
        Capture quality status is unknown. The server response did not include a quality gate result.
        Treat estimates with caution.
      </p>
    </div>
  );
}

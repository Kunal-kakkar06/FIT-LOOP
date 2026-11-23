import React from 'react';
import { AlertTriangle, Info, CheckCircle2, XCircle, Sliders } from 'lucide-react';
import { MeasurementStatus } from '../api/types';

export default function ResultsPanel({ result, onBackToScan }) {
  if (!result) return null;

  const { is_mock, disclaimer, calibration, measurements, warnings, status } = result;

  const anatomicalOrder = [
    { key: 'chest', label: 'Chest Circumference', description: 'Maximal thoracic projection' },
    { key: 'waist', label: 'Waist Circumference', description: 'Narrowest abdominal notch' },
    { key: 'abdomen', label: 'Abdomen Circumference', description: 'Umbilical landmark level' },
    { key: 'hips', label: 'Hip Circumference', description: 'Maximal gluteal prominence' },
  ];

  return (
    <div className="space-y-6 animate-fadeIn" data-testid="results-panel">
      {/* 1. Prominent Mock Data Disclaimer Banner */}
      {is_mock ? (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 shadow-sm" data-testid="mock-warning-banner">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-800">
                Simulated Mock Test Data — Not Real Inference
              </h4>
              <p className="text-xs text-amber-700 leading-relaxed">
                {disclaimer || 'This preview displays mock geometric outputs for UI architecture verification. No machine learning model or inference weights were executed.'}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm">
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-900">
            <CheckCircle2 className="w-4 h-4 text-blue-600" />
            <span>Executed via Live Model 1 Endpoint</span>
          </div>
        </div>
      )}

      {/* 2. Capture Quality Gate Status Banner (Stage 7G) */}
      <QualityGateBanner result={result} />

      {/* 3. Top Summary Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Estimation Engine Status
          </span>
          <div className="flex items-center gap-3 mt-1">
            <h2 className="text-lg font-bold text-slate-800">
              Circumference Estimates
            </h2>
            <span
              className={`text-xs font-semibold px-2.5 py-0.5 rounded-full ${
                status === MeasurementStatus.SUCCESS
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border border-amber-200'
              }`}
              data-testid="engine-status-badge"
            >
              {status}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onBackToScan}
          className="px-4 py-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors self-start md:self-auto"
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

          return (
            <div
              key={key}
              className={`rounded-xl border p-5 shadow-sm transition-all flex flex-col justify-between ${
                hasVal
                  ? isPartial
                    ? 'bg-amber-50/20 border-amber-300 ring-1 ring-amber-300/40'
                    : 'bg-white border-slate-200'
                  : 'bg-red-50/40 border-red-200'
              }`}
              data-testid={`measurement-card-${key}`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    {key}
                  </span>
                  {hasVal && (
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                        isPartial
                          ? 'bg-amber-100 text-amber-800 border border-amber-200'
                          : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      }`}
                      data-testid={`measurement-status-${key}`}
                    >
                      {m.status}
                    </span>
                  )}
                </div>
                <h3 className="text-sm font-semibold text-slate-800 mt-1">{label}</h3>
                <p className="text-[11px] text-slate-500 mt-0.5 mb-4">{description}</p>

                {hasVal ? (
                  <div className="my-2">
                    {result?.quality_status === 'REJECT' && (
                      <div className="mb-2 py-1 px-2 bg-red-100/90 text-red-800 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 border border-red-300" data-testid={`card-reject-tag-${key}`}>
                        <AlertTriangle className="w-3 h-3 text-red-700 shrink-0" />
                        <span>Diagnostic Only — Not Approved</span>
                      </div>
                    )}
                    <div className="flex items-baseline gap-1.5">
                      <span className={`text-3xl font-extrabold font-mono tracking-tight ${result?.quality_status === 'REJECT' ? 'text-red-900/70 line-through decoration-red-400' : 'text-slate-900'}`}>
                        {m.circumference_cm.toFixed(1)}
                      </span>
                      <span className="text-sm font-semibold text-slate-400">cm</span>
                    </div>

                    {isPartial && (
                      <div className="mt-2 text-[11px] text-amber-700 bg-amber-50 rounded-lg p-2 border border-amber-200/80 flex items-start gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-600" />
                        <span>Approximation may be compromised by slice irregularities.</span>
                      </div>
                    )}


                    <div className="mt-3 pt-3 border-t border-slate-100 space-y-1 text-[11px] text-slate-500 font-mono">
                      <div className="flex justify-between">
                        <span>Front axis (2a):</span>
                        <span className="text-slate-700 font-medium">
                          {m.semi_major_cm ? (m.semi_major_cm * 2).toFixed(1) + ' cm' : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Side axis (2b):</span>
                        <span className="text-slate-700 font-medium">
                          {m.semi_minor_cm ? (m.semi_minor_cm * 2).toFixed(1) + ' cm' : '—'}
                        </span>
                      </div>
                    </div>

                    {hasWarnings && (
                      <div className="mt-2 pt-2 border-t border-slate-100 text-[10px] text-amber-700 space-y-1">
                        {m.warnings.map((w, idx) => (
                          <p key={idx} className="leading-tight" data-testid={`card-warning-${key}`}>
                            ⚠ {w}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="my-4 py-3 px-3 bg-red-100/60 rounded-lg text-red-700 text-xs flex items-center gap-2">
                    <XCircle className="w-4 h-4 shrink-0" />
                    <span>Failed / Uncalibrated</span>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-2 text-[10px] text-slate-400 flex items-center justify-between">
                <span>Ramanujan Ellipse</span>
                <span className="capitalize">{m?.status || 'Unavailable'}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* 5. Independent Dual-View Calibration Diagnostics */}
      {calibration && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Sliders className="w-4 h-4 text-blue-600" />
            <h3 className="text-sm font-bold text-slate-800">
              Independent Metric Scale Calibration
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
              <span className="text-xs font-semibold text-blue-700 uppercase">Front View Scale</span>
              <div className="mt-2 text-xs space-y-1 text-slate-600">
                <p>Silhouette Height: <span className="font-mono font-medium text-slate-800">{calibration.front?.silhouette_height_px || '—'} px</span></p>
                <p>Scale Ratio: <span className="font-mono font-medium text-slate-800" data-testid="front-scale-display">{calibration.front?.scale_cm_per_pixel ? (calibration.front.scale_cm_per_pixel * 10).toFixed(4) + ' mm/px' : '—'}</span></p>
                <p>Status: <span className="font-medium text-emerald-600">{calibration.front?.status || '—'}</span></p>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-100">
              <span className="text-xs font-semibold text-blue-700 uppercase">Side View Scale</span>
              <div className="mt-2 text-xs space-y-1 text-slate-600">
                <p>Silhouette Height: <span className="font-mono font-medium text-slate-800">{calibration.side?.silhouette_height_px || '—'} px</span></p>
                <p>Scale Ratio: <span className="font-mono font-medium text-slate-800" data-testid="side-scale-display">{calibration.side?.scale_cm_per_pixel ? (calibration.side.scale_cm_per_pixel * 10).toFixed(4) + ' mm/px' : '—'}</span></p>
                <p>Status: <span className="font-medium text-emerald-600">{calibration.side?.status || '—'}</span></p>
              </div>
            </div>
          </div>

          {calibration.assumptions && calibration.assumptions.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <span className="text-[11px] font-semibold uppercase text-slate-400">Assumptions & Caveats</span>
              <ul className="mt-1 list-disc list-inside text-xs text-slate-500 space-y-0.5">
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
        <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4 shadow-sm">
          <div className="flex items-start gap-2.5">
            <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-xs font-bold text-amber-800">Pipeline Quality Notices</h4>
              <ul className="mt-1 list-disc list-inside text-xs text-amber-700 space-y-0.5">
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

/**
 * Displays the Stage 7G capture quality gate status.
 *
 * Behaviour by quality_status:
 *   PASS    → green informational note (checks passed; accuracy NOT guaranteed)
 *   WARNING → amber banner listing capture issues with remediation
 *   REJECT  → red prominent banner; estimates flagged as diagnostic-only
 *   UNKNOWN → amber notice; missing status is never silently treated as PASS
 *   null/undefined → same as UNKNOWN
 */
export function QualityGateBanner({ result }) {
  const qs = result?.quality_status;
  const gate = result?.quality_gate;
  const issues = gate?.issues ?? [];

  if (qs === 'PASS') {
    return (
      <div
        className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 shadow-sm flex items-center gap-2"
        data-testid="quality-gate-pass-banner"
      >
        <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
        <p className="text-xs text-emerald-800 font-medium">
          Capture quality checks passed.{' '}
          <span className="font-normal opacity-80">
            A PASS indicates geometric and calibration constraints are met — it does not certify anthropometric accuracy.
          </span>
        </p>
      </div>
    );
  }

  if (qs === 'WARNING') {
    return (
      <div
        className="rounded-xl border border-amber-300 bg-amber-50 p-4 shadow-sm"
        data-testid="quality-gate-warning-banner"
      >
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-800">
              Capture Quality Warning
            </h4>
            <p className="text-xs text-amber-700 leading-relaxed">
              One or more non-critical capture issues were detected. Estimates are shown but may be less reliable.
            </p>
            {issues.length > 0 && (
              <ul className="text-[11px] text-amber-700 space-y-1 mt-1 list-disc list-inside">
                {issues.map((issue, i) => (
                  <li key={i} data-testid={`quality-issue-${issue.code}`}>
                    <span className="font-semibold">[{issue.code}]</span> {issue.message}
                    {issue.actionable_fix && (
                      <span className="block pl-4 text-amber-600 italic">Fix: {issue.actionable_fix}</span>
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
        className="rounded-xl border-2 border-red-400 bg-red-50 p-4 shadow-sm"
        data-testid="quality-gate-reject-banner"
      >
        <div className="flex items-start gap-3">
          <XCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <h4 className="text-xs font-bold uppercase tracking-wider text-red-800">
              ⛔ Capture Quality Gate: REJECTED — Diagnostic Values Only
            </h4>
            <p className="text-xs text-red-700 leading-relaxed font-medium">
              Critical capture quality violations were detected. Estimates below are for diagnostic inspection only and{' '}
              <strong>must not be used as approved measurements.</strong>{' '}
              Recapture using the remediation guidance below.
            </p>
            {issues.length > 0 && (
              <ul className="text-[11px] text-red-700 space-y-1 mt-1 list-disc list-inside">
                {issues.map((issue, i) => (
                  <li key={i} data-testid={`quality-issue-${issue.code}`}>
                    <span className={`font-semibold ${issue.severity === 'CRITICAL' ? 'text-red-800' : 'text-red-600'}`}>
                      [{issue.severity}] [{issue.code}]
                    </span>{' '}
                    {issue.message}
                    {issue.actionable_fix && (
                      <span className="block pl-4 text-red-600 italic">Fix: {issue.actionable_fix}</span>
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

  // UNKNOWN or missing quality_status — never silently treat as PASS
  return (
    <div
      className="rounded-xl border border-amber-200 bg-amber-50/60 p-3 shadow-sm flex items-center gap-2"
      data-testid="quality-gate-unknown-banner"
    >
      <Info className="w-4 h-4 text-amber-500 shrink-0" />
      <p className="text-xs text-amber-800">
        Capture quality status is unknown. The server response did not include a quality gate result.
        Treat estimates with caution.
      </p>
    </div>
  );
}

import React from 'react';
import { Play, Loader2, RefreshCw } from 'lucide-react';

export default function ActionControls({
  onScan,
  onReset,
  isLoading,
  canScan,
  isMockMode,
  simulateFailure,
  onToggleSimulateFailure,
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
      {/* Simulation Controls for testing */}
      <div className="flex items-center gap-3 w-full sm:w-auto">
        <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-600 select-none">
          <input
            type="checkbox"
            checked={simulateFailure}
            onChange={(e) => onToggleSimulateFailure(e.target.checked)}
            className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 w-4 h-4 cursor-pointer"
            data-testid="simulate-failure-checkbox"
          />
          <span>Simulate Calibration Failure (Test Error State)</span>
        </label>
      </div>

      {/* Primary Actions */}
      <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
        <button
          type="button"
          onClick={onReset}
          disabled={isLoading}
          className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 rounded-lg transition-colors flex items-center gap-1.5"
          data-testid="reset-button"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Clear All
        </button>

        <button
          type="button"
          onClick={onScan}
          disabled={!canScan || isLoading}
          className={`px-6 py-2.5 text-xs font-semibold rounded-lg shadow-sm flex items-center gap-2 transition-all ${
            isLoading
              ? 'bg-blue-400 text-white cursor-not-allowed'
              : canScan
              ? 'bg-slate-900 hover:bg-blue-600 text-white cursor-pointer active:scale-95'
              : 'bg-slate-200 text-slate-400 cursor-not-allowed'
          }`}
          data-testid="scan-button"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing Geometric Slices...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>{isMockMode ? 'Run Simulated Scan' : 'Run Model 1 Scan'}</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

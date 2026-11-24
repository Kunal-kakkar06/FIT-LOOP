import React from 'react';
import { Play, Loader2, RefreshCw } from 'lucide-react';

export default function ActionControls({ onScan, onReset, isLoading, canScan, isMockMode, simulateFailure, onToggleSimulateFailure }) {
  return (
    <div className="rounded-xl border p-5 flex flex-col sm:flex-row items-center justify-between gap-4"
      style={{ backgroundColor: '#252018', borderColor: '#342A1C' }}>
      {/* Debug toggle */}
      <label className="flex items-center gap-2 cursor-pointer select-none text-xs" style={{ color: '#5A4D3A' }}>
        <input type="checkbox" checked={simulateFailure} onChange={(e) => onToggleSimulateFailure(e.target.checked)}
          data-testid="simulate-failure-checkbox"
          className="w-4 h-4 cursor-pointer rounded"
          style={{ accentColor: '#D97706' }}
        />
        <span>Simulate Calibration Failure</span>
      </label>

      {/* Actions */}
      <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
        <button type="button" onClick={onReset} disabled={isLoading}
          data-testid="reset-button"
          className="px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-all disabled:opacity-40"
          style={{ backgroundColor: '#201B12', border: '1px solid #2A2218', color: '#7A6A55' }}
          onMouseEnter={(e) => e.currentTarget.style.color = '#EDE8DF'}
          onMouseLeave={(e) => e.currentTarget.style.color = '#7A6A55'}>
          <RefreshCw className="w-3.5 h-3.5" /> Clear All
        </button>

        <button type="button" onClick={onScan} disabled={!canScan || isLoading}
          data-testid="scan-button"
          className="px-6 py-2.5 text-xs font-bold rounded-lg flex items-center gap-2 transition-all"
          style={isLoading
            ? { backgroundColor: '#78350F', color: '#FDE68A', cursor: 'not-allowed' }
            : canScan
            ? { backgroundColor: '#D97706', color: '#0E0C07', cursor: 'pointer' }
            : { backgroundColor: '#1F1A0F', color: '#3A3020', cursor: 'not-allowed' }}>
          {isLoading
            ? <><Loader2 className="w-4 h-4 animate-spin" />Processing…</>
            : <><Play className="w-4 h-4 fill-current" />{isMockMode ? 'Run Simulated Scan' : 'Run Body Scan'}</>}
        </button>
      </div>
    </div>
  );
}

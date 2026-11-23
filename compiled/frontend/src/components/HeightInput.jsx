import React from 'react';
import { HelpCircle, AlertCircle } from 'lucide-react';
import { HEIGHT_MIN_CM, HEIGHT_MAX_CM } from '../api/client';

export default function HeightInput({ value, onChange, error }) {
  const numVal = parseFloat(value);
  const isValid = !isNaN(numVal) && numVal >= HEIGHT_MIN_CM && numVal <= HEIGHT_MAX_CM;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <label htmlFor="height-input" className="text-sm font-semibold text-slate-800">
          Standing Height (Reference Stature)
        </label>
        <div className="group relative cursor-pointer">
          <HelpCircle className="w-4 h-4 text-slate-400 hover:text-slate-600" />
          <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block w-64 p-2 bg-slate-800 text-white text-xs rounded-lg shadow-lg z-10">
            Standing height calibrates pixel-to-metric scale independently for front and side views ($H_{'{'}true{'}'} / h_{'{'}silhouette{'}'}$).
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500 mb-4">
        Input the participant's exact barefoot height in centimetres. Valid range: {HEIGHT_MIN_CM}–{HEIGHT_MAX_CM} cm.
      </p>

      <div className="relative max-w-xs">
        <input
          id="height-input"
          data-testid="height-input"
          type="number"
          step="0.1"
          placeholder="e.g. 175.5"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`w-full px-3.5 py-2.5 rounded-lg border text-sm font-mono focus:outline-none focus:ring-2 transition-all ${
            error
              ? 'border-red-400 ring-1 ring-red-400 focus:ring-red-400'
              : 'border-slate-300 focus:border-blue-500 focus:ring-blue-100'
          }`}
        />
        <span className="absolute right-3.5 top-2.5 text-xs font-semibold text-slate-400 pointer-events-none">
          cm
        </span>
      </div>

      {error ? (
        <div className="flex items-center gap-1.5 mt-2.5 text-xs text-red-600">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : value && isValid ? (
        <p className="text-xs text-emerald-600 font-medium mt-2">
          ✓ Valid human stature calibration baseline.
        </p>
      ) : null}
    </div>
  );
}

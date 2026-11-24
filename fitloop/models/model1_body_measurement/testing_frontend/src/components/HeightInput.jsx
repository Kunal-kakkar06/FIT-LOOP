import React, { useState } from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { HEIGHT_MIN_CM, HEIGHT_MAX_CM } from '../api/client';

function cmToFtIn(cm) {
  const totalInches = cm / 2.54;
  const ft = Math.floor(totalInches / 12);
  const inches = Math.round(totalInches % 12);
  return { ft, inches };
}
function ftInToCm(ft, inches) { return ((ft * 12) + inches) * 2.54; }

const inputStyle = {
  backgroundColor: '#0E0C07',
  border: '1px solid #2A2218',
  color: '#EDE8DF',
  borderRadius: '0.75rem',
};

export default function HeightInput({ value, onChange, error }) {
  const [unit, setUnit] = useState('cm');
  const numVal = parseFloat(value);
  const isValid = !isNaN(numVal) && numVal >= HEIGHT_MIN_CM && numVal <= HEIGHT_MAX_CM;
  const { ft, inches } = cmToFtIn(isNaN(numVal) ? 175 : numVal);

  const handleFtChange  = (ftNew)  => onChange(ftInToCm(parseFloat(ftNew)  || 0, inches).toFixed(1));
  const handleInChange  = (inNew)  => onChange(ftInToCm(ft, parseFloat(inNew)  || 0).toFixed(1));

  return (
    <div className="rounded-2xl border p-5" style={{ backgroundColor: '#252018', borderColor: '#342A1C' }}>
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm font-semibold" style={{ color: '#F5F0E8' }}>Standing Height</label>
        {/* Toggle */}
        <div className="flex items-center rounded-lg p-0.5 text-xs font-semibold gap-0.5" style={{ backgroundColor: '#0E0C07' }}>
          {['cm', 'ft/in'].map((u) => (
            <button key={u} onClick={() => setUnit(u)}
              className="px-3 py-1 rounded-md transition-all"
              style={unit === u
                ? { backgroundColor: '#D97706', color: '#0E0C07' }
                : { color: '#7A6A55', backgroundColor: 'transparent' }}>
              {u}
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs mb-4" style={{ color: '#5A4D3A' }}>
        Enter your exact barefoot height. Valid range: {HEIGHT_MIN_CM}–{HEIGHT_MAX_CM} cm.
      </p>

      {unit === 'cm' ? (
        <div className="relative max-w-xs">
          <input id="height-input" data-testid="height-input" type="number" step="0.1" placeholder="175.0"
            value={value} onChange={(e) => onChange(e.target.value)}
            className="w-full px-3.5 py-2.5 pr-12 text-sm font-mono focus:outline-none transition-all"
            style={{ ...inputStyle, borderColor: error ? '#7F1D1D' : '#2A2218', color: '#D97706' }}
            onFocus={(e) => e.target.style.borderColor = '#D97706'}
            onBlur={(e)  => e.target.style.borderColor = error ? '#7F1D1D' : '#2A2218'}
          />
          <span className="absolute right-3.5 top-2.5 text-xs font-semibold pointer-events-none" style={{ color: '#5A4D3A' }}>cm</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 max-w-xs">
          <div className="relative flex-1">
            <input type="number" min="3" max="8" step="1" value={ft}
              onChange={(e) => handleFtChange(e.target.value)}
              className="w-full px-3.5 py-2.5 pr-8 text-sm font-mono focus:outline-none transition-all"
              style={inputStyle}
              onFocus={(e) => e.target.style.borderColor = '#D97706'}
              onBlur={(e)  => e.target.style.borderColor = '#2A2218'}
            />
            <span className="absolute right-3 top-2.5 text-xs font-semibold pointer-events-none" style={{ color: '#5A4D3A' }}>ft</span>
          </div>
          <div className="relative flex-1">
            <input type="number" min="0" max="11" step="1" value={inches}
              onChange={(e) => handleInChange(e.target.value)}
              className="w-full px-3.5 py-2.5 pr-8 text-sm font-mono focus:outline-none transition-all"
              style={inputStyle}
              onFocus={(e) => e.target.style.borderColor = '#D97706'}
              onBlur={(e)  => e.target.style.borderColor = '#2A2218'}
            />
            <span className="absolute right-3 top-2.5 text-xs font-semibold pointer-events-none" style={{ color: '#5A4D3A' }}>in</span>
          </div>
          <span className="text-xs whitespace-nowrap font-mono" style={{ color: '#7A6A55' }}>= {numVal.toFixed(1)} cm</span>
        </div>
      )}

      {error ? (
        <div className="flex items-center gap-1.5 mt-2.5 text-xs" style={{ color: '#F87171' }}>
          <AlertCircle className="w-3.5 h-3.5 shrink-0" /><span>{error}</span>
        </div>
      ) : value && isValid ? (
        <div className="flex items-center gap-1.5 mt-2.5 text-xs font-medium" style={{ color: '#6EE7B7' }}>
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
          Valid baseline — {numVal.toFixed(1)} cm ({ft} ft {inches} in)
        </div>
      ) : null}
    </div>
  );
}

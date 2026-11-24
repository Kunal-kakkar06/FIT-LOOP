import React, { useRef, useState } from 'react';
import { Upload, X, RefreshCw, AlertCircle, Eye } from 'lucide-react';

const C = {
  bg:       '#181410',
  bg2:      '#0E0C07',
  border:   '#2A2218',
  amber:    '#D97706',
  amberDim: '#92400E',
  text:     '#EDE8DF',
  muted:    '#7A6A55',
  faint:    '#3A3020',
};

/* SVG silhouettes – visible warm gray-brown */
const FrontSilhouette = () => (
  <svg viewBox="0 0 80 180" className="absolute inset-0 w-full h-full pointer-events-none" fill="#7A6248" opacity="0.32">
    <ellipse cx="40" cy="20" rx="12" ry="14" />
    <rect x="35" y="33" width="10" height="8" rx="3" />
    <path d="M20 41 Q14 70 16 100 L64 100 Q66 70 60 41 Q50 46 40 46 Q30 46 20 41Z" />
    <path d="M20 44 Q6 60 8 95 L14 95 Q12 65 24 50Z" />
    <path d="M60 44 Q74 60 72 95 L66 95 Q68 65 56 50Z" />
    <path d="M16 100 L24 155 L36 155 L40 115 L44 155 L56 155 L64 100Z" />
    <ellipse cx="27" cy="160" rx="8" ry="4" />
    <ellipse cx="53" cy="160" rx="8" ry="4" />
  </svg>
);
const SideSilhouette = () => (
  <svg viewBox="0 0 80 180" className="absolute inset-0 w-full h-full pointer-events-none" fill="#7A6248" opacity="0.32">
    <ellipse cx="42" cy="20" rx="11" ry="13" />
    <rect x="38" y="32" width="8" height="8" rx="3" />
    <path d="M32 40 Q28 70 30 100 L52 100 Q56 70 50 40Z" />
    <path d="M50 44 Q60 65 58 95 L52 95 Q54 68 46 50Z" />
    <path d="M30 100 L35 155 L46 155 L50 100Z" />
    <ellipse cx="40" cy="160" rx="12" ry="4" />
  </svg>
);

function ExampleModal({ viewType, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(14,12,7,0.85)' }} onClick={onClose}>
      <div className="rounded-2xl border p-6 max-w-xs w-full mx-4" style={{ backgroundColor: '#201B12', borderColor: '#2A2218' }} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-bold capitalize" style={{ color: '#F5F0E8' }}>{viewType} Capture — Example</h4>
          <button onClick={onClose} className="p-1 rounded-lg transition-colors" style={{ color: '#7A6A55' }}
            onMouseEnter={(e) => e.currentTarget.style.color = '#EDE8DF'} onMouseLeave={(e) => e.currentTarget.style.color = '#7A6A55'}>
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="rounded-xl h-56 flex items-center justify-center relative overflow-hidden border" style={{ backgroundColor: '#0E0C07', borderColor: '#2A2218' }}>
          {viewType === 'front' ? <FrontSilhouette /> : <SideSilhouette />}
          <div className="relative z-10 text-center px-4">
            <p className="text-xs font-semibold mb-1" style={{ color: '#D97706' }}>{viewType === 'front' ? 'A‑Pose (Front View)' : '90° Side Profile'}</p>
            <p className="text-[11px]" style={{ color: '#7A6A55' }}>
              {viewType === 'front'
                ? 'Arms slightly away from torso, feet together, facing camera.'
                : 'Turned 90° left or right, arms slightly back, neutral posture.'}
            </p>
          </div>
        </div>
        <ul className="mt-4 space-y-1.5 text-[11px]" style={{ color: '#7A6A55' }}>
          {['Full head‑to‑toe in frame, no cropping', 'Plain background preferred', 'Bare feet or tight socks', 'Good even lighting'].map((tip, i) => (
            <li key={i} className="flex items-start gap-2"><span style={{ color: '#D97706' }} className="font-bold mt-0.5">•</span>{tip}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function ImageUploadCard({ viewType, title, description, file, previewUrl, onFileSelect, onFileRemove, error }) {
  const fileInputRef = useRef(null);
  const [showExample, setShowExample] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    if (e.dataTransfer.files?.[0]) onFileSelect(e.dataTransfer.files[0]);
  };

  const isAmber = viewType === 'front';

  return (
    <>
      {showExample && <ExampleModal viewType={viewType} onClose={() => setShowExample(false)} />}
      <div className="relative flex flex-col rounded-2xl border transition-all"
        style={{ backgroundColor: '#181410', borderColor: error ? '#7F1D1D' : '#2A2218' }}>

        {/* Header */}
        <div className="flex items-start justify-between px-5 pt-5 pb-3 gap-3">
          <div>
            <span className="inline-block text-[10px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full mb-1.5 border"
              style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: '#D97706' }}>
              {viewType} Perspective
            </span>
            <h3 className="text-sm font-semibold" style={{ color: '#F5F0E8' }}>{title}</h3>
            <p className="text-xs mt-0.5 leading-relaxed max-w-[260px]" style={{ color: '#7A6A55' }}>{description}</p>
          </div>
          <button onClick={() => setShowExample(true)}
            className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold rounded-lg transition-all"
            style={{ backgroundColor: '#D97706', color: '#0E0C07' }}>
            <Eye className="w-3 h-3" /> Example
          </button>
        </div>

        <div className="px-5 pb-5">
          <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp"
            onChange={(e) => { if (e.target.files?.[0]) onFileSelect(e.target.files[0]); }}
            className="hidden" data-testid={`file-input-${viewType}`}
          />

          {previewUrl ? (
            <div className="relative group w-full h-64 rounded-xl overflow-hidden flex items-center justify-center border"
              style={{ backgroundColor: '#0E0C07', borderColor: '#2A2218' }}>
              <img src={previewUrl} alt={`${viewType} preview`} className="w-full h-full object-contain" />
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3"
                style={{ backgroundColor: 'rgba(14,12,7,0.75)' }}>
                <button onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                  style={{ backgroundColor: '#201B12', border: '1px solid #3A2E1A', color: '#EDE8DF' }}>
                  <RefreshCw className="w-3.5 h-3.5" /> Replace
                </button>
                <button onClick={onFileRemove}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                  style={{ backgroundColor: '#7F1D1D', color: '#FEE2E2' }}>
                  <X className="w-3.5 h-3.5" /> Remove
                </button>
              </div>
            </div>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              data-testid={`dropzone-${viewType}`}
              className="relative w-full h-64 rounded-xl cursor-pointer flex flex-col items-center justify-center p-6 text-center transition-all overflow-hidden border-2 border-dashed"
              style={{
                backgroundColor: dragOver ? 'rgba(217,119,6,0.06)' : '#0E0C07',
                borderColor: dragOver ? '#D97706' : '#2A2218',
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = '#3A2E1A'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = dragOver ? '#D97706' : '#2A2218'}
            >
              {viewType === 'front' ? <FrontSilhouette /> : <SideSilhouette />}
              <div className="relative z-10 flex flex-col items-center">
                <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3 shadow-lg"
                  style={{ backgroundColor: '#D97706' }}>
                  <Upload className="w-5 h-5" style={{ color: '#0E0C07' }} />
                </div>
                <p className="text-sm font-semibold" style={{ color: '#EDE8DF' }}>Click to upload or drag & drop</p>
                <p className="text-xs mt-1" style={{ color: '#5A4D3A' }}>
                  ({viewType === 'front' ? 'Front‑facing, arms slightly away' : 'Side‑profile, natural posture'})
                </p>
                <p className="text-[11px] mt-1" style={{ color: '#3A3020' }}>JPG, PNG, or WebP — head‑to‑toe</p>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-1.5 mt-3 text-xs" style={{ color: '#F87171' }}>
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" /><span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

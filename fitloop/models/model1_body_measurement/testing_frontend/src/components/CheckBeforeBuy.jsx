import React, { useState, useRef } from 'react';
import { ShoppingBag, Upload, AlertTriangle, CheckCircle2, XCircle, RefreshCw, Sparkles, Shirt, X } from 'lucide-react';
import { runCheckBeforeBuy } from '../api/wardrobeClient';

const pageBg   = '#0E0C07';
const cardBg   = '#181410';
const card2Bg  = '#201B12';
const subBg    = '#252018';    // subsection panels — warm sand-brown
const insetBg  = '#1C1810';   // deep inset (dropzone)
const border   = '#2A2218';
const subBorder= '#342A1C';   // subsection border — slightly warmer
const amber    = '#D97706';
const textMain = '#EDE8DF';
const textMuted= '#7A6A55';
const textFaint= '#3A3020';

const SAMPLE_CHIPS = [
  { label: '🫒 Olive Bomber Jacket', url: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&q=80' },
  { label: '🤍 White Linen Shirt',   url: 'https://images.unsplash.com/photo-1620012253295-c15cc3e65df4?w=400&q=80' },
  { label: '🖤 Black Hoodie',        url: 'https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=400&q=80' },
];

function ProgressBar({ label, value, color }) {
  const bgMap = { amber: '#D97706', red: '#DC2626', green: '#10B981', blue: '#6366F1' };
  const bg = bgMap[color] || bgMap.amber;
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs" style={{ color: textMuted }}>{label}</span>
        <span className="text-xs font-bold font-mono" style={{ color: textMain }}>{Math.round(value)}%</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: subBg }}>
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(100, Math.max(0, value))}%`, backgroundColor: bg }} />
      </div>
    </div>
  );
}

function getVerdictConfig(verdict, score) {
  if (verdict === 'DUPLICATE') return {
    bg: '#1F1210', borderColor: '#4A2020', textColor: '#FCA5A5',
    badge: { bg: '#DC2626', text: '#FEF2F2' },
    riskBadge: { label: 'High Duplicate Risk', bg: '#2A1818', text: '#F87171' },
    icon: <XCircle className="w-7 h-7 shrink-0" style={{ color: '#F87171' }} />,
    title: 'Duplicate Detected — Avoid Purchase',
    sub: `Match Confidence: ${(score*100).toFixed(1)}% (threshold ≥ 85%)`,
    barColor: 'red',
  };
  if (verdict === 'SIMILAR') return {
    bg: '#1F1A0F', borderColor: '#3A2E1A', textColor: '#FCD34D',
    badge: { bg: '#D97706', text: '#0E0C07' },
    riskBadge: { label: 'Moderate Duplicate Risk', bg: '#261E0F', text: '#FCD34D' },
    icon: <AlertTriangle className="w-7 h-7 shrink-0" style={{ color: '#FCD34D' }} />,
    title: 'Similar Item in Your Closet',
    sub: `Similarity: ${(score*100).toFixed(1)}% (range 65%–84%)`,
    barColor: 'amber',
  };
  return {
    bg: '#0F1F12', borderColor: '#1A3A20', textColor: '#6EE7B7',
    badge: { bg: '#10B981', text: '#0E0C07' },
    riskBadge: { label: 'Low Duplicate Risk', bg: '#0F1F12', text: '#6EE7B7' },
    icon: <CheckCircle2 className="w-7 h-7 shrink-0" style={{ color: '#6EE7B7' }} />,
    title: 'Unique Item — Safe to Buy',
    sub: `Novelty: ${((1-score)*100).toFixed(1)}% unique`,
    barColor: 'green',
  };
}

export default function CheckBeforeBuy({ userId }) {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [showPipeline, setShowPipeline] = useState(false);
  const fileRef = useRef(null);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setPreview(URL.createObjectURL(f)); setError(null); setResult(null); }
  };

  const handleChip = async (chip) => {
    setError(null); setResult(null);
    try {
      const resp = await fetch(chip.url);
      const blob = await resp.blob();
      const f = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
      setFile(f); setPreview(chip.url);
    } catch { setError('Could not load sample. Please upload manually.'); }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError('Please upload a photo.'); return; }
    setIsEvaluating(true); setError(null);
    try { const res = await runCheckBeforeBuy(file, userId); setResult(res); }
    catch (err) { setError(err.message || 'Evaluation failed'); }
    finally { setIsEvaluating(false); }
  };

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border p-6" style={{ backgroundColor: cardBg, borderColor: border }}>
        {/* Header */}
        <div className="flex items-start justify-between mb-5">
          <div>
            <h2 className="text-lg font-bold flex items-center gap-2" style={{ color: textMain }}>
              <ShoppingBag className="w-5 h-5" style={{ color: amber }} />
              Check‑Before‑Buy
            </h2>
            <p className="text-xs mt-1 max-w-md" style={{ color: textMuted }}>
              Upload a photo of an item you're considering — the AI cross‑checks it against your entire wardrobe.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setShowPipeline(!showPipeline)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold rounded-lg border transition-all"
              style={{ backgroundColor: card2Bg, borderColor: border, color: textMuted }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = amber; e.currentTarget.style.color = amber; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = border; e.currentTarget.style.color = textMuted; }}>
              Pipeline Specs
            </button>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold border"
              style={{ backgroundColor: '#1F1210', borderColor: '#3A1818', color: '#F87171' }}>
              Duplicate AI
            </span>
          </div>
        </div>

        {showPipeline && (
          <div className="mb-5 p-4 rounded-xl border text-xs space-y-1.5" style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
            <p className="font-semibold mb-2" style={{ color: textMain }}>🔬 Comparison Engine</p>
            <p><span className="font-semibold" style={{ color: amber }}>1. Category:</span> YOLO26n determines garment type compatibility</p>
            <p><span className="font-semibold" style={{ color: amber }}>2. Color:</span> CIE LAB Delta‑E perceptual distance on core 60% crop</p>
            <p><span className="font-semibold" style={{ color: amber }}>3. Texture:</span> MobileNetV3 576‑dim cosine similarity for pattern matching</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Upload box */}
            <div>
              <div className="border-2 border-dashed rounded-xl p-5 text-center flex flex-col justify-center items-center min-h-[200px] transition-colors"
                style={{ backgroundColor: insetBg, borderColor: preview ? border : '#3A2E1A' }}>
                {preview ? (
                  <div className="relative w-full h-48 flex items-center justify-center">
                    <img src={preview} alt="Prospective Item" className="max-h-full max-w-full rounded-xl object-contain" />
                    <button type="button" onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                      className="absolute top-2 right-2 flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded-full border"
                      style={{ backgroundColor: card2Bg, borderColor: border, color: textMuted }}>
                      <X className="w-3 h-3" /> Change
                    </button>
                  </div>
                ) : (
                  <label className="cursor-pointer flex flex-col items-center w-full" onClick={() => fileRef.current?.click()}>
                    <input ref={fileRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                    <div className="w-12 h-12 rounded-2xl border flex items-center justify-center mb-3"
                      style={{ backgroundColor: card2Bg, borderColor: '#3A2E1A' }}>
                      <Upload className="w-5 h-5" style={{ color: amber }} />
                    </div>
                    <span className="text-sm font-semibold" style={{ color: textMain }}>Upload Prospective Purchase</span>
                    <span className="text-xs mt-1" style={{ color: textMuted }}>Tag photo, hanger, or screenshot</span>
                  </label>
                )}
              </div>
              {/* Sample chips */}
              <div className="mt-3 flex items-center flex-wrap gap-2">
                <span className="text-[11px] font-medium" style={{ color: textFaint }}>Try a sample:</span>
                {SAMPLE_CHIPS.map((chip) => (
                  <button key={chip.label} type="button" onClick={() => handleChip(chip)}
                    className="px-2.5 py-1 text-[11px] font-semibold rounded-full border transition-all"
                    style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: amber }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#261E0F'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#1F1A0F'}>
                    {chip.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Action panel */}
            <div className="flex flex-col justify-between gap-4">
              <div className="p-4 rounded-xl border text-xs space-y-3" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                {[
                  { step: '1', label: 'Category Match', desc: 'Garment type compatibility via YOLO detection' },
                  { step: '2', label: 'Color Similarity', desc: 'CIE LAB perceptual distance on dominant hues' },
                  { step: '3', label: 'Visual / Texture', desc: 'Pattern similarity via 576‑dim embeddings' },
                ].map((s) => (
                  <div key={s.step} className="flex items-start gap-3">
                    <span className="w-5 h-5 rounded-full text-[10px] font-black flex items-center justify-center shrink-0 mt-0.5"
                      style={{ backgroundColor: '#1F1A0F', color: amber }}>{s.step}</span>
                    <div>
                      <p className="font-semibold" style={{ color: textMain }}>{s.label}</p>
                      <p className="leading-snug" style={{ color: textMuted }}>{s.desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              {error && (
                <div className="p-3 rounded-xl border text-xs" style={{ backgroundColor: '#1F1210', borderColor: '#4A2020', color: '#F87171' }}>{error}</div>
              )}

              <button type="submit" disabled={isEvaluating || !file}
                className="w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
                style={isEvaluating || !file
                  ? { backgroundColor: '#1F1A0F', color: textFaint, cursor: 'not-allowed' }
                  : { backgroundColor: amber, color: '#0E0C07' }}>
                {isEvaluating
                  ? <><RefreshCw className="w-4 h-4 animate-spin" />Cross‑checking your wardrobe…</>
                  : <><Sparkles className="w-4 h-4" />Evaluate Against My Wardrobe</>}
              </button>
            </div>
          </div>
        </form>

        {/* Result */}
        {result && (() => {
          const cfg = getVerdictConfig(result.verdict, result.score);
          const match = result.matching_item;
          const matchImg = match?.crop_url || match?.image_url;
          const matchHex = match?.dominant_hex || match?.dominant_color?.hex || '#475569';
          const catPct  = result.metrics?.category_match != null ? result.metrics.category_match * 100 : (result.verdict === 'DUPLICATE' ? 100 : 60);
          const colorPct = result.metrics?.color_delta_e != null ? Math.max(0, 100 - result.metrics.color_delta_e * 4) : result.score * 100;
          const visualPct = (result.metrics?.visual_similarity || result.score) * 100;

          return (
            <div className="mt-6 rounded-2xl border p-6 space-y-5" style={{ backgroundColor: cfg.bg, borderColor: cfg.borderColor }}>
              {/* Header */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3">
                  {cfg.icon}
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase" style={{ backgroundColor: cfg.badge.bg, color: cfg.badge.text }}>{result.verdict}</span>
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold border" style={{ backgroundColor: cfg.riskBadge.bg, color: cfg.riskBadge.text, borderColor: cfg.borderColor }}>{cfg.riskBadge.label}</span>
                    </div>
                    <h3 className="text-base font-bold mt-1" style={{ color: cfg.textColor }}>{cfg.title}</h3>
                    <p className="text-xs opacity-80 mt-0.5" style={{ color: cfg.textColor }}>{cfg.sub}</p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-3xl font-black font-mono" style={{ color: textMain }}>{(result.score*100).toFixed(0)}%</span>
                  <span className="block text-[10px] uppercase font-semibold mt-0.5" style={{ color: textFaint }}>Match Index</span>
                </div>
              </div>

              {/* Advice */}
              <div className="p-4 rounded-xl border text-sm" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                <span className="font-semibold block mb-1" style={{ color: textMain }}>FitLoop Advice</span>
                <p className="text-xs leading-relaxed" style={{ color: textMuted }}>{result.advice}</p>
              </div>

              {/* Bars */}
              <div className="p-4 rounded-xl border space-y-3" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                <h4 className="text-xs font-bold uppercase tracking-wide mb-3" style={{ color: textMuted }}>Match Breakdown</h4>
                <ProgressBar label="Category Match"    value={catPct}   color={cfg.barColor} />
                <ProgressBar label="Color Similarity"  value={colorPct} color={cfg.barColor} />
                <ProgressBar label="Texture & Pattern" value={visualPct} color={cfg.barColor} />
              </div>

              {/* Side-by-side */}
              {match && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl border" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                    <h4 className="text-xs font-bold uppercase mb-3" style={{ color: textMuted }}>Your Prospective Buy</h4>
                    <div className="w-full h-32 rounded-lg overflow-hidden mb-2" style={{ backgroundColor: '#0E0C07' }}>
                      {preview ? <img src={preview} alt="Prospective" className="w-full h-full object-contain" />
                        : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-7 h-7" style={{ color: textFaint }} /></div>}
                    </div>
                    <p className="text-xs font-semibold capitalize" style={{ color: textMain }}>{result.category}</p>
                    <div className="flex items-center gap-1.5 mt-1">
                      <span className="w-3 h-3 rounded-full border" style={{ backgroundColor: result.dominant_hex, borderColor: '#3A3020' }} />
                      <span className="text-[10px] font-mono" style={{ color: textMuted }}>{result.dominant_hex}</span>
                    </div>
                  </div>
                  <div className="p-4 rounded-xl border" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                    <h4 className="text-xs font-bold uppercase mb-3" style={{ color: textMuted }}>Closest Closet Match</h4>
                    <div className="w-full h-32 rounded-lg overflow-hidden mb-2" style={{ backgroundColor: '#0E0C07' }}>
                      {matchImg
                        ? <img src={matchImg.startsWith('http') ? matchImg : `http://127.0.0.1:8000${matchImg}`} alt="Match" className="w-full h-full object-cover" />
                        : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-7 h-7" style={{ color: textFaint }} /></div>}
                    </div>
                    <p className="text-xs font-semibold capitalize" style={{ color: textMain }}>{match.category}</p>
                    <div className="flex items-center gap-1.5 mt-1">
                      <span className="w-3 h-3 rounded-full border" style={{ backgroundColor: matchHex, borderColor: '#3A3020' }} />
                      <span className="text-[10px] font-mono" style={{ color: textMuted }}>{matchHex}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

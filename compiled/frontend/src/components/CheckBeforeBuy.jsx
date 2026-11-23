import React, { useState } from 'react';
import { ShoppingBag, Upload, AlertTriangle, CheckCircle2, XCircle, ArrowRight, RefreshCw, Sparkles } from 'lucide-react';
import { runCheckBeforeBuy } from '../api/wardrobeClient';

export default function CheckBeforeBuy({ userId }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload a photo of the prospective purchase.');
      return;
    }
    setIsEvaluating(true);
    setError(null);
    try {
      const res = await runCheckBeforeBuy(file, userId);
      setResult(res);
    } catch (err) {
      setError(err.message || 'Evaluation failed');
    } finally {
      setIsEvaluating(false);
    }
  };

  const getVerdictBadge = (verdict, score) => {
    if (verdict === 'DUPLICATE') {
      return {
        bg: 'bg-rose-50 border-rose-200 text-rose-800',
        badge: 'bg-rose-600 text-white',
        icon: <XCircle className="w-8 h-8 text-rose-600 shrink-0" />,
        title: 'Duplicate Detected — Avoid Purchase',
        subtitle: `Match Confidence: ${(score * 100).toFixed(1)}% (Threshold ≥ 85%)`,
      };
    }
    if (verdict === 'SIMILAR') {
      return {
        bg: 'bg-amber-50 border-amber-200 text-amber-800',
        badge: 'bg-amber-600 text-white',
        icon: <AlertTriangle className="w-8 h-8 text-amber-600 shrink-0" />,
        title: 'Similar Item in Your Closet',
        subtitle: `Similarity Score: ${(score * 100).toFixed(1)}% (Range 65% - 84%)`,
      };
    }
    return {
      bg: 'bg-emerald-50 border-emerald-200 text-emerald-800',
      badge: 'bg-emerald-600 text-white',
      icon: <CheckCircle2 className="w-8 h-8 text-emerald-600 shrink-0" />,
      title: 'Unique Item — Recommended Buy',
      subtitle: `Novelty Score: ${((1 - score) * 100).toFixed(1)}% Unique`,
    };
  };

  return (
    <div className="space-y-8">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <ShoppingBag className="w-6 h-6 text-indigo-600" />
              Check-Before-Buy Duplicate Prevention
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Upload a snapshot of an item you are considering buying (in-store tag, rack photo, or online screenshot).
            </p>
          </div>
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200">
            Duplicate Prevention AI
          </span>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Upload Box */}
            <div className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center hover:border-indigo-500 transition-colors bg-slate-50 flex flex-col justify-center items-center relative">
              {preview ? (
                <div className="relative w-full h-56 flex items-center justify-center">
                  <img
                    src={preview}
                    alt="Prospective Item"
                    className="max-h-full max-w-full rounded-lg object-contain shadow-sm"
                  />
                  <button
                    type="button"
                    onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                    className="absolute top-2 right-2 bg-slate-900/70 text-white rounded-full px-2 py-1 text-xs hover:bg-slate-900"
                  >
                    Change
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer flex flex-col items-center">
                  <Upload className="w-10 h-10 text-slate-400 mb-2" />
                  <span className="text-sm font-semibold text-slate-700">Upload Prospective Purchase Photo</span>
                  <span className="text-xs text-slate-500 mt-1">Single garment photo or store hanger</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>

            {/* Explanation & Action */}
            <div className="flex flex-col justify-between space-y-4">
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-slate-800">Multi-Attribute Comparison Engine:</h4>
                <div className="space-y-2 text-xs text-slate-600">
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="font-semibold text-slate-900">1. Category Compatibility</span>
                    <p className="text-slate-500 mt-0.5">YOLO26n determines matching category weights (e.g. T-Shirt vs Trousers).</p>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="font-semibold text-slate-900">2. CIE LAB Delta-E Metric</span>
                    <p className="text-slate-500 mt-0.5">Calculates Euclidean perceptual color difference across core 60% crop.</p>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="font-semibold text-slate-900">3. MobileNetV3 Visual Cosine Match</span>
                    <p className="text-slate-500 mt-0.5">Compares 576-dimensional pattern and texture embeddings.</p>
                  </div>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-xs">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isEvaluating || !file}
                className={`w-full py-3 px-4 rounded-xl font-semibold text-white shadow-md flex items-center justify-center gap-2 transition-all ${
                  isEvaluating || !file
                    ? 'bg-slate-400 cursor-not-allowed'
                    : 'bg-rose-600 hover:bg-rose-700 hover:shadow-lg active:scale-[0.99]'
                }`}
              >
                {isEvaluating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Cross-Checking Against Your Wardrobe...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Evaluate Against My Wardrobe
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* Evaluation Verdict Result */}
        {result && (() => {
          const config = getVerdictBadge(result.verdict, result.score);
          return (
            <div className={`mt-6 p-6 rounded-2xl border ${config.bg} space-y-6 transition-all`}>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  {config.icon}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase ${config.badge}`}>
                        {result.verdict}
                      </span>
                      <h3 className="text-lg font-bold">{config.title}</h3>
                    </div>
                    <p className="text-xs mt-1 opacity-90">{config.subtitle}</p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-2xl font-black font-mono">{(result.score * 100).toFixed(0)}%</span>
                  <span className="block text-[10px] uppercase font-semibold text-slate-500">Match Index</span>
                </div>
              </div>

              {/* Advice */}
              <div className="bg-white/80 backdrop-blur-xs p-4 rounded-xl border border-black/5 text-sm">
                <span className="font-semibold text-slate-900 block mb-1">FitLoop Purchase Advice:</span>
                <p className="text-slate-700">{result.advice}</p>
              </div>

              {/* Metrics & Side by Side */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-white/70 p-4 rounded-xl border border-black/5">
                  <h4 className="text-xs font-bold text-slate-800 uppercase mb-3">Detection Metrics</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Detected Category:</span>
                      <span className="font-semibold text-slate-800 capitalize">{result.category}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500">Dominant Color:</span>
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-3.5 h-3.5 rounded-full border border-slate-300"
                          style={{ backgroundColor: result.dominant_hex }}
                        />
                        <span className="font-mono text-slate-800">{result.dominant_hex}</span>
                      </div>
                    </div>
                    {result.metrics && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-slate-500">CIE LAB ΔE Color Distance:</span>
                          <span className="font-mono font-semibold">{result.metrics.color_delta_e?.toFixed(1) ?? 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Visual Feature Cosine Match:</span>
                          <span className="font-mono font-semibold">{((result.metrics.visual_similarity || 0) * 100).toFixed(1)}%</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {result.matching_item && (
                  <div className="bg-white/70 p-4 rounded-xl border border-black/5">
                    <h4 className="text-xs font-bold text-slate-800 uppercase mb-3">Closest Closet Match</h4>
                    <div className="flex items-center gap-3">
                      <div className="w-16 h-16 bg-slate-200 rounded-lg overflow-hidden shrink-0">
                        {result.matching_item.crop_url ? (
                          <img
                            src={`http://127.0.0.1:8000${result.matching_item.crop_url}`}
                            alt="Closest Match"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-400">
                            <Shirt className="w-6 h-6" />
                          </div>
                        )}
                      </div>
                      <div className="text-xs space-y-1">
                        <p className="font-semibold text-slate-900 capitalize">{result.matching_item.category}</p>
                        <div className="flex items-center gap-1.5">
                          <span
                            className="w-3 h-3 rounded-full border border-slate-300"
                            style={{ backgroundColor: result.matching_item.dominant_hex }}
                          />
                          <span className="font-mono text-slate-600">{result.matching_item.dominant_hex}</span>
                        </div>
                        <p className="text-[10px] text-slate-500">ID: {result.matching_item.id?.slice(0, 8)}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}

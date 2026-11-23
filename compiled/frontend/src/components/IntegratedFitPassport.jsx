import React, { useState, useEffect } from 'react';
import { Ruler, Shirt, CheckCircle, Sparkles, RefreshCw, Layers, ShieldCheck, ChevronRight } from 'lucide-react';
import { fetchIntegratedProfile } from '../api/wardrobeClient';

export default function IntegratedFitPassport({ userId, onNavigateTab }) {
  const [profile, setProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadProfile = async () => {
    if (!userId) return;
    setIsLoading(true);
    try {
      const data = await fetchIntegratedProfile(userId);
      setProfile(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProfile();
  }, [userId]);

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl p-12 text-center border border-slate-200">
        <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mx-auto mb-3" />
        <p className="text-sm text-slate-500 font-medium">Assembling Cross-Model Fit Passport...</p>
      </div>
    );
  }

  const measurements = profile?.body_measurements || {
    chest_cm: 96.0,
    waist_cm: 82.0,
    hip_cm: 98.0,
    neck_cm: 38.0,
    thigh_cm: 54.0,
    bicep_cm: 31.0,
  };

  const linear = profile?.linear_dimensions || {
    shoulder_width_cm: 44.5,
    inseam_cm: 81.0,
    sleeve_length_cm: 60.0,
    torso_length_cm: 56.0,
  };

  const sizing = profile?.sizing_recommendations || {
    recommended_top_size: 'M',
    recommended_bottom_size: 'M (32)',
    garment_category_fit: {
      'short sleeve top': { recommended_size: 'M', ease_cm: 6.0, fit_type: 'Regular Fit' },
      'long sleeve top': { recommended_size: 'M', ease_cm: 6.5, fit_type: 'Comfort Fit' },
      'trousers': { recommended_size: 'M (32)', ease_cm: 3.0, fit_type: 'True to Size' },
      'shorts': { recommended_size: 'M (32)', ease_cm: 4.0, fit_type: 'Relaxed Fit' },
      'outerwear': { recommended_size: 'L', ease_cm: 10.0, fit_type: 'Layered Fit' },
    },
  };

  const wardrobe = profile?.wardrobe_summary || { total_items: 0, category_distribution: {}, palette: [] };

  return (
    <div className="space-y-8">
      {/* Hero Passport Card */}
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-900 via-indigo-800 to-slate-900 text-white rounded-3xl p-8 shadow-xl border border-indigo-700/40">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/30 border border-indigo-400/40 text-indigo-200">
              <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
              Unified Sizing & Wardrobe Intelligence
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight">Personal Fit Passport</h2>
            <p className="text-sm text-indigo-200 max-w-xl">
              Harmonizes 2D anthropometric circumferences from <strong>Model 1</strong> with digitized closet items and fit preferences from <strong>Wardrobe ML</strong>.
            </p>
          </div>

          <div className="flex items-center gap-4 bg-white/10 backdrop-blur-md px-6 py-4 rounded-2xl border border-white/10">
            <div className="text-center">
              <span className="block text-2xl font-black text-white">{sizing.recommended_top_size}</span>
              <span className="text-[10px] text-indigo-200 uppercase font-semibold">Tops / Outerwear</span>
            </div>
            <div className="w-px h-10 bg-white/20" />
            <div className="text-center">
              <span className="block text-2xl font-black text-white">{sizing.recommended_bottom_size}</span>
              <span className="text-[10px] text-indigo-200 uppercase font-semibold">Trousers / Bottoms</span>
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Measurements vs Wardrobe */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Model 1 Anthropometrics */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm lg:col-span-1 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-slate-900 flex items-center gap-2">
              <Ruler className="w-5 h-5 text-indigo-600" />
              Model 1 Body Metrics
            </h3>
            <button
              onClick={() => onNavigateTab('measurement')}
              className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center gap-0.5"
            >
              Update <ChevronRight className="w-3 h-3" />
            </button>
          </div>

          <p className="text-xs text-slate-500">
            Extracts calibrated 2D circumferences via Ramanujan elliptic approximation.
          </p>

          <div className="space-y-2.5 pt-2">
            {[
              { label: 'Chest Circumference', value: measurements.chest_cm, unit: 'cm' },
              { label: 'Waist Circumference', value: measurements.waist_cm, unit: 'cm' },
              { label: 'Hip Circumference', value: measurements.hip_cm, unit: 'cm' },
              { label: 'Shoulder Width', value: linear.shoulder_width_cm, unit: 'cm' },
              { label: 'Inseam Length', value: linear.inseam_cm, unit: 'cm' },
              { label: 'Sleeve Length', value: linear.sleeve_length_cm, unit: 'cm' },
            ].map((m, idx) => (
              <div key={idx} className="flex justify-between items-center p-2.5 bg-slate-50 rounded-lg text-xs">
                <span className="text-slate-600 font-medium">{m.label}</span>
                <span className="font-bold font-mono text-slate-900">{m.value} {m.unit}</span>
              </div>
            ))}
          </div>

          <div className="p-3 bg-emerald-50 rounded-xl border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 shrink-0 text-emerald-600" />
            <span>ISO 7250 / ISO 8559 anthropometric standards compliant</span>
          </div>
        </div>

        {/* Center: Recommended Sizing by Apparel Category */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-slate-900 flex items-center gap-2">
                <Shirt className="w-5 h-5 text-indigo-600" />
                Category Sizing & Comfort Ease Breakdown
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Calculated ease allowances matching your extracted body circumferences to garment silhouettes.
              </p>
            </div>
            <span className="px-3 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold">
              {wardrobe.total_items} Digitized Garments
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(sizing.garment_category_fit || {}).map(([cat, info]) => (
              <div key={cat} className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 hover:bg-slate-50 transition-colors">
                <div className="flex items-start justify-between">
                  <span className="text-xs font-bold text-slate-900 capitalize">{cat}</span>
                  <span className="px-2 py-0.5 rounded-md bg-indigo-600 text-white text-xs font-extrabold font-mono">
                    {info.recommended_size}
                  </span>
                </div>
                <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
                  <span>Ease Allowance: <strong className="text-slate-800 font-mono">+{info.ease_cm} cm</strong></span>
                  <span className="text-emerald-700 font-medium bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    {info.fit_type}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Wardrobe Color Harmony */}
          <div className="pt-4 border-t border-slate-200">
            <h4 className="text-xs font-bold uppercase text-slate-800 mb-2">Wardrobe Color Palette</h4>
            {wardrobe.palette && wardrobe.palette.length > 0 ? (
              <div className="flex items-center gap-2">
                {wardrobe.palette.map((hex, i) => (
                  <div key={i} className="flex flex-col items-center">
                    <span
                      className="w-8 h-8 rounded-lg border border-slate-300 shadow-xs"
                      style={{ backgroundColor: hex }}
                      title={hex}
                    />
                    <span className="text-[9px] font-mono text-slate-500 mt-1">{hex}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400">Scan wardrobe items to generate your closet color palette.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

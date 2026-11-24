import React, { useState, useEffect } from 'react';
import { Ruler, Shirt, Sparkles, RefreshCw, ChevronRight, Download, ShieldCheck, CheckCircle2, Sliders, Zap } from 'lucide-react';
import { fetchIntegratedProfile } from '../api/wardrobeClient';

const pageBg   = '#0E0C07';
const cardBg   = '#181410';
const subBg    = '#252018';    // subsection panels — warm sand-brown
const subBorder= '#342A1C';   // subsection border — warmer
const border   = '#2A2218';
const amber    = '#D97706';
const textMain = '#EDE8DF';
const textMuted= '#7A6A55';
const textFaint= '#3A3020';

/* Body wireframe zones */
const ZONE_PATHS = {
  chest:    'M28 48 Q24 55 24 65 L56 65 Q56 55 52 48 Q42 52 40 52 Q38 52 28 48Z',
  waist:    'M24 65 L56 65 L58 78 L22 78Z',
  hip:      'M22 78 Q18 90 20 100 L60 100 Q62 90 58 78Z',
  shoulder: 'M28 42 L52 42 L54 50 Q46 54 40 54 Q34 54 26 50Z',
  inseam:   'M20 100 L28 148 L36 148 L40 118 L44 148 L52 148 L60 100Z',
  sleeve:   'M18 44 Q8 58 10 90 L18 90 Q14 62 24 50Z',
};

function BodyWireframe({ highlightZone }) {
  return (
    <svg viewBox="0 0 80 180" className="w-full h-full" fill="none">
      <ellipse cx="40" cy="20" rx="12" ry="14" fill="#2A2218" stroke="#3A3020" strokeWidth="1" />
      <rect x="35" y="33" width="10" height="8" rx="3" fill="#2A2218" stroke="#3A3020" strokeWidth="1" />
      {Object.entries(ZONE_PATHS).map(([zone, d]) => (
        <path key={zone} d={d}
          fill={highlightZone === zone ? amber : '#2A2218'}
          stroke={highlightZone === zone ? '#F59E0B' : '#3A3020'}
          strokeWidth="1"
          className="transition-all duration-300"
          opacity={highlightZone === zone ? 1 : 0.8}
        />
      ))}
    </svg>
  );
}

function colorDistance(hex1, hex2) {
  const parse = (h) => { const c = (h||'000000').replace('#',''); return [parseInt(c.slice(0,2),16),parseInt(c.slice(2,4),16),parseInt(c.slice(4,6),16)]; };
  const [r1,g1,b1]=parse(hex1); const [r2,g2,b2]=parse(hex2);
  return Math.sqrt((r1-r2)**2+(g1-g2)**2+(b1-b2)**2);
}

function ColorPaletteBar({ palette }) {
  if (!palette || palette.length === 0) return (
    <p className="text-xs" style={{ color: textFaint }}>Scan wardrobe items to generate your colour palette.</p>
  );
  const groups = [];
  for (const hex of palette) {
    const existing = groups.find((g) => colorDistance(g.hex, hex) < 40);
    if (existing) existing.count++;
    else groups.push({ hex, count: 1 });
  }
  groups.sort((a,b) => b.count - a.count);
  const total = groups.reduce((s,g) => s+g.count, 0);
  return (
    <div className="space-y-2">
      <div className="flex h-8 rounded-xl overflow-hidden border" style={{ borderColor: border }}>
        {groups.map((g,i) => (
          <div key={i} style={{ width: `${(g.count/total)*100}%`, backgroundColor: g.hex }}
            title={`${g.hex} · ${g.count} item${g.count!==1?'s':''}`}
            className="transition-all hover:opacity-80 cursor-pointer" />
        ))}
      </div>
      <div className="flex flex-wrap gap-3">
        {groups.map((g,i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full border shrink-0" style={{ backgroundColor: g.hex, borderColor: '#3A3020' }} />
            <span className="text-[10px] font-mono" style={{ color: textMuted }}>{g.hex}</span>
            <span className="text-[10px]" style={{ color: textFaint }}>· {g.count} item{g.count!==1?'s':''}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function classifyTop(chest) {
  if (chest < 88) return 'XS';
  if (chest < 94) return 'S';
  if (chest < 102) return 'M';
  if (chest < 110) return 'L';
  if (chest < 118) return 'XL';
  return 'XXL';
}

function classifyBottom(waist) {
  if (waist < 74) return 'XS (28)';
  if (waist < 80) return 'S (30)';
  if (waist < 86) return 'M (32)';
  if (waist < 92) return 'L (34)';
  if (waist < 98) return 'XL (36)';
  return 'XXL (38)';
}

export default function IntegratedFitPassport({ userId = '1', scanResult, onNavigateTab }) {
  const [profile, setProfile]         = useState(null);
  const [isLoading, setIsLoading]     = useState(true);
  const [hoveredZone, setHoveredZone] = useState(null);
  const [exportNotice, setExportNotice] = useState(false);

  const loadProfile = async () => {
    setIsLoading(true);
    try {
      const data = await fetchIntegratedProfile(userId);
      setProfile(data);
    } catch (e) {
      console.error('Failed to load profile:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProfile();
  }, [userId]);

  // Quick helper to simulate/load real sample scan parameters
  const handleLoadSampleScan = async (sampleChest = 98.4, sampleWaist = 81.2, sampleHip = 99.6, sampleH = 178.0) => {
    setIsLoading(true);
    try {
      await fetch(`/api/v1/profile/${userId}/measurements`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chest_cm: sampleChest,
          waist_cm: sampleWaist,
          hip_cm: sampleHip,
          standing_height_cm: sampleH,
        }),
      });
      await loadProfile();
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Resolve whether real measurements are available (either from active scan in memory or from backend)
  const isFromActiveScan = Boolean(scanResult?.measurements);
  const hasRealMeasurements = isFromActiveScan || Boolean(profile?.has_body_measurements);

  // Extract dynamic values
  let chest = 96.0;
  let waist = 82.0;
  let hip = 98.0;
  let shoulder = 44.5;
  let inseam = 81.0;
  let sleeve = 60.0;
  let stature = 175.0;

  if (isFromActiveScan) {
    const m = scanResult.measurements;
    chest = m?.chest?.circumference_cm ?? chest;
    waist = m?.waist?.circumference_cm ?? waist;
    hip = m?.hips?.circumference_cm ?? m?.hip?.circumference_cm ?? hip;
    stature = scanResult.calibration?.known_height_cm || 175.0;
    shoulder = scanResult.linear_dimensions?.shoulder_width_cm ?? Math.round(stature * 0.252 * 10) / 10;
    inseam = scanResult.linear_dimensions?.inseam_cm ?? Math.round(stature * 0.465 * 10) / 10;
    sleeve = scanResult.linear_dimensions?.sleeve_length_cm ?? Math.round(stature * 0.342 * 10) / 10;
  } else if (profile?.body_measurements) {
    const bm = profile.body_measurements;
    const ld = profile.linear_dimensions || {};
    chest = bm.chest_cm ?? bm.chest ?? chest;
    waist = bm.waist_cm ?? bm.waist ?? waist;
    hip = bm.hip_cm ?? bm.hip ?? hip;
    shoulder = ld.shoulder_width_cm ?? shoulder;
    inseam = ld.inseam_cm ?? inseam;
    sleeve = ld.sleeve_length_cm ?? sleeve;
  }

  // Dynamic ISO sizes based on actual measured circumferences
  const recTop = classifyTop(chest);
  const recBottom = classifyBottom(waist);

  const wardrobe = profile?.wardrobe_summary || { total_items: 0, category_distribution: {}, palette: [] };
  const categoryFit = profile?.sizing_recommendations?.garment_category_fit || {
    'short sleeve top': { recommended_size: recTop, ease_cm: 6.0, fit_type: 'Regular Fit', wardrobe_count: 0 },
    'long sleeve top':  { recommended_size: recTop, ease_cm: 6.5, fit_type: 'Comfort Fit', wardrobe_count: 0 },
    'trousers':         { recommended_size: recBottom, ease_cm: 3.0, fit_type: 'True to Size', wardrobe_count: 0 },
    'shorts':           { recommended_size: recBottom, ease_cm: 4.0, fit_type: 'Relaxed Fit', wardrobe_count: 0 },
    'outerwear':        { recommended_size: recTop, ease_cm: 10.0, fit_type: 'Layered Fit', wardrobe_count: 0 },
  };

  const bodyRows = [
    { key: 'chest',    zone: 'chest',    label: 'Chest',    value: Number(chest).toFixed(1),     unit: 'cm' },
    { key: 'waist',    zone: 'waist',    label: 'Waist',    value: Number(waist).toFixed(1),     unit: 'cm' },
    { key: 'hip',      zone: 'hip',      label: 'Hip',      value: Number(hip).toFixed(1),       unit: 'cm' },
    { key: 'shoulder', zone: 'shoulder', label: 'Shoulder', value: Number(shoulder).toFixed(1),  unit: 'cm' },
    { key: 'inseam',   zone: 'inseam',   label: 'Inseam',   value: Number(inseam).toFixed(1),    unit: 'cm' },
    { key: 'sleeve',   zone: 'sleeve',   label: 'Sleeve',   value: Number(sleeve).toFixed(1),    unit: 'cm' },
  ];

  if (isLoading && !profile) {
    return (
      <div className="rounded-2xl p-16 text-center border" style={{ backgroundColor: cardBg, borderColor: border }}>
        <RefreshCw className="w-7 h-7 animate-spin mx-auto mb-3" style={{ color: amber }} />
        <p className="text-sm font-medium" style={{ color: textMuted }}>Assembling your Fit Passport…</p>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Export Feedback Toast */}
      {exportNotice && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 rounded-xl border shadow-xl text-xs font-semibold"
          style={{ backgroundColor: cardBg, borderColor: amber, color: textMain }}>
          <CheckCircle2 className="w-4 h-4" style={{ color: '#10B981' }} />
          <span>Fit Passport Card exported to clipboard & JSON profile!</span>
        </div>
      )}

      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-3xl text-white shadow-xl"
        style={{ background: 'linear-gradient(135deg, #1A1208 0%, #2A1E0A 40%, #1A1208 100%)', border: '1px solid #3A2E1A' }}>
        <div className="absolute inset-0 opacity-[0.04] pointer-events-none"
          style={{ backgroundImage: 'radial-gradient(circle at 1px 1px, #D97706 1px, transparent 0)', backgroundSize: '28px 28px' }} />
        <div className="absolute top-0 right-0 w-64 h-64 rounded-full pointer-events-none opacity-20"
          style={{ background: 'radial-gradient(circle, #D97706 0%, transparent 70%)', transform: 'translate(30%, -30%)' }} />

        <div className="relative z-10 p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border"
                style={{ backgroundColor: 'rgba(217,119,6,0.15)', borderColor: 'rgba(217,119,6,0.3)', color: '#FCD34D' }}>
                <Sparkles className="w-3.5 h-3.5" />
                Unified Sizing & Wardrobe Intelligence
              </div>
              {hasRealMeasurements ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border"
                  style={{ backgroundColor: '#0F2614', borderColor: '#1E4A28', color: '#4ADE80' }}>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Live AI Measured (Model 1)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border"
                  style={{ backgroundColor: '#20180F', borderColor: '#3A2E1A', color: amber }}>
                  <Sliders className="w-3.5 h-3.5" />
                  Stature Baseline (175 cm)
                </span>
              )}
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight" style={{ color: '#F5F0E8' }}>Personal Fit Passport</h2>
            <p className="text-sm max-w-md leading-relaxed" style={{ color: '#7A6A55' }}>
              Real body measurements dynamically linked to your digitized wardrobe — customized sizing allowances and color profile.
            </p>
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            {/* Top & Bottom Size Pills */}
            <div className="flex items-center gap-4 px-6 py-4 rounded-2xl border"
              style={{ backgroundColor: 'rgba(14,12,7,0.5)', borderColor: 'rgba(217,119,6,0.2)' }}>
              <div className="text-center">
                <span className="block text-2xl font-black" style={{ color: amber }}>{recTop}</span>
                <span className="text-[10px] uppercase font-semibold" style={{ color: textFaint }}>Tops</span>
              </div>
              <div className="w-px h-10" style={{ backgroundColor: '#3A2E1A' }} />
              <div className="text-center">
                <span className="block text-2xl font-black" style={{ color: amber }}>{recBottom}</span>
                <span className="text-[10px] uppercase font-semibold" style={{ color: textFaint }}>Bottoms</span>
              </div>
            </div>

            <button
              onClick={() => {
                const fitCardData = JSON.stringify({
                  user: profile?.name || 'Kunal Kakkar',
                  measurements: { chest, waist, hip, shoulder, inseam, sleeve },
                  sizing: { top: recTop, bottom: recBottom },
                  wardrobe_items: wardrobe.total_items,
                }, null, 2);
                navigator.clipboard?.writeText(fitCardData);
                setExportNotice(true);
                setTimeout(() => setExportNotice(false), 3000);
              }}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all border cursor-pointer hover:brightness-110"
              style={{ backgroundColor: 'rgba(217,119,6,0.15)', borderColor: 'rgba(217,119,6,0.3)', color: '#FCD34D' }}>
              <Download className="w-4 h-4" /> Export Fit Card
            </button>
          </div>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Body Metrics + Wireframe */}
        <div className="rounded-2xl border p-6 space-y-4 lg:col-span-1" style={{ backgroundColor: cardBg, borderColor: border }}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold flex items-center gap-2 text-sm" style={{ color: textMain }}>
                <Ruler className="w-4 h-4" style={{ color: amber }} /> Body Metrics
              </h3>
              <span className="text-[11px]" style={{ color: textMuted }}>
                {hasRealMeasurements ? 'Live Circumferences' : 'Calibrated Anthropometry'}
              </span>
            </div>
            <button
              onClick={() => onNavigateTab('measurement')}
              className="text-xs font-semibold flex items-center gap-0.5 hover:underline cursor-pointer"
              style={{ color: amber }}>
              Measure in Tab 1 <ChevronRight className="w-3 h-3" />
            </button>
          </div>

          <div className="flex gap-4">
            <div className="w-20 shrink-0"><BodyWireframe highlightZone={hoveredZone} /></div>
            <div className="flex-1 space-y-1.5">
              {bodyRows.map((row) => (
                <div key={row.key}
                  onMouseEnter={() => setHoveredZone(row.zone)}
                  onMouseLeave={() => setHoveredZone(null)}
                  className="flex justify-between items-center px-3 py-2 rounded-xl text-xs cursor-default transition-all"
                  style={{
                    backgroundColor: hoveredZone === row.zone ? '#1F1A0F' : subBg,
                    border: `1px solid ${hoveredZone === row.zone ? '#3A2E1A' : subBorder}`,
                  }}>
                  <span style={{ color: hoveredZone === row.zone ? amber : textMuted }}>{row.label}</span>
                  <span className="font-bold font-mono" style={{ color: hoveredZone === row.zone ? '#FCD34D' : textMain }}>
                    {row.value} {row.unit}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick sync & sample buttons */}
          <div className="pt-2 flex items-center gap-2">
            <button
              onClick={() => handleLoadSampleScan(94.5, 79.8, 97.2, 175.0)}
              className="flex-1 flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl border text-[11px] font-semibold transition-all cursor-pointer hover:brightness-125"
              style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
              <Zap className="w-3 h-3" style={{ color: amber }} /> Sample Size M
            </button>
            <button
              onClick={() => handleLoadSampleScan(106.0, 92.5, 105.0, 182.0)}
              className="flex-1 flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl border text-[11px] font-semibold transition-all cursor-pointer hover:brightness-125"
              style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
              <Zap className="w-3 h-3" style={{ color: amber }} /> Sample Size L
            </button>
          </div>

          <div className="flex items-center gap-2 p-3 rounded-xl border text-xs"
            style={{ backgroundColor: '#0F1F12', borderColor: '#1A3A20', color: '#6EE7B7' }}>
            <ShieldCheck className="w-4 h-4 shrink-0" style={{ color: '#10B981' }} />
            <span>ISO 7250 / ISO 8559 Anthropometric Standards</span>
          </div>
        </div>

        {/* Sizing breakdown */}
        <div className="rounded-2xl border p-6 space-y-5 lg:col-span-2" style={{ backgroundColor: cardBg, borderColor: border }}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold flex items-center gap-2 text-sm" style={{ color: textMain }}>
                <Shirt className="w-4 h-4" style={{ color: amber }} /> Category Sizing Breakdown
              </h3>
              <p className="text-xs mt-0.5" style={{ color: textMuted }}>
                Ease allowances tailored to your exact measurements across your wardrobe categories.
              </p>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-semibold border"
              style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
              {wardrobe.total_items} garments in closet
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(categoryFit).map(([cat, info]) => {
              const count = info.wardrobe_count || wardrobe.category_distribution?.[cat] || 0;
              return (
                <div key={cat} className="p-4 rounded-xl border transition-all hover:-translate-y-0.5"
                  style={{ backgroundColor: subBg, borderColor: subBorder }}
                  onMouseEnter={(e) => e.currentTarget.style.borderColor = '#4A3A28'}
                  onMouseLeave={(e) => e.currentTarget.style.borderColor = subBorder}>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold capitalize block" style={{ color: textMain }}>{cat}</span>
                      {count > 0 && (
                        <span className="text-[10px]" style={{ color: textMuted }}>
                          {count} item{count !== 1 ? 's' : ''} in wardrobe
                        </span>
                      )}
                    </div>
                    <span className="px-2.5 py-1 rounded-lg text-xs font-extrabold font-mono border"
                      style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: amber }}>
                      {info.recommended_size}
                    </span>
                  </div>
                  <div className="mt-3 flex items-center justify-between text-xs" style={{ color: textMuted }}>
                    <span>Ease: <strong className="font-mono" style={{ color: textMain }}>+{info.ease_cm} cm</strong></span>
                    <span className="px-2 py-0.5 rounded-lg border font-semibold text-[10px]"
                      style={{ backgroundColor: '#0F1F12', borderColor: '#1A3A20', color: '#6EE7B7' }}>
                      {info.fit_type}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Colour palette bar */}
          <div className="pt-4 border-t" style={{ borderColor: border }}>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: textMuted }}>
                Wardrobe Colour Palette ({wardrobe.palette?.length || 0} pieces)
              </h4>
              <button onClick={loadProfile} className="text-xs font-semibold flex items-center gap-1 cursor-pointer" style={{ color: amber }}>
                <RefreshCw className="w-3 h-3" /> Sync
              </button>
            </div>
            <ColorPaletteBar palette={wardrobe.palette} />
          </div>
        </div>
      </div>
    </div>
  );
}

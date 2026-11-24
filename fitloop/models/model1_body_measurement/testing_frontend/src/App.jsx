import React, { useState, useEffect } from 'react';
import { Ruler, Shirt, ShoppingBag, Sparkles, User, AlertCircle } from 'lucide-react';
import ImageUploadCard from './components/ImageUploadCard';
import HeightInput from './components/HeightInput';
import ActionControls from './components/ActionControls';
import ResultsPanel from './components/ResultsPanel';
import WardrobeScanner from './components/WardrobeScanner';
import CheckBeforeBuy from './components/CheckBeforeBuy';
import IntegratedFitPassport from './components/IntegratedFitPassport';
import { estimateCircumferences, validateScanRequest } from './api/client';
import { fetchSystemHealth, getOrCreateUser } from './api/wardrobeClient';

const TABS = [
  { id: 'measurement', label: 'Body Measurement', icon: Ruler },
  { id: 'wardrobe', label: 'Wardrobe Scanner', icon: Shirt },
  { id: 'check_buy', label: 'Check‑Before‑Buy', icon: ShoppingBag },
  { id: 'passport', label: 'Fit Passport', icon: Sparkles },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('measurement');
  const [currentUser, setCurrentUser] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);

  const [frontFile, setFrontFile] = useState(null);
  const [frontPreview, setFrontPreview] = useState(null);
  const [sideFile, setSideFile] = useState(null);
  const [sidePreview, setSidePreview] = useState(null);
  const [heightCm, setHeightCm] = useState('175.0');

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessages, setErrorMessages] = useState([]);
  const [result, setResult] = useState(null);
  const [simulateFailure, setSimulateFailure] = useState(false);

  const checkHealth = async () => {
    try {
      const health = await fetchSystemHealth();
      setBackendHealth(health);
      const user = await getOrCreateUser('kunal@fitloop.ai', 'Kunal Kakkar');
      if (user) setCurrentUser(user);
      return true;
    } catch { return false; }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(async () => { const ok = await checkHealth(); if (ok) clearInterval(interval); }, 2500);
    return () => clearInterval(interval);
  }, []);

  const handleFrontSelect = (f) => { setFrontFile(f); setFrontPreview(URL.createObjectURL(f)); setErrorMessages([]); };
  const handleFrontRemove = () => { if (frontPreview) URL.revokeObjectURL(frontPreview); setFrontFile(null); setFrontPreview(null); };
  const handleSideSelect = (f) => { setSideFile(f); setSidePreview(URL.createObjectURL(f)); setErrorMessages([]); };
  const handleSideRemove = () => { if (sidePreview) URL.revokeObjectURL(sidePreview); setSideFile(null); setSidePreview(null); };
  const handleReset = () => { handleFrontRemove(); handleSideRemove(); setHeightCm('175.0'); setErrorMessages([]); setResult(null); };

  const handleScan = async () => {
    setErrorMessages([]);
    const clientErrors = validateScanRequest({ frontImage: frontFile, sideImage: sideFile, heightCm });
    if (clientErrors.length > 0) { setErrorMessages(clientErrors); return; }
    setIsLoading(true);
    try {
      const response = await estimateCircumferences({
        frontImage: frontFile,
        sideImage: sideFile,
        heightCm,
        userId: currentUser?.id || '1',
        simulateFailure,
      });
      setResult(response);
    } catch (err) { setErrorMessages(err.validationErrors || [err.message || 'Unexpected error.']); }
    finally { setIsLoading(false); }
  };

  const isMockMode = import.meta.env.VITE_USE_MOCK_API === 'true';
  const canScan = Boolean(frontFile && sideFile && heightCm);

  return (
    <div className="min-h-screen flex flex-col font-sans" style={{ backgroundColor: '#0E0C07', color: '#EDE8DF' }}>

      {/* ── Header ── */}
      <header className="sticky top-0 z-30 border-b" style={{ backgroundColor: '#181410', borderColor: '#2A2218' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl flex items-center justify-center font-black text-xs tracking-widest shadow-lg"
              style={{ background: 'linear-gradient(135deg, #D97706, #92400E)', color: '#FEF3C7' }}>
              FL
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-tight" style={{ color: '#F5F0E8' }}>FitLoop AI</span>
              <span className="hidden sm:inline text-[10px] font-semibold px-2 py-0.5 rounded-full border"
                style={{ backgroundColor: '#251E12', borderColor: '#3A2E1A', color: '#D97706' }}>
                Multi‑Model Studio
              </span>
            </div>
          </div>

          {/* Right pills */}
          <div className="flex items-center gap-2.5">
            <div className={`hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border`}
              style={backendHealth
                ? { backgroundColor: '#0F1F12', borderColor: '#1A3A20', color: '#6EE7B7' }
                : { backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: '#D97706' }}>
              <span className={`w-1.5 h-1.5 rounded-full`}
                style={{ backgroundColor: backendHealth ? '#10B981' : '#D97706', animation: backendHealth ? 'pulse 2s infinite' : 'none' }} />
              {backendHealth ? 'AI Engine Ready' : 'Connecting…'}
            </div>
            {currentUser && (
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border"
                style={{ backgroundColor: '#201B12', borderColor: '#2A2218', color: '#A89070' }}>
                <User className="w-3 h-3" style={{ color: '#D97706' }} />
                {currentUser.name}
              </div>
            )}
          </div>
        </div>

        {/* Tab pill bar */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 pb-2">
          <div className="inline-flex items-center rounded-xl p-1 gap-0.5" style={{ backgroundColor: '#0E0C07' }}>
            {TABS.map((tab, i) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 whitespace-nowrap"
                  style={active
                    ? { backgroundColor: '#D97706', color: '#0E0C07' }
                    : { color: '#7A6A55', backgroundColor: 'transparent' }}
                  onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = '#D97706'; }}
                  onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = '#7A6A55'; }}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{i + 1}. </span>{tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 w-full space-y-5">
        {activeTab === 'measurement' && (
          <>
            {result ? (
              <ResultsPanel result={result} onBackToScan={() => setResult(null)} />
            ) : (
              <div className="space-y-5">
                {/* Page header card */}
                <div className="rounded-2xl border p-6" style={{ backgroundColor: '#181410', borderColor: '#2A2218' }}>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h1 className="text-lg font-bold flex items-center gap-2" style={{ color: '#F5F0E8' }}>
                        <Ruler className="w-5 h-5" style={{ color: '#D97706' }} />
                        Dual‑View Body Measurement
                      </h1>
                      <p className="text-xs mt-1 max-w-xl leading-relaxed" style={{ color: '#7A6A55' }}>
                        Upload a front‑facing and side‑profile capture — the AI calculates circumferences using dual‑perspective calibration.
                      </p>
                    </div>
                    <span className="shrink-0 inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border"
                      style={{ backgroundColor: '#1A2A18', borderColor: '#2A4A26', color: '#86EFAC' }}>
                      Calibrated (Stage 4B)
                    </span>
                  </div>
                </div>

                {errorMessages.length > 0 && (
                  <div className="rounded-xl border p-4" style={{ backgroundColor: '#252018', borderColor: '#342A1C' }}>
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" style={{ color: '#F87171' }} />
                      <ul className="text-xs space-y-0.5 list-disc list-inside" style={{ color: '#FCA5A5' }}>
                        {errorMessages.map((msg, i) => <li key={i}>{msg}</li>)}
                      </ul>
                    </div>
                  </div>
                )}

                <HeightInput value={heightCm} onChange={setHeightCm}
                  error={heightCm && (parseFloat(heightCm) < 100 || parseFloat(heightCm) > 250) ? 'Stature must be between 100–250 cm.' : null}
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <ImageUploadCard viewType="front" title="Front‑Facing Capture"
                    description="Head to toe in frame, arms in A‑pose slightly away from torso, facing forward."
                    file={frontFile} previewUrl={frontPreview}
                    onFileSelect={handleFrontSelect} onFileRemove={handleFrontRemove}
                    error={!frontFile && errorMessages.some((e) => e.includes('Front-view')) ? 'Front photograph is required' : null}
                  />
                  <ImageUploadCard viewType="side" title="Side‑Profile Capture"
                    description="Turned 90° left or right, neutral posture, arms slightly back to expose abdomen."
                    file={sideFile} previewUrl={sidePreview}
                    onFileSelect={handleSideSelect} onFileRemove={handleSideRemove}
                    error={!sideFile && errorMessages.some((e) => e.includes('Side-view')) ? 'Side photograph is required' : null}
                  />
                </div>
                <ActionControls onScan={handleScan} onReset={handleReset} isLoading={isLoading}
                  canScan={canScan} isMockMode={isMockMode}
                  simulateFailure={simulateFailure} onToggleSimulateFailure={setSimulateFailure}
                />
              </div>
            )}
          </>
        )}
        {activeTab === 'wardrobe' && <WardrobeScanner userId={currentUser?.id || '1'} />}
        {activeTab === 'check_buy' && <CheckBeforeBuy userId={currentUser?.id || '1'} />}
        {activeTab === 'passport' && (
          <IntegratedFitPassport
            userId={currentUser?.id || '1'}
            scanResult={result}
            onNavigateTab={setActiveTab}
          />
        )}
      </main>

      <footer className="border-t py-4 text-center text-[11px]" style={{ borderColor: '#2A2218', backgroundColor: '#181410', color: '#4A3D2A' }}>
        FitLoop AI · Body Measurement + Wardrobe ML + Duplicate Prevention · v2.0
      </footer>
    </div>
  );
}

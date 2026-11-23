import React, { useState, useEffect } from 'react';
import { Camera, Sparkles, AlertCircle, Shirt, ShoppingBag, Ruler, CheckCircle2, User, RefreshCw } from 'lucide-react';
import ImageUploadCard from './components/ImageUploadCard';
import HeightInput from './components/HeightInput';
import ActionControls from './components/ActionControls';
import ResultsPanel from './components/ResultsPanel';
import WardrobeScanner from './components/WardrobeScanner';
import CheckBeforeBuy from './components/CheckBeforeBuy';
import IntegratedFitPassport from './components/IntegratedFitPassport';
import { estimateCircumferences, validateScanRequest } from './api/client';
import { fetchSystemHealth, getOrCreateUser } from './api/wardrobeClient';

export default function App() {
  // Navigation tab state: 'measurement' | 'wardrobe' | 'check_buy' | 'passport'
  const [activeTab, setActiveTab] = useState('measurement');

  // Active user state
  const [currentUser, setCurrentUser] = useState(null);

  // System backend health state
  const [backendHealth, setBackendHealth] = useState(null);
  const [isHealthLoading, setIsHealthLoading] = useState(false);

  // Model 1 Input form state
  const [frontFile, setFrontFile] = useState(null);
  const [frontPreview, setFrontPreview] = useState(null);
  const [sideFile, setSideFile] = useState(null);
  const [sidePreview, setSidePreview] = useState(null);
  const [heightCm, setHeightCm] = useState('175.0');

  // Model 1 Execution state
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessages, setErrorMessages] = useState([]);
  const [result, setResult] = useState(null);
  const [simulateFailure, setSimulateFailure] = useState(false);

  // Check system health function
  const checkHealth = async () => {
    setIsHealthLoading(true);
    try {
      const health = await fetchSystemHealth();
      setBackendHealth(health);
      const user = await getOrCreateUser('kunal@fitloop.ai', 'Kunal Kakkar');
      if (user) setCurrentUser(user);
      return true;
    } catch (err) {
      console.warn('Backend not responding to health check:', err);
      return false;
    } finally {
      setIsHealthLoading(false);
    }
  };

  // Initialize user and backend health
  useEffect(() => {
    checkHealth();

    // Poll health until online
    const interval = setInterval(async () => {
      const ok = await checkHealth();
      if (ok) clearInterval(interval);
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  // File Handlers for Model 1
  const handleFrontSelect = (file) => {
    setFrontFile(file);
    setFrontPreview(URL.createObjectURL(file));
    setErrorMessages([]);
  };

  const handleFrontRemove = () => {
    if (frontPreview) URL.revokeObjectURL(frontPreview);
    setFrontFile(null);
    setFrontPreview(null);
  };

  const handleSideSelect = (file) => {
    setSideFile(file);
    setSidePreview(URL.createObjectURL(file));
    setErrorMessages([]);
  };

  const handleSideRemove = () => {
    if (sidePreview) URL.revokeObjectURL(sidePreview);
    setSideFile(null);
    setSidePreview(null);
  };

  const handleReset = () => {
    handleFrontRemove();
    handleSideRemove();
    setHeightCm('175.0');
    setErrorMessages([]);
    setResult(null);
  };

  // Run Model 1 Scan / Estimation
  const handleScan = async () => {
    setErrorMessages([]);

    const clientErrors = validateScanRequest({
      frontImage: frontFile,
      sideImage: sideFile,
      heightCm,
    });

    if (clientErrors.length > 0) {
      setErrorMessages(clientErrors);
      return;
    }

    setIsLoading(true);
    try {
      const response = await estimateCircumferences({
        frontImage: frontFile,
        sideImage: sideFile,
        heightCm,
        simulateFailure,
      });
      setResult(response);
    } catch (err) {
      if (err.validationErrors) {
        setErrorMessages(err.validationErrors);
      } else {
        setErrorMessages([err.message || 'An unexpected scan error occurred.']);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const isMockMode = import.meta.env.VITE_USE_MOCK_API === 'true';
  const canScan = Boolean(frontFile && sideFile && heightCm);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-900 text-white flex items-center justify-center font-black text-sm tracking-wider shadow-sm">
              FL
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-slate-900 tracking-tight">FitLoop AI</span>
                <span className="text-[10px] font-semibold bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200">
                  Multi-Model Studio
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Model 1 (Anthropometric) + Wardrobe ML (Garment Detection & Sizing)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Backend Health Badge */}
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border bg-slate-50 border-slate-200 text-slate-700">
              <span className={`w-2 h-2 rounded-full ${backendHealth ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`} />
              <span>{backendHealth ? 'Backend Online (Port 8000)' : 'Connecting to API...'}</span>
            </div>

            {/* User Pill */}
            {currentUser && (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200">
                <User className="w-3.5 h-3.5 text-indigo-600" />
                <span>{currentUser.name}</span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-2 border-t border-slate-100 overflow-x-auto py-1">
          {[
            { id: 'measurement', label: '1. Body Measurement (Model 1)', icon: <Ruler className="w-4 h-4" /> },
            { id: 'wardrobe', label: '2. Wardrobe Scanner (Wardrobe ML)', icon: <Shirt className="w-4 h-4" /> },
            { id: 'check_buy', label: '3. Check-Before-Buy (Duplicate AI)', icon: <ShoppingBag className="w-4 h-4" /> },
            { id: 'passport', label: '4. Integrated Fit Passport', icon: <Sparkles className="w-4 h-4" /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-lg transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 w-full space-y-6">
        {/* TAB 1: Model 1 Body Measurement */}
        {activeTab === 'measurement' && (
          <>
            {result ? (
              <ResultsPanel result={result} onBackToScan={() => setResult(null)} />
            ) : (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div>
                      <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                        <Ruler className="w-6 h-6 text-indigo-600" />
                        Dual-View Body Circumference Estimation (Model 1)
                      </h1>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                        Upload synchronized front and side photographs taken in neutral A-pose. The pipeline determines independent pixel-to-metric ratios from standing height and calculates circumferences via the Ramanujan ellipse model.
                      </p>
                    </div>
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Stage 4B Calibrated
                    </span>
                  </div>
                </div>

                {errorMessages.length > 0 && (
                  <div className="rounded-xl border border-red-300 bg-red-50 p-4 shadow-sm animate-fadeIn">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-red-800">Validation or Execution Error</h4>
                        <ul className="mt-1 list-disc list-inside text-xs text-red-700 space-y-0.5">
                          {errorMessages.map((msg, i) => (
                            <li key={i}>{msg}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {/* Stature Input */}
                <HeightInput
                  value={heightCm}
                  onChange={setHeightCm}
                  error={
                    heightCm && (parseFloat(heightCm) < 100 || parseFloat(heightCm) > 250)
                      ? 'Stature must be between 100 cm and 250 cm.'
                      : null
                  }
                />

                {/* Front and Side Uploads */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ImageUploadCard
                    viewType="front"
                    title="Front-Facing Capture"
                    description="Head to toe in frame, arms in A-pose slightly away from torso, facing forward."
                    file={frontFile}
                    previewUrl={frontPreview}
                    onFileSelect={handleFrontSelect}
                    onFileRemove={handleFrontRemove}
                    error={!frontFile && errorMessages.some((e) => e.includes('Front-view')) ? 'Front photograph is required' : null}
                  />

                  <ImageUploadCard
                    viewType="side"
                    title="Side-Profile Capture"
                    description="Turned 90° left or right, neutral posture, arms slightly back to expose abdomen."
                    file={sideFile}
                    previewUrl={sidePreview}
                    onFileSelect={handleSideSelect}
                    onFileRemove={handleSideRemove}
                    error={!sideFile && errorMessages.some((e) => e.includes('Side-view')) ? 'Side photograph is required' : null}
                  />
                </div>

                {/* Action Controls */}
                <ActionControls
                  onScan={handleScan}
                  onReset={handleReset}
                  isLoading={isLoading}
                  canScan={canScan}
                  isMockMode={isMockMode}
                  simulateFailure={simulateFailure}
                  onToggleSimulateFailure={setSimulateFailure}
                />
              </div>
            )}
          </>
        )}

        {/* TAB 2: Wardrobe ML Scanner */}
        {activeTab === 'wardrobe' && (
          <WardrobeScanner userId={currentUser?.id || 'kunal_01'} />
        )}

        {/* TAB 3: Check-Before-Buy Duplicate Prevention */}
        {activeTab === 'check_buy' && (
          <CheckBeforeBuy userId={currentUser?.id || 'kunal_01'} />
        )}

        {/* TAB 4: Integrated Fit Passport */}
        {activeTab === 'passport' && (
          <IntegratedFitPassport
            userId={currentUser?.id || 'kunal_01'}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-400">
        FitLoop AI Platform · Model 1 (Anthropometric) & Wardrobe ML (Intelligence & Duplicate Prevention) · Integrated Prototype
      </footer>
    </div>
  );
}

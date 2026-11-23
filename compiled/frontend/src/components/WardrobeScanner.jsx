import React, { useState, useEffect } from 'react';
import { Shirt, Upload, CheckCircle, AlertCircle, RefreshCw, Palette, Layers, Eye } from 'lucide-react';
import { uploadWardrobeScan, fetchUserWardrobe } from '../api/wardrobeClient';

export default function WardrobeScanner({ userId }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanType, setScanType] = useState('closet');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [wardrobe, setWardrobe] = useState([]);
  const [isLoadingWardrobe, setIsLoadingWardrobe] = useState(false);
  const [error, setError] = useState(null);

  const loadWardrobe = async () => {
    if (!userId) return;
    setIsLoadingWardrobe(true);
    try {
      const data = await fetchUserWardrobe(userId);
      setWardrobe(data.wardrobe || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoadingWardrobe(false);
    }
  };

  useEffect(() => {
    loadWardrobe();
  }, [userId]);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setError(null);
    }
  };

  const handleScanSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a photo of your closet, flat-lay, or clothing rack.');
      return;
    }
    setIsScanning(true);
    setError(null);
    try {
      const res = await uploadWardrobeScan(file, userId, scanType);
      setScanResult(res);
      await loadWardrobe();
    } catch (err) {
      setError(err.message || 'Scanning failed');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Upload & Scan Section */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Shirt className="w-6 h-6 text-indigo-600" />
              Wardrobe Scanning & Garment Detection
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Powered by <strong>YOLO26n (5.4MB)</strong> garment bounds + <strong>CIE LAB K-Means</strong> color palette extraction + <strong>MobileNetV3</strong> 576-dim embeddings.
            </p>
          </div>
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
            Wardrobe ML Engine
          </span>
        </div>

        <form onSubmit={handleScanSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Dropzone */}
            <div className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center hover:border-indigo-500 transition-colors bg-slate-50 flex flex-col justify-center items-center relative">
              {preview ? (
                <div className="relative w-full h-56 flex items-center justify-center">
                  <img
                    src={preview}
                    alt="Scan Preview"
                    className="max-h-full max-w-full rounded-lg object-contain shadow-sm"
                  />
                  <button
                    type="button"
                    onClick={() => { setFile(null); setPreview(null); }}
                    className="absolute top-2 right-2 bg-slate-900/70 text-white rounded-full px-2 py-1 text-xs hover:bg-slate-900"
                  >
                    Change
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer flex flex-col items-center">
                  <Upload className="w-10 h-10 text-slate-400 mb-2" />
                  <span className="text-sm font-semibold text-slate-700">Upload Closet or Garment Photo</span>
                  <span className="text-xs text-slate-500 mt-1">Supports JPG, PNG, WebP</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>

            {/* Controls */}
            <div className="flex flex-col justify-between space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Scan Perspective / Layout</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'closet', label: 'Closet / Rack' },
                    { id: 'flat_lay', label: 'Flat-Lay' },
                    { id: 'single_garment', label: 'Single Item' },
                  ].map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setScanType(t.id)}
                      className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-all ${
                        scanType === t.id
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                          : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="mt-4 p-3 bg-indigo-50/60 rounded-lg text-xs text-indigo-900 space-y-1">
                  <p className="font-semibold flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-indigo-600" /> Auto-Extraction Pipeline:
                  </p>
                  <p>• YOLO26n crops each detected garment automatically.</p>
                  <p>• K-Means extracts core 60% dominant & secondary CIE LAB hex colors.</p>
                  <p>• MobileNetV3 calculates normalized unit vectors for visual pattern matching.</p>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isScanning || !file}
                className={`w-full py-3 px-4 rounded-xl font-semibold text-white shadow-md flex items-center justify-center gap-2 transition-all ${
                  isScanning || !file
                    ? 'bg-slate-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg active:scale-[0.99]'
                }`}
              >
                {isScanning ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Detecting Garments & Extracting Colors...
                  </>
                ) : (
                  <>
                    <Shirt className="w-4 h-4" />
                    Scan & Digitize into Wardrobe
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* Scan Results Feedback */}
        {scanResult && (
          <div className="mt-6 pt-6 border-t border-slate-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-600" />
                Detected {scanResult.detected_count} Garments in Scan
              </h3>
              <span className="text-xs text-slate-500 font-mono">Scan ID: {scanResult.scan_id}</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {scanResult.items.map((item, idx) => (
                <div key={idx} className="bg-slate-50 rounded-xl p-3 border border-slate-200 flex flex-col items-center text-center">
                  <div className="w-full h-32 bg-slate-200 rounded-lg overflow-hidden mb-2 flex items-center justify-center">
                    {item.crop_url ? (
                      <img
                        src={`http://127.0.0.1:8000${item.crop_url}`}
                        alt={item.category}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <Shirt className="w-8 h-8 text-slate-400" />
                    )}
                  </div>
                  <span className="font-semibold text-xs text-slate-900 capitalize">{item.category}</span>
                  <span className="text-[10px] text-slate-500">Conf: {Math.round(item.confidence * 100)}%</span>
                  
                  {/* Colors */}
                  <div className="flex items-center gap-1.5 mt-2">
                    <span
                      className="w-4 h-4 rounded-full border border-slate-300 shadow-xs"
                      style={{ backgroundColor: item.dominant_hex }}
                      title={`Dominant: ${item.dominant_hex}`}
                    />
                    <span
                      className="w-4 h-4 rounded-full border border-slate-300 shadow-xs"
                      style={{ backgroundColor: item.secondary_hex }}
                      title={`Secondary: ${item.secondary_hex}`}
                    />
                    <span className="text-[10px] text-slate-600 font-mono ml-1">{item.dominant_hex}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Digitized Wardrobe Gallery */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <Palette className="w-5 h-5 text-indigo-600" />
              Digitized Wardrobe Inventory ({wardrobe.length} Items)
            </h3>
            <p className="text-xs text-slate-500">Items persisted in SQLite multi-tenant database for this user.</p>
          </div>
          <button
            onClick={loadWardrobe}
            className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 font-semibold p-1.5 rounded-lg hover:bg-indigo-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoadingWardrobe ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {wardrobe.length === 0 ? (
          <div className="text-center py-12 text-slate-400 text-sm border-2 border-dashed border-slate-200 rounded-xl">
            <Shirt className="w-12 h-12 mx-auto mb-2 opacity-40 text-slate-400" />
            <p>Your digital wardrobe is currently empty.</p>
            <p className="text-xs text-slate-500 mt-1">Upload a closet photo above to populate your inventory.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {wardrobe.map((item) => (
              <div
                key={item.id}
                className="group relative bg-slate-50 hover:bg-indigo-50/40 rounded-xl p-2.5 border border-slate-200 hover:border-indigo-300 transition-all shadow-xs"
              >
                <div className="w-full h-28 bg-slate-200 rounded-lg overflow-hidden mb-2">
                  {item.crop_url ? (
                    <img
                      src={`http://127.0.0.1:8000${item.crop_url}`}
                      alt={item.category}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400">
                      <Shirt className="w-6 h-6" />
                    </div>
                  )}
                </div>
                <div className="text-xs font-semibold text-slate-800 truncate capitalize" title={item.category}>
                  {item.category}
                </div>
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-200">
                  <div className="flex items-center gap-1">
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-slate-300"
                      style={{ backgroundColor: item.dominant_hex }}
                      title={`Dominant: ${item.dominant_hex}`}
                    />
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-slate-300"
                      style={{ backgroundColor: item.secondary_hex }}
                      title={`Secondary: ${item.secondary_hex}`}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">{item.dominant_hex}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

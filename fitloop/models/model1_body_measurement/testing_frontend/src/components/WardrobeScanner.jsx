import React, { useState, useEffect, useRef } from 'react';
import { Shirt, Upload, CheckCircle, AlertCircle, RefreshCw, Search, X, Tag, Trash2, SlidersHorizontal, Loader2, Check } from 'lucide-react';
import { uploadWardrobeScan, fetchUserWardrobe, deleteWardrobeItem, updateWardrobeItem } from '../api/wardrobeClient';

const CATEGORIES = ['All', 'Tops', 'Outerwear', 'Trousers', 'Shorts', 'Dress'];

const EDITABLE_CATEGORIES = [
  'short sleeve top',
  'long sleeve top',
  'short sleeve outwear',
  'long sleeve outwear',
  'vest',
  'trousers',
  'shorts',
  'skirt',
  'dress',
];

function matchCategory(itemCat, filter) {
  if (filter === 'All') return true;
  return itemCat?.toLowerCase().includes(filter.toLowerCase());
}

function hexToFriendlyName(hex) {
  if (!hex) return 'Unknown';
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  const max = Math.max(r,g,b), min = Math.min(r,g,b);
  const l = (max+min)/2/255;
  if (l < 0.15) return 'Charcoal';
  if (l > 0.85) return 'White';
  const s = max===min ? 0 : (l<0.5 ? (max-min)/(max+min) : (max-min)/(2*255-max-min));
  if (s < 0.1) return l < 0.4 ? 'Dark Grey' : 'Light Grey';
  let hue = max===r ? ((g-b)/(max-min))%6 : max===g ? (b-r)/(max-min)+2 : (r-g)/(max-min)+4;
  hue = Math.round(hue * 60); if (hue < 0) hue += 360;
  if (hue < 20||hue>=345) return l<0.5?'Dark Red':'Rose';
  if (hue < 40) return l<0.5?'Brown':'Orange';
  if (hue < 65) return l<0.5?'Olive':'Yellow';
  if (hue < 155) return l<0.5?'Forest Green':'Green';
  if (hue < 185) return 'Teal';
  if (hue < 255) return l<0.5?'Navy':'Blue';
  if (hue < 290) return 'Purple';
  if (hue < 330) return 'Violet';
  return 'Pink';
}

const SAMPLE_DEMO_URL = 'https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=400&q=80';

const cardBg    = '#181410';
const subBg     = '#252018';   // subsection panels — warm sand-brown
const insetBg   = '#1C1810';   // deep inset (dropzone inner, scan items)
const pageBg    = '#0E0C07';
const border    = '#2A2218';
const subBorder = '#342A1C';   // subsection border — slightly warmer
const amber     = '#D97706';
const textMain  = '#EDE8DF';
const textMuted = '#7A6A55';
const textFaint = '#3A3020';

export default function WardrobeScanner({ userId }) {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [scanType, setScanType]     = useState('closet');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [wardrobe, setWardrobe]     = useState([]);
  const [isLoadingWardrobe, setIsLoadingWardrobe] = useState(false);
  const [error, setError]           = useState(null);
  const [catFilter, setCatFilter]   = useState('All');
  const [search, setSearch]         = useState('');
  const [showPipeline, setShowPipeline] = useState(false);

  // Edit & Delete modal states
  const [editingItem, setEditingItem]     = useState(null);
  const [editCategory, setEditCategory]   = useState('');
  const [editColor, setEditColor]         = useState('#000000');
  const [deletingItem, setDeletingItem]   = useState(null);
  const [isActionLoading, setIsActionLoading] = useState(false);
  const [toastMessage, setToastMessage]   = useState(null);

  const fileRef = useRef(null);

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
      setError('Please select a photo.');
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

  const handleSampleDemo = async () => {
    setError(null);
    try {
      const resp = await fetch(SAMPLE_DEMO_URL);
      const blob = await resp.blob();
      setFile(new File([blob], 'sample_closet.jpg', { type: 'image/jpeg' }));
      setPreview(SAMPLE_DEMO_URL);
    } catch {
      setError('Could not load sample image. Please upload manually.');
    }
  };

  // Trigger Edit Modal
  const handleStartEdit = (item) => {
    setEditingItem(item);
    setEditCategory(item.category || 'short sleeve top');
    setEditColor(item.dominant_hex || '#2B4C7E');
  };

  // Submit Edit
  const handleSaveEdit = async () => {
    if (!editingItem) return;
    setIsActionLoading(true);
    try {
      await updateWardrobeItem(userId, editingItem.id, {
        category: editCategory,
        colour_hex: editColor,
        colour_name: hexToFriendlyName(editColor),
      });
      setWardrobe((prev) =>
        prev.map((it) =>
          it.id === editingItem.id
            ? { ...it, category: editCategory, dominant_hex: editColor }
            : it
        )
      );
      setToastMessage(`Updated to "${editCategory}" successfully!`);
      setTimeout(() => setToastMessage(null), 3000);
      setEditingItem(null);
    } catch (err) {
      setError(err.message || 'Failed to update garment details.');
    } finally {
      setIsActionLoading(false);
    }
  };

  // Submit Delete
  const handleConfirmDelete = async () => {
    if (!deletingItem) return;
    setIsActionLoading(true);
    try {
      await deleteWardrobeItem(userId, deletingItem.id);
      setWardrobe((prev) => prev.filter((it) => it.id !== deletingItem.id));
      setToastMessage('Garment removed from your wardrobe.');
      setTimeout(() => setToastMessage(null), 3000);
      setDeletingItem(null);
    } catch (err) {
      setError(err.message || 'Failed to remove garment.');
    } finally {
      setIsActionLoading(false);
    }
  };

  const filteredWardrobe = wardrobe.filter((item) =>
    matchCategory(item.category, catFilter) &&
    (!search || item.category?.toLowerCase().includes(search.toLowerCase()) || item.dominant_hex?.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="space-y-5">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl border shadow-2xl text-xs font-semibold animate-fadeIn"
          style={{ backgroundColor: '#181410', borderColor: amber, color: textMain }}>
          <CheckCircle className="w-4 h-4" style={{ color: amber }} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Upload card */}
      <div className="rounded-2xl border p-6" style={{ backgroundColor: cardBg, borderColor: border }}>
        <div className="flex items-start justify-between mb-5">
          <div>
            <h2 className="text-lg font-bold flex items-center gap-2" style={{ color: textMain }}>
              <Shirt className="w-5 h-5" style={{ color: amber }} />
              Wardrobe Scanner
            </h2>
            <p className="text-xs mt-1" style={{ color: textMuted }}>
              Upload a closet photo, flat‑lay, or single garment shot — AI detects and digitizes each piece.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setShowPipeline(!showPipeline)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold rounded-lg border transition-all"
              style={{ backgroundColor: subBg, borderColor: border, color: textMuted }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = amber; e.currentTarget.style.color = amber; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = border; e.currentTarget.style.color = textMuted; }}>
              <SlidersHorizontal className="w-3 h-3" /> Pipeline Specs
            </button>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold border"
              style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: amber }}>
              Wardrobe ML
            </span>
          </div>
        </div>

        {showPipeline && (
          <div className="mb-5 p-4 rounded-xl border text-xs space-y-1.5" style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
            <p className="font-semibold mb-2" style={{ color: textMain }}>🔬 ML Pipeline Details</p>
            <p><span className="font-semibold" style={{ color: amber }}>1. Detection:</span> YOLO26n (5.4 MB) — garment bounding box extraction</p>
            <p><span className="font-semibold" style={{ color: amber }}>2. Color:</span> CIE LAB K‑Means on core 60% crop — dominant & secondary hex palette</p>
            <p><span className="font-semibold" style={{ color: amber }}>3. Embedding:</span> MobileNetV3 576‑dim normalized unit vector for visual similarity matching</p>
          </div>
        )}

        <form onSubmit={handleScanSubmit} className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Dropzone */}
            <div className={`border-2 border-dashed rounded-xl p-5 text-center flex flex-col justify-center items-center min-h-[200px] transition-colors`}
              style={{ backgroundColor: insetBg, borderColor: preview ? border : '#3A2E1A' }}>
              {preview ? (
                <div className="relative w-full h-48 flex items-center justify-center">
                  <img src={preview} alt="Scan Preview" className="max-h-full max-w-full rounded-xl object-contain" />
                  <button type="button" onClick={() => { setFile(null); setPreview(null); }}
                    className="absolute top-2 right-2 flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded-full border transition-colors"
                    style={{ backgroundColor: subBg, borderColor: border, color: textMuted }}>
                    <X className="w-3 h-3" /> Change
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer flex flex-col items-center w-full" onClick={() => fileRef.current?.click()}>
                  <input ref={fileRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                  <div className="w-12 h-12 rounded-2xl border flex items-center justify-center mb-3"
                    style={{ backgroundColor: subBg, borderColor: '#3A2E1A' }}>
                    <Upload className="w-5 h-5" style={{ color: amber }} />
                  </div>
                  <span className="text-sm font-semibold" style={{ color: textMain }}>Upload Closet or Garment Photo</span>
                  <span className="text-xs mt-1" style={{ color: textMuted }}>JPG, PNG, WebP</span>
                  <button type="button" onClick={(e) => { e.stopPropagation(); handleSampleDemo(); }}
                    className="mt-3 px-3 py-1.5 text-xs font-semibold rounded-lg border transition-colors"
                    style={{ backgroundColor: '#1F1A0F', borderColor: '#3A2E1A', color: amber }}>
                    ✦ Try Sample Closet Photo
                  </button>
                </label>
              )}
            </div>

            {/* Controls */}
            <div className="flex flex-col justify-between gap-4">
              <div>
                <label className="block text-xs font-semibold mb-2" style={{ color: textMuted }}>Scan Layout</label>
                <div className="grid grid-cols-3 gap-2">
                  {[{ id: 'closet', label: 'Closet / Rack' }, { id: 'flat_lay', label: 'Flat‑Lay' }, { id: 'single_garment', label: 'Single Item' }].map((t) => (
                    <button key={t.id} type="button" onClick={() => setScanType(t.id)}
                      className="px-2 py-2 text-xs font-semibold rounded-xl border transition-all"
                      style={scanType === t.id
                        ? { backgroundColor: amber, color: '#0E0C07', borderColor: amber }
                        : { backgroundColor: subBg, color: textMuted, borderColor: subBorder }}>
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-xl border text-xs flex items-center gap-2"
                  style={{ backgroundColor: '#1F1210', borderColor: '#4A2020', color: '#F87171' }}>
                  <AlertCircle className="w-4 h-4 shrink-0" /><span>{error}</span>
                </div>
              )}

              <button type="submit" disabled={isScanning || !file}
                className="w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all cursor-pointer"
                style={isScanning || !file
                  ? { backgroundColor: '#1F1A0F', color: textFaint, cursor: 'not-allowed' }
                  : { backgroundColor: amber, color: '#0E0C07' }}>
                {isScanning
                  ? <><RefreshCw className="w-4 h-4 animate-spin" />Detecting Garments…</>
                  : <><Shirt className="w-4 h-4" />Scan & Digitize into Wardrobe</>}
              </button>
            </div>
          </div>
        </form>

        {scanResult && (
          <div className="mt-5 pt-5 border-t" style={{ borderColor: border }}>
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle className="w-4 h-4" style={{ color: '#6EE7B7' }} />
              <h3 className="text-sm font-bold" style={{ color: textMain }}>
                Detected {scanResult.detected_count} Garment{scanResult.detected_count !== 1 ? 's' : ''}
              </h3>
              <span className="text-xs font-mono ml-auto" style={{ color: textFaint }}>ID: {String(scanResult.scan_id).slice(0,8)}</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {scanResult.items.map((item, idx) => (
                <div key={idx} className="rounded-xl p-3 border" style={{ backgroundColor: subBg, borderColor: subBorder }}>
                  <div className="w-full h-28 rounded-lg overflow-hidden mb-2" style={{ backgroundColor: '#0E0C07' }}>
                    {item.crop_url
                      ? <img src={`http://127.0.0.1:8000${item.crop_url}`} alt={item.category} className="w-full h-full object-cover" />
                      : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-7 h-7" style={{ color: textFaint }} /></div>}
                  </div>
                  <p className="text-xs font-semibold capitalize truncate" style={{ color: textMain }}>{item.category}</p>
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className="w-3.5 h-3.5 rounded-full border shrink-0" style={{ backgroundColor: item.dominant_hex, borderColor: '#3A3020' }} />
                    <span className="text-[10px] truncate" style={{ color: textMuted }}>{hexToFriendlyName(item.dominant_hex)}</span>
                    <span className="text-[10px] font-mono ml-auto" style={{ color: textFaint }}>{item.dominant_hex}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Wardrobe gallery */}
      <div className="rounded-2xl border p-6" style={{ backgroundColor: cardBg, borderColor: border }}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold" style={{ color: textMain }}>
            Your Wardrobe <span className="font-normal" style={{ color: textFaint }}>({wardrobe.length} items)</span>
          </h3>
          <button onClick={loadWardrobe}
            className="flex items-center gap-1.5 text-xs font-semibold p-1.5 rounded-lg transition-colors cursor-pointer"
            style={{ color: amber }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1F1A0F'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
            <RefreshCw className={`w-3.5 h-3.5 ${isLoadingWardrobe ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 flex-wrap mb-4">
          <div className="relative flex-1 min-w-[180px] max-w-xs">
            <Search className="absolute left-2.5 top-2 w-3.5 h-3.5 pointer-events-none" style={{ color: textFaint }} />
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search…"
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-xl border focus:outline-none transition-all"
              style={{ backgroundColor: pageBg, borderColor: border, color: textMain }}
              onFocus={(e) => e.target.style.borderColor = amber}
              onBlur={(e)  => e.target.style.borderColor = border}
            />
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {CATEGORIES.map((cat) => (
              <button key={cat} onClick={() => setCatFilter(cat)}
                className="px-3 py-1 text-xs font-semibold rounded-full border transition-all cursor-pointer"
                style={catFilter === cat
                  ? { backgroundColor: amber, color: '#0E0C07', borderColor: amber }
                  : { backgroundColor: subBg, color: textMuted, borderColor: subBorder }}>
                {cat}
              </button>
            ))}
          </div>
        </div>

        {wardrobe.length === 0 ? (
          <div className="text-center py-16 border-2 border-dashed rounded-2xl" style={{ borderColor: '#2A2218', color: textFaint }}>
            <Shirt className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-sm font-medium" style={{ color: textMuted }}>Your digital wardrobe is empty</p>
            <p className="text-xs mt-1" style={{ color: textFaint }}>Upload a closet photo above to populate your inventory.</p>
          </div>
        ) : filteredWardrobe.length === 0 ? (
          <div className="text-center py-10 text-sm" style={{ color: textMuted }}>No items match your search.</div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {filteredWardrobe.map((item) => (
              <div key={item.id}
                className="group relative rounded-xl border overflow-hidden transition-all duration-200 hover:-translate-y-0.5"
                style={{ backgroundColor: subBg, borderColor: subBorder }}>
                <div className="w-full h-28 overflow-hidden" style={{ backgroundColor: pageBg }}>
                  {item.crop_url
                    ? <img src={`http://127.0.0.1:8000${item.crop_url}`} alt={item.category} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                    : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-6 h-6" style={{ color: textFaint }} /></div>}
                </div>
                <div className="p-2.5">
                  <p className="text-xs font-semibold truncate capitalize" style={{ color: textMain }}>{item.category}</p>
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className="w-3 h-3 rounded-full border shrink-0" style={{ backgroundColor: item.dominant_hex, borderColor: '#3A3020' }} />
                    <span className="text-[10px] truncate" style={{ color: textMuted }}>{hexToFriendlyName(item.dominant_hex)}</span>
                  </div>
                  <p className="text-[9px] font-mono mt-0.5" style={{ color: textFaint }}>{item.dominant_hex}</p>
                </div>

                {/* Quick actions on hover */}
                <div className="absolute inset-x-0 bottom-0 flex items-center gap-1.5 p-2 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(to top, rgba(24, 20, 16, 0.95) 75%, transparent)` }}>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); handleStartEdit(item); }}
                    className="flex-1 flex items-center justify-center gap-1 py-1.5 text-[10px] font-semibold rounded-lg transition-colors cursor-pointer border hover:brightness-125"
                    style={{ backgroundColor: '#252018', borderColor: '#3A2E1A', color: amber }}>
                    <Tag className="w-2.5 h-2.5" /> Edit
                  </button>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setDeletingItem(item); }}
                    className="flex-1 flex items-center justify-center gap-1 py-1.5 text-[10px] font-semibold rounded-lg transition-colors cursor-pointer border hover:brightness-125"
                    style={{ backgroundColor: '#251616', borderColor: '#4A2020', color: '#F87171' }}>
                    <Trash2 className="w-2.5 h-2.5" /> Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Edit Garment Modal */}
      {editingItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: 'rgba(10, 8, 5, 0.85)', backdropFilter: 'blur(4px)' }}
          onClick={() => setEditingItem(null)}>
          <div className="w-full max-w-md rounded-2xl border p-6 space-y-5 animate-fadeIn"
            style={{ backgroundColor: cardBg, borderColor: subBorder }}
            onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between pb-3 border-b" style={{ borderColor: border }}>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: '#1F1A0F' }}>
                  <Tag className="w-4 h-4" style={{ color: amber }} />
                </div>
                <h3 className="text-sm font-bold" style={{ color: textMain }}>Edit Garment</h3>
              </div>
              <button onClick={() => setEditingItem(null)}
                className="p-1 rounded-lg text-xs transition-colors hover:brightness-125"
                style={{ color: textMuted }}>
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Garment preview */}
            <div className="flex items-center gap-4 p-3 rounded-xl border" style={{ backgroundColor: subBg, borderColor: subBorder }}>
              <div className="w-16 h-16 rounded-lg overflow-hidden shrink-0 border" style={{ backgroundColor: pageBg, borderColor: border }}>
                {editingItem.crop_url
                  ? <img src={`http://127.0.0.1:8000${editingItem.crop_url}`} alt="Crop preview" className="w-full h-full object-cover" />
                  : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-5 h-5" style={{ color: textFaint }} /></div>}
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-[10px] uppercase font-bold tracking-wider" style={{ color: amber }}>Currently</span>
                <p className="text-xs font-semibold capitalize truncate" style={{ color: textMain }}>{editingItem.category}</p>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className="w-3 h-3 rounded-full border shrink-0" style={{ backgroundColor: editingItem.dominant_hex, borderColor: '#3A3020' }} />
                  <span className="text-[11px]" style={{ color: textMuted }}>{hexToFriendlyName(editingItem.dominant_hex)}</span>
                  <span className="text-[10px] font-mono ml-auto" style={{ color: textFaint }}>{editingItem.dominant_hex}</span>
                </div>
              </div>
            </div>

            {/* Select category */}
            <div>
              <label className="block text-xs font-semibold mb-2" style={{ color: textMuted }}>
                Category
              </label>
              <div className="grid grid-cols-3 gap-1.5 max-h-36 overflow-y-auto p-1 rounded-xl border"
                style={{ backgroundColor: insetBg, borderColor: border }}>
                {EDITABLE_CATEGORIES.map((cat) => (
                  <button key={cat} type="button" onClick={() => setEditCategory(cat)}
                    className="px-2 py-1.5 text-[11px] font-semibold rounded-lg capitalize border transition-all text-left truncate"
                    style={editCategory === cat
                      ? { backgroundColor: amber, color: '#0E0C07', borderColor: amber }
                      : { backgroundColor: subBg, color: textMuted, borderColor: subBorder }}>
                    {cat}
                  </button>
                ))}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-[11px]" style={{ color: textFaint }}>Custom:</span>
                <input
                  type="text"
                  value={editCategory}
                  onChange={(e) => setEditCategory(e.target.value)}
                  className="flex-1 px-2.5 py-1 text-xs rounded-lg border focus:outline-none"
                  style={{ backgroundColor: insetBg, borderColor: border, color: textMain }}
                  placeholder="e.g. blazer, hoodie..."
                />
              </div>
            </div>

            {/* Dominant Color */}
            <div>
              <label className="block text-xs font-semibold mb-2" style={{ color: textMuted }}>
                Dominant Color
              </label>
              <div className="flex items-center gap-3 p-2 rounded-xl border" style={{ backgroundColor: insetBg, borderColor: border }}>
                <input
                  type="color"
                  value={editColor}
                  onChange={(e) => setEditColor(e.target.value)}
                  className="w-8 h-8 rounded cursor-pointer bg-transparent border-0"
                />
                <input
                  type="text"
                  value={editColor}
                  onChange={(e) => setEditColor(e.target.value)}
                  className="px-2.5 py-1 text-xs font-mono rounded-lg border focus:outline-none w-28 uppercase"
                  style={{ backgroundColor: subBg, borderColor: border, color: textMain }}
                />
                <span className="text-xs font-medium" style={{ color: textMuted }}>
                  {hexToFriendlyName(editColor)}
                </span>
              </div>
            </div>

            {/* Modal actions */}
            <div className="flex items-center gap-2 pt-2">
              <button
                type="button"
                onClick={() => setEditingItem(null)}
                className="flex-1 py-2.5 px-3 rounded-xl text-xs font-semibold border transition-colors cursor-pointer"
                style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
                Cancel
              </button>
              <button
                type="button"
                disabled={isActionLoading || !editCategory}
                onClick={handleSaveEdit}
                className="flex-1 py-2.5 px-3 rounded-xl text-xs font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                style={isActionLoading || !editCategory
                  ? { backgroundColor: '#1F1A0F', color: textFaint, cursor: 'not-allowed' }
                  : { backgroundColor: amber, color: '#0E0C07' }}>
                {isActionLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Remove Confirmation Modal */}
      {deletingItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: 'rgba(10, 8, 5, 0.85)', backdropFilter: 'blur(4px)' }}
          onClick={() => setDeletingItem(null)}>
          <div className="w-full max-w-sm rounded-2xl border p-6 space-y-4 animate-fadeIn"
            style={{ backgroundColor: cardBg, borderColor: '#4A2020' }}
            onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                style={{ backgroundColor: '#2A1414', border: '1px solid #4A2020' }}>
                <Trash2 className="w-5 h-5" style={{ color: '#F87171' }} />
              </div>
              <div>
                <h3 className="text-sm font-bold" style={{ color: textMain }}>Remove Garment?</h3>
                <p className="text-[11px]" style={{ color: textMuted }}>This item will be permanently removed.</p>
              </div>
            </div>

            {/* Preview thumbnail */}
            <div className="flex items-center gap-3 p-3 rounded-xl border" style={{ backgroundColor: subBg, borderColor: subBorder }}>
              <div className="w-12 h-12 rounded-lg overflow-hidden shrink-0 border" style={{ backgroundColor: pageBg, borderColor: border }}>
                {deletingItem.crop_url
                  ? <img src={`http://127.0.0.1:8000${deletingItem.crop_url}`} alt="Crop preview" className="w-full h-full object-cover" />
                  : <div className="w-full h-full flex items-center justify-center"><Shirt className="w-4 h-4" style={{ color: textFaint }} /></div>}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold capitalize truncate" style={{ color: textMain }}>{deletingItem.category}</p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: deletingItem.dominant_hex }} />
                  <span className="text-[10px]" style={{ color: textMuted }}>{hexToFriendlyName(deletingItem.dominant_hex)}</span>
                </div>
              </div>
            </div>

            <p className="text-xs leading-relaxed" style={{ color: '#A08870' }}>
              Are you sure you want to delete this piece from your digital wardrobe inventory?
            </p>

            <div className="flex items-center gap-2 pt-2">
              <button
                type="button"
                onClick={() => setDeletingItem(null)}
                className="flex-1 py-2.5 px-3 rounded-xl text-xs font-semibold border transition-colors cursor-pointer"
                style={{ backgroundColor: subBg, borderColor: subBorder, color: textMuted }}>
                Cancel
              </button>
              <button
                type="button"
                disabled={isActionLoading}
                onClick={handleConfirmDelete}
                className="flex-1 py-2.5 px-3 rounded-xl text-xs font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                style={{ backgroundColor: '#EF4444', color: '#FFFFFF' }}>
                {isActionLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                Remove Item
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

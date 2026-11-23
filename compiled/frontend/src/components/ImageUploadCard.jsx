import React, { useRef } from 'react';
import { Upload, X, RefreshCw, AlertCircle } from 'lucide-react';

export default function ImageUploadCard({
  viewType, // 'front' | 'side'
  title,
  description,
  file,
  previewUrl,
  onFileSelect,
  onFileRemove,
  error,
}) {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={`relative flex flex-col rounded-xl border bg-white p-5 shadow-sm transition-all ${
        error
          ? 'border-red-400 ring-1 ring-red-400'
          : previewUrl
          ? 'border-slate-300'
          : 'border-slate-200 hover:border-slate-300'
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="inline-block text-[11px] font-semibold tracking-wider uppercase text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
            {viewType} Perspective
          </span>
          <h3 className="text-sm font-semibold text-slate-800 mt-1">{title}</h3>
        </div>
        {previewUrl && (
          <span className="text-xs text-slate-400 font-mono">
            {(file?.size ? (file.size / 1024).toFixed(0) + ' KB' : '')}
          </span>
        )}
      </div>

      <p className="text-xs text-slate-500 mb-4">{description}</p>

      {/* Hidden native input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleInputChange}
        className="hidden"
        data-testid={`file-input-${viewType}`}
      />

      {previewUrl ? (
        <div className="relative group w-full h-72 rounded-lg bg-slate-900/5 overflow-hidden flex items-center justify-center border border-slate-200">
          <img
            src={previewUrl}
            alt={`${viewType} preview`}
            className="w-full h-full object-contain"
          />

          {/* Action Overlay */}
          <div className="absolute inset-0 bg-slate-900/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/90 hover:bg-white text-slate-800 rounded-lg text-xs font-medium shadow-sm transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Replace
            </button>
            <button
              type="button"
              onClick={onFileRemove}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-medium shadow-sm transition-colors"
            >
              <X className="w-3.5 h-3.5" />
              Remove
            </button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="w-full h-72 rounded-lg border-2 border-dashed border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-blue-50/20 cursor-pointer flex flex-col items-center justify-center p-6 text-center transition-colors"
          data-testid={`dropzone-${viewType}`}
        >
          <div className="w-12 h-12 rounded-full bg-white shadow-sm flex items-center justify-center text-slate-500 mb-3 group-hover:text-blue-600">
            <Upload className="w-5 h-5" />
          </div>
          <p className="text-sm font-medium text-slate-700">
            Click to upload or drag & drop
          </p>
          <p className="text-xs text-slate-400 mt-1">
            JPG, PNG, or WebP (head-to-toe unclipped)
          </p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-1.5 mt-3 text-xs text-red-600">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

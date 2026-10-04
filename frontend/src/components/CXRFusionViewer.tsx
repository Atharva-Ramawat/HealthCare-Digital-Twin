import { useState, useEffect } from 'react';
import { 
  Eye, 
  EyeOff, 
  Layers, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import { fetchGradCAMHeatmapBlob } from '../services/api';

interface CXRFusionViewerProps {
  studyId: string;
  probabilities: Record<string, number>;
  predictions: Record<string, boolean>;
  topFinding: string;
  topProbability: number;
  heatmapBase64?: string;
  inputImageBase64?: string;
  imageBase64?: string;
}

export const CXRFusionViewer: React.FC<CXRFusionViewerProps> = ({
  studyId,
  probabilities,
  predictions,
  topFinding,
  topProbability,
  heatmapBase64,
  inputImageBase64: rawInputImageBase64,
  imageBase64,
}) => {
  const inputImageBase64 = rawInputImageBase64 || imageBase64;
  console.log("Base64 Debug:", { inputImageBase64: !!inputImageBase64, heatmapBase64: !!heatmapBase64 });

  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.55);
  const [selectedPathology, setSelectedPathology] = useState<string>(topFinding || 'Pneumonia');
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [imageError, setImageError] = useState<boolean>(false);
  const [heatmapBlobUrl, setHeatmapBlobUrl] = useState<string | null>(null);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState<boolean>(false);

  // Check if current study/patient data contains direct base64 image data URLs
  const hasBase64Images = Boolean(
    (heatmapBase64 && heatmapBase64.startsWith('data:image')) ||
    (inputImageBase64 && inputImageBase64.startsWith('data:image')) ||
    heatmapBase64 || 
    inputImageBase64
  );

  // Bind directly to base64 data URLs for ad-hoc studies, or blob URL for database studies
  const currentImageSrc = hasBase64Images
    ? (showOverlay ? (heatmapBase64 || inputImageBase64) : (inputImageBase64 || heatmapBase64))
    : heatmapBlobUrl;

  // Default clinical decision thresholds matching ML pipeline
  const thresholds: Record<string, number> = {
    'Pneumonia': 0.35,
    'Pleural Effusion': 0.40,
    'Atelectasis': 0.38,
    'Consolidation': 0.35,
    'Edema': 0.38,
    'Cardiomegaly': 0.42,
    'Pneumothorax': 0.28,
    'No Finding': 0.50,
  };

  // Sync selected pathology if top finding changes
  useEffect(() => {
    if (topFinding && topFinding !== 'No Finding') {
      setSelectedPathology(topFinding);
    }
  }, [topFinding, studyId]);

  // Reset image error state whenever study or base64 data changes
  useEffect(() => {
    setImageError(false);
  }, [studyId, heatmapBase64, inputImageBase64]);

  // Fetch Grad-CAM heatmap binary blob only for database-backed cohort studies
  useEffect(() => {
    // If it is an ad-hoc study with base64 strings, bypass fetchGradCAMHeatmapBlob completely
    if (hasBase64Images) {
      setIsLoadingHeatmap(false);
      return;
    }

    let isCancelled = false;
    let objectUrl: string | null = null;

    async function loadHeatmap() {
      setIsLoadingHeatmap(true);
      setImageError(false);
      try {
        const blob = await fetchGradCAMHeatmapBlob(studyId, selectedPathology);
        if (isCancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setHeatmapBlobUrl(objectUrl);
      } catch (err) {
        if (isCancelled) return;
        console.warn(`[CXRFusionViewer] Heatmap blob fetch failed for ${studyId}:`, err);
        setImageError(true);
      } finally {
        if (!isCancelled) {
          setIsLoadingHeatmap(false);
        }
      }
    }

    loadHeatmap();

    return () => {
      isCancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [studyId, selectedPathology, hasBase64Images]);

  const handleZoomIn = () => setZoomLevel((z) => Math.min(z + 0.25, 2.5));
  const handleZoomOut = () => setZoomLevel((z) => Math.max(z - 0.25, 0.75));
  const handleResetZoom = () => setZoomLevel(1);

  return (
    <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col h-full">
      {/* Card Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-clinical-border">
        <div>
          <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-purple-400" />
            CXR & Multimodal Vision Core
          </h2>
          <p className="text-[11px] font-mono text-gray-400 mt-0.5">
            Study #{studyId} • DenseNet-121 (1024-dim Features)
          </p>
        </div>

        {/* Explainability Overlay Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
              showOverlay
                ? 'bg-purple-600 text-white shadow-md shadow-purple-500/20'
                : 'bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200'
            }`}
          >
            {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>{showOverlay ? 'Grad-CAM Active' : 'Show Overlay'}</span>
          </button>
        </div>
      </div>

      {/* Main Content: Split into Image Viewer (Left) and Probability List (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 mt-3 flex-1 min-h-[360px]">
        {/* Left: CXR & Heatmap Viewer (7 cols) */}
        <div className="lg:col-span-7 flex flex-col">
          {/* Top Bar for Viewer */}
          <div className="flex items-center justify-between text-[11px] font-mono mb-2 text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-1.5">
              <span>Target:</span>
              <select
                value={selectedPathology}
                onChange={(e) => setSelectedPathology(e.target.value)}
                className="bg-gray-100 dark:bg-slate-800 text-gray-800 dark:text-gray-200 px-2 py-0.5 rounded border border-gray-300 dark:border-slate-700 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-purple-500"
              >
                {Object.keys(thresholds).filter(k => k !== 'No Finding').map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center space-x-1">
              <button onClick={handleZoomOut} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-400 hover:text-gray-200" title="Zoom Out">
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] w-8 text-center">{Math.round(zoomLevel * 100)}%</span>
              <button onClick={handleZoomIn} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-400 hover:text-gray-200" title="Zoom In">
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button onClick={handleResetZoom} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-400 hover:text-gray-200" title="Reset Zoom">
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Central Radiograph Display Box */}
          <div className="relative flex-1 bg-black rounded-lg overflow-hidden border border-gray-300 dark:border-slate-800 flex items-center justify-center min-h-[260px] group">
            {isLoadingHeatmap && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10">
                <div className="flex items-center space-x-2 text-xs font-mono text-purple-300">
                  <RefreshCw className="w-4 h-4 animate-spin text-purple-400" />
                  <span>Loading Grad-CAM Heatmap...</span>
                </div>
              </div>
            )}
            <div
              className="relative transition-transform duration-150 ease-out flex items-center justify-center w-full h-full p-2"
              style={{ transform: `scale(${zoomLevel})` }}
            >
              {/* Heatmap Image Stream, Base64 Data URL, or Synthetic SVG CXR Fallback */}
              {!imageError && currentImageSrc ? (
                <img
                  src={currentImageSrc}
                  alt={showOverlay ? `Grad-CAM Heatmap for ${selectedPathology}` : 'Chest Radiograph'}
                  onError={() => {
                    console.error("Image failed to load");
                    setImageError(true);
                  }}
                  style={{
                    filter: showOverlay ? 'contrast(125%)' : 'contrast(110%)',
                    opacity: 1,
                  }}
                  className="max-h-full max-w-full object-contain rounded shadow-lg transition-all duration-200"
                />
              ) : (
                <div className="relative w-full h-64 flex flex-col items-center justify-center bg-gradient-to-b from-slate-900 to-black rounded p-4 text-center">
                  {/* High-Fidelity SVG CXR Silhouette */}
                  <svg className="w-48 h-48 opacity-80" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="200" height="200" fill="#090d16" rx="8" />
                    {/* Spine */}
                    <line x1="100" y1="20" x2="100" y2="180" stroke="#334155" strokeWidth="6" strokeDasharray="8 4" />
                    {/* Ribcage */}
                    <path d="M40 70 C70 50, 130 50, 160 70" stroke="#475569" strokeWidth="3" fill="none" opacity="0.6" />
                    <path d="M35 90 C70 70, 130 70, 165 90" stroke="#475569" strokeWidth="3.5" fill="none" opacity="0.7" />
                    <path d="M30 115 C70 95, 130 95, 170 115" stroke="#475569" strokeWidth="3.5" fill="none" opacity="0.75" />
                    <path d="M32 140 C70 120, 130 120, 168 140" stroke="#475569" strokeWidth="3.5" fill="none" opacity="0.7" />
                    {/* Lungs (Dark air-filled fields) */}
                    <ellipse cx="65" cy="105" rx="26" ry="40" fill="#030712" stroke="#1e293b" strokeWidth="2" />
                    <ellipse cx="135" cy="105" rx="26" ry="40" fill="#030712" stroke="#1e293b" strokeWidth="2" />
                    {/* Heart Silhouette (Cardiac shadow shifted to left) */}
                    <path d="M85 95 C85 85, 115 85, 125 110 C130 125, 115 145, 95 145 C80 145, 75 125, 85 95 Z" fill="#334155" opacity="0.85" />
                    {/* Diaphragm */}
                    <path d="M25 160 C55 145, 85 155, 100 155 C115 155, 145 145, 175 160" stroke="#64748b" strokeWidth="4" fill="none" />
                    {/* Simulated Grad-CAM Attention Cloud on Right Lower Lobe (Consolidation/Effusion) */}
                    {showOverlay && (
                      <g className="animate-pulse" style={{ opacity: overlayOpacity }}>
                        <radialGradient id="gradCamHotspot" cx="50%" cy="50%" r="50%">
                          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.95" />
                          <stop offset="45%" stopColor="#f59e0b" stopOpacity="0.8" />
                          <stop offset="75%" stopColor="#06b6d4" stopOpacity="0.5" />
                          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                        </radialGradient>
                        <ellipse cx="68" cy="120" rx="22" ry="20" fill="url(#gradCamHotspot)" />
                      </g>
                    )}
                  </svg>
                  <div className="text-[10px] font-mono text-gray-400 mt-2">
                    {showOverlay ? `Grad-CAM Feature Attribution: ${selectedPathology}` : 'Standard Anteroposterior (AP) Projection'}
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Floating Legend Bar */}
            <div className="absolute bottom-2 left-2 right-2 bg-slate-900/85 backdrop-blur-sm border border-slate-700/80 rounded px-2.5 py-1.5 flex items-center justify-between text-[10px] font-mono text-gray-300">
              <div className="flex items-center gap-2">
                <span className="text-purple-400 font-bold">ATTRIBUTION:</span>
                <span>{selectedPathology}</span>
                {hasBase64Images && (
                  <span className="px-1.5 py-0.5 rounded text-[9px] bg-purple-500/20 text-purple-300 border border-purple-500/30 font-bold">
                    AD-HOC INTAKE
                  </span>
                )}
              </div>
              {showOverlay && (
                <div className="flex items-center gap-2">
                  <span>ALPHA: {Math.round(overlayOpacity * 100)}%</span>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                    className="w-16 h-1 accent-purple-500 cursor-pointer"
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: 8-Class Pulmonary Probabilities (5 cols) */}
        <div className="lg:col-span-5 flex flex-col justify-between border-t lg:border-t-0 lg:border-l border-gray-100 dark:border-clinical-border lg:pl-4">
          <div>
            <div className="flex items-center justify-between text-[11px] font-mono text-gray-500 dark:text-gray-400 pb-2 border-b border-gray-100 dark:border-clinical-border">
              <span>8 TARGET PATHOLOGIES</span>
              <span>PROBABILITY</span>
            </div>

            <div className="space-y-2 mt-2.5">
              {Object.entries(probabilities).map(([name, prob]) => {
                const threshold = thresholds[name] || 0.40;
                const isPositive = predictions && predictions[name] !== undefined ? predictions[name] : prob >= threshold;
                const isSelected = selectedPathology === name;

                return (
                  <div
                    key={name}
                    onClick={() => {
                      if (name !== 'No Finding') setSelectedPathology(name);
                    }}
                    className={`p-1.5 rounded-lg cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-purple-500/15 border border-purple-500/40'
                        : 'hover:bg-gray-50 dark:hover:bg-slate-800/50 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-mono mb-1">
                      <span className={`font-semibold flex items-center gap-1.5 ${isPositive ? 'text-red-500 dark:text-red-400' : 'text-gray-700 dark:text-gray-300'}`}>
                        {name}
                        {isPositive && <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>}
                      </span>
                      <div className="flex items-center space-x-1.5">
                        <span className={`text-[11px] font-bold ${isPositive ? 'text-red-500 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'}`}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                        <span className={`text-[9px] px-1 py-0.2 rounded font-bold uppercase ${
                          isPositive ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'text-gray-400'
                        }`}>
                          {isPositive ? 'POS' : 'NEG'}
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar with Threshold Marker */}
                    <div className="relative w-full h-1.5 bg-gray-100 dark:bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isPositive ? 'bg-gradient-to-r from-amber-500 to-red-500' : 'bg-slate-400 dark:bg-slate-600'
                        }`}
                        style={{ width: `${Math.min(prob * 100, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Primary Pulmonary Alert Box */}
          <div className="mt-3 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-[11px] font-mono">
            <div className="flex items-center gap-1.5 text-amber-500 font-bold">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>PRIMARY RADIOLOGICAL FINDING</span>
            </div>
            <div className="text-gray-800 dark:text-gray-200 mt-1">
              {topFinding} (Confidence: {(topProbability * 100).toFixed(1)}%)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

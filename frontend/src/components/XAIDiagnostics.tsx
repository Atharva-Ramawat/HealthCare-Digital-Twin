import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import {
  BrainCircuit,
  Eye,
  Sliders,
  Sparkles,
} from 'lucide-react';
import type { ShapFeatureImportance, GradCAMSettings } from '../types/digitalTwin';

interface XAIDiagnosticsProps {
  isDarkMode?: boolean;
}

// Mock SHAP Attribution Data
const MOCK_SHAP_FEATURES: ShapFeatureImportance[] = [
  {
    feature: 'SpO2 Desaturation (<90%)',
    category: 'vitals',
    impactPercentage: 16.4,
    description: 'Profound impairment in arterial oxygenation drives acute deterioration scoring.',
    baselineValue: '86% current',
  },
  {
    feature: 'CXR Consolidation (Layer 16)',
    category: 'imaging',
    impactPercentage: 14.8,
    description: 'Dense alveolar opacification detected in right middle and lower lobes.',
    baselineValue: 'Prob: 0.84',
  },
  {
    feature: 'Tachypnea (RR > 28 bpm)',
    category: 'vitals',
    impactPercentage: 11.2,
    description: 'Compensatory hyperventilation indicative of respiratory muscle load.',
    baselineValue: '31 bpm current',
  },
  {
    feature: 'Tachycardia (HR > 115 bpm)',
    category: 'vitals',
    impactPercentage: 9.6,
    description: 'Elevated cardiac output attempting to compensate for systemic hypoxemia.',
    baselineValue: '122 bpm current',
  },
  {
    feature: 'CXR Pleural Effusion',
    category: 'imaging',
    impactPercentage: 7.5,
    description: 'Fluid blunting costophrenic angles with partial basilar compressive atelectasis.',
    baselineValue: 'Prob: 0.62',
  },
  {
    feature: 'Hypotension (SBP < 90 mmHg)',
    category: 'vitals',
    impactPercentage: 5.2,
    description: 'Early capillary leak or low systemic vascular resistance.',
    baselineValue: '88 mmHg',
  },
  {
    feature: 'Age > 65 Years',
    category: 'history',
    impactPercentage: 3.1,
    description: 'Reduced baseline pulmonary physiological reserve.',
    baselineValue: '68 yrs',
  },
  {
    feature: 'Adequate Urine Output (>0.8 ml/kg/h)',
    category: 'vitals',
    impactPercentage: -4.5,
    description: 'Maintained renal perfusion mitigating immediate multi-organ failure risk.',
    baselineValue: '1.1 ml/kg/h',
  },
  {
    feature: 'Normal Glasgow Coma Scale (15)',
    category: 'vitals',
    impactPercentage: -6.2,
    description: 'Intact cerebral perfusion and neurological response blunts acute mortality score.',
    baselineValue: '15 / 15',
  },
];

// Sample Pathologies with Hotspot coordinates on the X-ray
const PATHOLOGY_HOTSPOTS: Record<
  string,
  { cx: number; cy: number; rx: number; ry: number; prob: number; description: string }
> = {
  Consolidation: {
    cx: 58,
    cy: 52,
    rx: 24,
    ry: 20,
    prob: 0.84,
    description: 'Focal air-space opacification with air bronchograms in right lower lung zone.',
  },
  'Pleural Effusion': {
    cx: 72,
    cy: 78,
    rx: 22,
    ry: 16,
    prob: 0.62,
    description: 'Homogeneous dependent opacity with meniscus sign obliterating right costophrenic angle.',
  },
  Pneumothorax: {
    cx: 30,
    cy: 28,
    rx: 18,
    ry: 22,
    prob: 0.12,
    description: 'Visceral pleural line with peripheral lucency void of bronchovascular markings.',
  },
  Edema: {
    cx: 50,
    cy: 48,
    rx: 36,
    ry: 30,
    prob: 0.58,
    description: 'Bilateral perihilar batwing opacities and cephalization of pulmonary vasculature.',
  },
  Cardiomegaly: {
    cx: 48,
    cy: 64,
    rx: 28,
    ry: 24,
    prob: 0.44,
    description: 'Cardiothoracic ratio exceeding 0.50 on standard posteroanterior projection.',
  },
};

export const XAIDiagnostics: React.FC<XAIDiagnosticsProps> = ({ isDarkMode = true }) => {
  const [settings, setSettings] = useState<GradCAMSettings>({
    selectedPathology: 'Consolidation',
    threshold: 0.35,
    opacity: 0.7,
    palette: 'jet',
  });

  const activeHotspot = PATHOLOGY_HOTSPOTS[settings.selectedPathology] || PATHOLOGY_HOTSPOTS.Consolidation;

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto bg-gray-50 dark:bg-clinical-dark p-4 md:p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-gray-200 dark:border-clinical-border gap-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-purple-500/10 text-purple-500 border border-purple-500/30">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-mono tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-2">
              Explainable AI (XAI) & Saliency Diagnostics
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/15 text-purple-400 border border-purple-500/30">
                SHAP & Grad-CAM
              </span>
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono mt-0.5">
              Multimodal attribution breakdown: Visual Gradient-weighted Class Activation Mapping & SHAP feature contributions.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-2.5 py-1 rounded bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-slate-700">
            Target Layer: <span className="text-purple-400 font-bold">denseblock4.denselayer16.conv2</span>
          </span>
        </div>
      </div>

      {/* Main Grid: SHAP Plot on Left, Grad-CAM Interactive Thresholding on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* LEFT COLUMN: SHAP Attribution Bar Chart (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <div>
                <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-500" />
                  SHAP Summary Plot (Feature Risk Impact)
                </h3>
                <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Marginal contribution of clinical variables to aggregate ICU deterioration score
                </p>
              </div>

              <div className="flex items-center gap-3 text-[10px] font-mono">
                <span className="flex items-center gap-1 text-red-400">
                  <span className="w-2 h-2 rounded-full bg-red-500"></span> Escalates Risk
                </span>
                <span className="flex items-center gap-1 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Mitigates Risk
                </span>
              </div>
            </div>

            {/* Recharts Horizontal Bar Chart */}
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={MOCK_SHAP_FEATURES}
                  margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#1e293b' : '#e2e8f0'} horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[-10, 20]}
                    unit="%"
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <YAxis
                    dataKey="feature"
                    type="category"
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 10, fontFamily: 'monospace' }}
                    width={135}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDarkMode ? '#0f172a' : '#ffffff',
                      borderColor: isDarkMode ? '#334155' : '#cbd5e1',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      borderRadius: '8px',
                    }}
                    formatter={(val: any, _name: any, item: any) => [
                      `${val > 0 ? `+${val}%` : `${val}%`}`,
                      item.payload.description,
                    ]}
                  />
                  <ReferenceLine x={0} stroke={isDarkMode ? '#475569' : '#94a3b8'} strokeWidth={1.5} />
                  <Bar dataKey="impactPercentage" name="Risk Impact (%)" radius={[0, 4, 4, 0]}>
                    {MOCK_SHAP_FEATURES.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.impactPercentage >= 0 ? (entry.impactPercentage > 10 ? '#ef4444' : '#f59e0b') : '#10b981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Top Findings Table */}
            <div className="border border-gray-200 dark:border-slate-800 rounded-lg overflow-hidden text-xs font-mono">
              <table className="w-full text-left">
                <thead className="bg-gray-100 dark:bg-slate-900/80 text-[10px] text-gray-400 uppercase">
                  <tr>
                    <th className="p-2">Feature</th>
                    <th className="p-2">Category</th>
                    <th className="p-2">Observed</th>
                    <th className="p-2 text-right">Attribution</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-slate-800 text-[11px]">
                  {MOCK_SHAP_FEATURES.slice(0, 4).map((f, i) => (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                      <td className="p-2 font-bold text-gray-800 dark:text-gray-200">{f.feature}</td>
                      <td className="p-2">
                        <span className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-slate-800 text-[10px] text-gray-500 dark:text-gray-400 uppercase">
                          {f.category}
                        </span>
                      </td>
                      <td className="p-2 text-gray-400">{f.baselineValue}</td>
                      <td className="p-2 text-right font-bold text-red-400">+{f.impactPercentage}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Interactive Grad-CAM Heatmap Viewer & Thresholding (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <div>
                <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-cyan-500" />
                  Grad-CAM Spatial Localization Viewer
                </h3>
                <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Anatomical attribution mapped onto frontal chest radiograph
                </p>
              </div>

              {/* Pathology Selector Dropdown */}
              <select
                value={settings.selectedPathology}
                onChange={(e) => setSettings((prev) => ({ ...prev, selectedPathology: e.target.value }))}
                className="text-xs font-mono bg-gray-100 dark:bg-slate-900 border border-gray-300 dark:border-slate-700 rounded-lg px-2.5 py-1 text-gray-800 dark:text-gray-200 focus:outline-none focus:border-purple-500"
              >
                {Object.keys(PATHOLOGY_HOTSPOTS).map((pathology) => (
                  <option key={pathology} value={pathology}>
                    {pathology} ({(PATHOLOGY_HOTSPOTS[pathology].prob * 100).toFixed(0)}%)
                  </option>
                ))}
              </select>
            </div>

            {/* CXR Radiograph with Simulated Grad-CAM Canvas Overlay */}
            <div className="relative aspect-square max-h-72 w-full mx-auto bg-black rounded-lg overflow-hidden border border-gray-700 flex items-center justify-center shadow-inner">
              {/* Radiograph SVG Anatomical Silhouette */}
              <svg className="w-full h-full object-cover" viewBox="0 0 100 100">
                {/* Background Lung Fields */}
                <rect width="100" height="100" fill="#0d1117" />
                {/* Ribcage Wireframe */}
                <path d="M 20 20 Q 50 15 80 20" stroke="#21262d" strokeWidth="1" fill="none" />
                <path d="M 18 30 Q 50 24 82 30" stroke="#21262d" strokeWidth="1.2" fill="none" />
                <path d="M 16 42 Q 50 35 84 42" stroke="#21262d" strokeWidth="1.2" fill="none" />
                <path d="M 16 55 Q 50 48 84 55" stroke="#21262d" strokeWidth="1.2" fill="none" />
                <path d="M 18 68 Q 50 62 82 68" stroke="#21262d" strokeWidth="1.2" fill="none" />

                {/* Left & Right Lung Parenchyma */}
                <path
                  d="M 22 25 C 20 45, 20 70, 36 82 C 45 84, 46 72, 46 50 C 46 30, 35 22, 22 25 Z"
                  fill="#161b22"
                  stroke="#30363d"
                  strokeWidth="0.8"
                />
                <path
                  d="M 78 25 C 80 45, 80 70, 64 82 C 55 84, 54 72, 54 50 C 54 30, 65 22, 78 25 Z"
                  fill="#161b22"
                  stroke="#30363d"
                  strokeWidth="0.8"
                />

                {/* Cardiac Silhouette */}
                <path
                  d="M 46 45 C 44 65, 38 78, 52 82 C 62 82, 60 65, 54 45 Z"
                  fill="#21262d"
                  stroke="#484f58"
                  strokeWidth="1"
                />
                {/* Trachea & Main Bronchi */}
                <line x1="50" y1="10" x2="50" y2="35" stroke="#484f58" strokeWidth="2" />
                <line x1="50" y1="35" x2="40" y2="45" stroke="#484f58" strokeWidth="1.5" />
                <line x1="50" y1="35" x2="60" y2="45" stroke="#484f58" strokeWidth="1.5" />

                {/* DYNAMIC GRAD-CAM ATTRIBUTION HOTSPOT */}
                <defs>
                  <radialGradient id="gradcamGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity={settings.opacity} />
                    <stop offset="35%" stopColor="#f59e0b" stopOpacity={settings.opacity * 0.8} />
                    <stop offset="65%" stopColor="#06b6d4" stopOpacity={settings.opacity * 0.5} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                  </radialGradient>
                </defs>

                {/* Heatmap Ellipse positioned at Pathology Focal Hotspot */}
                <ellipse
                  cx={activeHotspot.cx}
                  cy={activeHotspot.cy}
                  rx={activeHotspot.rx * (1 + (1 - settings.threshold) * 0.5)}
                  ry={activeHotspot.ry * (1 + (1 - settings.threshold) * 0.5)}
                  fill="url(#gradcamGlow)"
                  filter={`blur(${Math.max(1, Math.round(settings.threshold * 4))}px)`}
                />

                {/* Crosshairs at Focal Maximum */}
                <circle
                  cx={activeHotspot.cx}
                  cy={activeHotspot.cy}
                  r="2"
                  fill="#ffffff"
                  stroke="#ef4444"
                  strokeWidth="0.5"
                />
              </svg>

              {/* Overlay Metadata Tag */}
              <div className="absolute top-2 left-2 px-2 py-1 rounded bg-black/75 backdrop-blur-md text-[10px] font-mono text-gray-200 border border-white/10">
                <span className="text-purple-400 font-bold">{settings.selectedPathology}</span> • Saliency IoU: 0.78
              </div>

              <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/75 backdrop-blur-md text-[10px] font-mono text-gray-300 border border-white/10">
                Threshold: {(settings.threshold).toFixed(2)} | Alpha: {Math.round(settings.opacity * 100)}%
              </div>
            </div>

            {/* Saliency Description */}
            <div className="p-2.5 rounded-lg border border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-slate-900/50 text-xs font-mono text-gray-600 dark:text-gray-300">
              <span className="font-bold text-gray-800 dark:text-gray-200">Localization Analysis: </span>
              {activeHotspot.description}
            </div>

            {/* Threshold & Opacity Controls */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-gray-700 dark:text-gray-300 font-bold flex items-center gap-1">
                    <Sliders className="w-3.5 h-3.5 text-purple-400" />
                    Cutoff Threshold
                  </span>
                  <span className="text-purple-400 font-bold">{settings.threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.85"
                  step="0.05"
                  value={settings.threshold}
                  onChange={(e) => setSettings((prev) => ({ ...prev, threshold: Number(e.target.value) }))}
                  className="w-full accent-purple-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
                />
                <div className="flex justify-between text-[10px] font-mono text-gray-400">
                  <span>Broader Area</span>
                  <span>Focal Peak</span>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-gray-700 dark:text-gray-300 font-bold flex items-center gap-1">
                    <Eye className="w-3.5 h-3.5 text-cyan-400" />
                    Heatmap Opacity
                  </span>
                  <span className="text-cyan-400 font-bold">{Math.round(settings.opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={settings.opacity}
                  onChange={(e) => setSettings((prev) => ({ ...prev, opacity: Number(e.target.value) }))}
                  className="w-full accent-cyan-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
                />
                <div className="flex justify-between text-[10px] font-mono text-gray-400">
                  <span>Transparent</span>
                  <span>Solid</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

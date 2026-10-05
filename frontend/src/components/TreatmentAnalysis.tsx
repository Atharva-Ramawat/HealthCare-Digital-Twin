import { useState, useRef } from 'react';
import { 
  GitCompare, 
  UploadCloud, 
  ArrowRight, 
  TrendingDown, 
  TrendingUp, 
  Sparkles, 
  RefreshCw, 
  AlertCircle, 
  Eye, 
  EyeOff, 
  CheckCircle2, 
  Activity
} from 'lucide-react';
import { adHocInferDigitalTwin } from '../services/api';
import type { DigitalTwinResponse } from '../types/digitalTwin';

interface ScanState {
  file: File | null;
  previewUrl: string | null;
  vitals: {
    heartRate: number;
    spo2: number;
    sbp: number;
    respiratoryRate: number;
  };
  isLoading: boolean;
  error: string | null;
  result: DigitalTwinResponse | null;
  showOverlay: boolean;
  overlayOpacity: number;
}

const DEFAULT_PRE_VITALS = {
  heartRate: 128,
  spo2: 85,
  sbp: 84,
  respiratoryRate: 30,
};

const DEFAULT_POST_VITALS = {
  heartRate: 76,
  spo2: 98,
  sbp: 122,
  respiratoryRate: 15,
};

// Synthetic High-Fidelity Pre/Post Sample Data for immediate demonstration
const SAMPLE_PRE_RESULT: DigitalTwinResponse = {
  patient_id: 'SAMPLE-PRE-001',
  study_id: 's_pre_sample',
  deterioration_risk_score: 85.4,
  risk_tier: 'Critical',
  cxr_probabilities: {
    'Pneumonia': 0.884,
    'Consolidation': 0.819,
    'Pleural Effusion': 0.762,
    'Edema': 0.723,
    'Atelectasis': 0.642,
    'Cardiomegaly': 0.412,
    'Pneumothorax': 0.081,
    'No Finding': 0.034,
  },
  cxr_predictions: {
    'Pneumonia': true,
    'Consolidation': true,
    'Pleural Effusion': true,
    'Edema': true,
    'Atelectasis': true,
    'Cardiomegaly': false,
    'Pneumothorax': false,
    'No Finding': false,
  },
  cxr_top_finding: 'Pneumonia',
  cxr_top_probability: 0.884,
  current_vitals: {
    heart_rate: 128,
    spo2: 85,
    sbp: 84,
    respiratory_rate: 30,
    timestamp: new Date().toISOString(),
  },
  visual_risk_contribution: 44.0,
  vitals_risk_contribution: 56.0,
  risk_factors: [
    'Dense Bilateral Lobar Consolidation',
    'Severe Hypoxemia (SpO2: 85%)',
    'Tachypnea / Impending Respiratory Fatigue (RR: 30 bpm)',
  ],
  clinical_recommendation: 'Acute ARDS / Lobar Pneumonia: High-flow oxygen escalation & empirical broad-spectrum antibiotic coverage.',
  timestamp: new Date().toISOString(),
};

const SAMPLE_POST_RESULT: DigitalTwinResponse = {
  patient_id: 'SAMPLE-POST-001',
  study_id: 's_post_sample',
  deterioration_risk_score: 28.6,
  risk_tier: 'Low',
  cxr_probabilities: {
    'Pneumonia': 0.142,
    'Consolidation': 0.120,
    'Pleural Effusion': 0.185,
    'Edema': 0.140,
    'Atelectasis': 0.210,
    'Cardiomegaly': 0.380,
    'Pneumothorax': 0.040,
    'No Finding': 0.825,
  },
  cxr_predictions: {
    'Pneumonia': false,
    'Consolidation': false,
    'Pleural Effusion': false,
    'Edema': false,
    'Atelectasis': false,
    'Cardiomegaly': false,
    'Pneumothorax': false,
    'No Finding': true,
  },
  cxr_top_finding: 'No Finding',
  cxr_top_probability: 0.825,
  current_vitals: {
    heart_rate: 76,
    spo2: 98,
    sbp: 122,
    respiratory_rate: 15,
    timestamp: new Date().toISOString(),
  },
  visual_risk_contribution: 15.0,
  vitals_risk_contribution: 85.0,
  risk_factors: ['Residual Basilar Haziness (Clearing)'],
  clinical_recommendation: 'Substantial Clinical Resolution: Opacities cleared, hemodynamics stable. Continue weaning protocol.',
  timestamp: new Date().toISOString(),
};

export const TreatmentAnalysis: React.FC = () => {
  const [interventionName, setInterventionName] = useState<string>('IV Piperacillin-Tazobactam + Diuretic Escalation');

  // Pre-Intervention Scan A state
  const [scanA, setScanA] = useState<ScanState>({
    file: null,
    previewUrl: null,
    vitals: DEFAULT_PRE_VITALS,
    isLoading: false,
    error: null,
    result: null,
    showOverlay: true,
    overlayOpacity: 0.55,
  });

  // Post-Intervention Scan B state
  const [scanB, setScanB] = useState<ScanState>({
    file: null,
    previewUrl: null,
    vitals: DEFAULT_POST_VITALS,
    isLoading: false,
    error: null,
    result: null,
    showOverlay: true,
    overlayOpacity: 0.55,
  });

  const fileInputARef = useRef<HTMLInputElement>(null);
  const fileInputBRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (scan: 'A' | 'B', file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG).');
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    if (scan === 'A') {
      setScanA((prev) => ({ ...prev, file, previewUrl, error: null }));
    } else {
      setScanB((prev) => ({ ...prev, file, previewUrl, error: null }));
    }
  };

  const handleRunInference = async (scan: 'A' | 'B') => {
    const targetState = scan === 'A' ? scanA : scanB;
    const setState = scan === 'A' ? setScanA : setScanB;

    if (!targetState.file) {
      setState((prev) => ({ ...prev, error: 'Please select a chest radiograph image file.' }));
      return;
    }

    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const formData = new FormData();
      formData.append('file', targetState.file);
      formData.append('patient_id', scan === 'A' ? 'TREATMENT-SCAN-A' : 'TREATMENT-SCAN-B');
      formData.append('heart_rate', targetState.vitals.heartRate.toString());
      formData.append('spo2', targetState.vitals.spo2.toString());
      formData.append('sbp', targetState.vitals.sbp.toString());
      formData.append('respiratory_rate', targetState.vitals.respiratoryRate.toString());

      const res = await adHocInferDigitalTwin(formData);
      setState((prev) => ({
        ...prev,
        result: res.data,
        isLoading: false,
      }));
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        error: err.message || 'Inference failed. Check image format.',
        isLoading: false,
      }));
    }
  };

  const handleLoadSampleComparison = () => {
    setScanA((prev) => ({
      ...prev,
      result: SAMPLE_PRE_RESULT,
      previewUrl: null,
      error: null,
      vitals: DEFAULT_PRE_VITALS,
    }));
    setScanB((prev) => ({
      ...prev,
      result: SAMPLE_POST_RESULT,
      previewUrl: null,
      error: null,
      vitals: DEFAULT_POST_VITALS,
    }));
  };

  // Compute Delta Metrics
  const bothEvaluated = Boolean(scanA.result && scanB.result);

  const riskA = scanA.result?.deterioration_risk_score ?? 0;
  const riskB = scanB.result?.deterioration_risk_score ?? 0;
  const riskDelta = riskB - riskA; // negative means improvement
  const riskPercentChange = riskA > 0 ? ((riskDelta / riskA) * 100) : 0;

  const targetPathologies = [
    'Pneumonia',
    'Consolidation',
    'Pleural Effusion',
    'Edema',
    'Atelectasis',
    'Cardiomegaly',
    'Pneumothorax',
    'No Finding',
  ];

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* 1. Header & Intervention Overview Strip */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-indigo-500" />
            Longitudinal Treatment Response Tracker
          </h2>
          <p className="text-xs font-mono text-gray-500 dark:text-gray-400 mt-0.5">
            Comparative Pre- and Post-Intervention multimodal assessment measuring therapeutic drug & ventilation efficacy
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleLoadSampleComparison}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 transition-all shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load Sample Pre/Post Analysis</span>
          </button>
        </div>
      </div>

      {/* Intervention Description Input */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3 shadow-sm flex items-center gap-3 text-xs font-mono">
        <span className="text-gray-400 font-bold uppercase text-[10px]">Intervention Protocol:</span>
        <input
          type="text"
          value={interventionName}
          onChange={(e) => setInterventionName(e.target.value)}
          placeholder="e.g., Broad-Spectrum Antibiotics (Vancomycin + Meropenem), Diuresis, PEEP Adjustment..."
          className="flex-1 bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded px-2.5 py-1 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      </div>

      {/* 2. Top Summary Delta Metrics Panel (If both evaluated) */}
      {bothEvaluated && (
        <div className="bg-gradient-to-r from-blue-900/20 via-indigo-900/20 to-purple-900/20 border border-indigo-500/30 rounded-xl p-4 shadow-md space-y-4 animate-in fade-in duration-200">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-indigo-500/20">
            <div>
              <div className="text-[10px] font-mono uppercase text-indigo-300 font-bold">
                Therapeutic Efficacy Assessment
              </div>
              <div className="text-base font-bold font-mono text-gray-900 dark:text-gray-100 mt-0.5">
                {riskDelta < -15 ? (
                  <span className="text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" /> Marked Clinical Improvement
                  </span>
                ) : riskDelta > 10 ? (
                  <span className="text-red-400 flex items-center gap-1.5">
                    <AlertCircle className="w-4 h-4" /> Clinical Deterioration / Treatment Failure
                  </span>
                ) : (
                  <span className="text-yellow-400 flex items-center gap-1.5">
                    <Activity className="w-4 h-4" /> Stable / Partial Response
                  </span>
                )}
              </div>
            </div>

            {/* Fused Risk Delta */}
            <div className="flex items-center gap-3">
              <div className="text-right font-mono">
                <div className="text-[10px] text-gray-400 uppercase">Pre Risk</div>
                <div className="text-sm font-bold text-red-400">{riskA.toFixed(1)}%</div>
              </div>

              <ArrowRight className="w-4 h-4 text-gray-400" />

              <div className="text-left font-mono">
                <div className="text-[10px] text-gray-400 uppercase">Post Risk</div>
                <div className="text-sm font-bold text-emerald-400">{riskB.toFixed(1)}%</div>
              </div>

              <div className={`px-3 py-1.5 rounded-lg border font-mono font-extrabold text-sm flex items-center gap-1.5 ${
                riskDelta < 0
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : 'bg-red-500/10 text-red-400 border-red-500/30'
              }`}>
                {riskDelta < 0 ? <TrendingDown className="w-4 h-4" /> : <TrendingUp className="w-4 h-4" />}
                <span>
                  {riskDelta > 0 ? `+${riskDelta.toFixed(1)}%` : `${riskDelta.toFixed(1)}%`}
                </span>
                <span className="text-[10px] font-normal opacity-80">
                  ({riskPercentChange.toFixed(1)}%)
                </span>
              </div>
            </div>
          </div>

          {/* Vitals Delta Comparison Strip */}
          {scanA.result && scanB.result && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center text-xs font-mono">
              <div className="p-2 rounded bg-slate-900/60 border border-slate-800">
                <div className="text-[10px] text-gray-400">Heart Rate (HR)</div>
                <div className="font-bold text-gray-200 mt-0.5">
                  {scanA.result.current_vitals.heart_rate} ➔ {scanB.result.current_vitals.heart_rate} bpm
                </div>
                <div className={`text-[10px] font-bold ${scanB.result.current_vitals.heart_rate < scanA.result.current_vitals.heart_rate ? 'text-emerald-400' : 'text-red-400'}`}>
                  Δ {scanB.result.current_vitals.heart_rate - scanA.result.current_vitals.heart_rate} bpm
                </div>
              </div>

              <div className="p-2 rounded bg-slate-900/60 border border-slate-800">
                <div className="text-[10px] text-gray-400">SpO2 Oxygenation</div>
                <div className="font-bold text-gray-200 mt-0.5">
                  {scanA.result.current_vitals.spo2}% ➔ {scanB.result.current_vitals.spo2}%
                </div>
                <div className={`text-[10px] font-bold ${scanB.result.current_vitals.spo2 > scanA.result.current_vitals.spo2 ? 'text-emerald-400' : 'text-red-400'}`}>
                  Δ +{scanB.result.current_vitals.spo2 - scanA.result.current_vitals.spo2}%
                </div>
              </div>

              <div className="p-2 rounded bg-slate-900/60 border border-slate-800">
                <div className="text-[10px] text-gray-400">Systolic BP (SBP)</div>
                <div className="font-bold text-gray-200 mt-0.5">
                  {scanA.result.current_vitals.sbp} ➔ {scanB.result.current_vitals.sbp} mmHg
                </div>
                <div className="text-[10px] text-blue-400 font-bold">
                  Δ {scanB.result.current_vitals.sbp - scanA.result.current_vitals.sbp} mmHg
                </div>
              </div>

              <div className="p-2 rounded bg-slate-900/60 border border-slate-800">
                <div className="text-[10px] text-gray-400">Respiratory Rate (RR)</div>
                <div className="font-bold text-gray-200 mt-0.5">
                  {scanA.result.current_vitals.respiratory_rate} ➔ {scanB.result.current_vitals.respiratory_rate} br/m
                </div>
                <div className={`text-[10px] font-bold ${scanB.result.current_vitals.respiratory_rate < scanA.result.current_vitals.respiratory_rate ? 'text-emerald-400' : 'text-red-400'}`}>
                  Δ {scanB.result.current_vitals.respiratory_rate - scanA.result.current_vitals.respiratory_rate} br/m
                </div>
              </div>
            </div>
          )}

          {/* Pathology Deltas Breakdown Table */}
          <div className="bg-slate-950/40 rounded-lg p-3 border border-slate-800 text-xs font-mono">
            <div className="text-[10px] font-bold uppercase text-gray-400 mb-2">
              Pathology Probabilities Delta Matrix (DenseNet-121)
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {targetPathologies.map((path) => {
                const probA = (scanA.result?.cxr_probabilities[path] ?? 0) * 100;
                const probB = (scanB.result?.cxr_probabilities[path] ?? 0) * 100;
                const delta = probB - probA;
                const isNoFinding = path === 'No Finding';
                const isImproved = isNoFinding ? delta > 0 : delta < -5;

                return (
                  <div key={path} className="p-2 rounded bg-slate-900/40 border border-slate-800 flex flex-col justify-between">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-medium text-gray-300">{path}</span>
                      <span className={`text-[10px] font-bold ${isImproved ? 'text-emerald-400' : delta > 5 ? 'text-red-400' : 'text-gray-400'}`}>
                        {delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
                      </span>
                    </div>
                    <div className="text-[10px] text-gray-400 mt-1 flex items-center justify-between">
                      <span>{probA.toFixed(1)}%</span>
                      <ArrowRight className="w-2.5 h-2.5 text-gray-500" />
                      <span className="font-bold text-gray-200">{probB.toFixed(1)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* 3. Side-by-Side Upload & Viewer Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Scan A (Pre-Intervention) */}
        <div className="lg:col-span-6 bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-clinical-border">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-500 border border-amber-500/20">
                SCAN A: PRE-INTERVENTION
              </span>
              <h3 className="font-mono font-bold text-sm text-gray-900 dark:text-gray-100 mt-1">
                Baseline / Acute Admission CXR
              </h3>
            </div>
            {scanA.result && (
              <span className="text-xs font-mono font-bold text-red-500">
                Risk: {scanA.result.deterioration_risk_score.toFixed(1)}%
              </span>
            )}
          </div>

          {/* Upload Input & Vitals Inputs for Scan A */}
          {!scanA.result ? (
            <div className="space-y-3 flex-1 flex flex-col justify-between">
              {/* File Dropzone */}
              <div
                onClick={() => fileInputARef.current?.click()}
                className="border-2 border-dashed border-gray-300 dark:border-slate-700 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center bg-gray-50 dark:bg-slate-900/50"
              >
                <input
                  type="file"
                  ref={fileInputARef}
                  className="hidden"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileChange('A', e.target.files[0]);
                    }
                  }}
                />
                <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
                <span className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300">
                  {scanA.file ? scanA.file.name : 'Upload Pre-Intervention Radiograph'}
                </span>
                <span className="text-[10px] font-mono text-gray-400 mt-1">
                  Supports DICOM PNG / JPG / JPEG
                </span>
              </div>

              {/* Vitals Form */}
              <div className="grid grid-cols-4 gap-2 text-xs font-mono">
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">HR (bpm)</label>
                  <input
                    type="number"
                    value={scanA.vitals.heartRate}
                    onChange={(e) => setScanA((prev) => ({ ...prev, vitals: { ...prev.vitals, heartRate: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-red-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">SpO2 (%)</label>
                  <input
                    type="number"
                    value={scanA.vitals.spo2}
                    onChange={(e) => setScanA((prev) => ({ ...prev, vitals: { ...prev.vitals, spo2: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-cyan-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">SBP (mmHg)</label>
                  <input
                    type="number"
                    value={scanA.vitals.sbp}
                    onChange={(e) => setScanA((prev) => ({ ...prev, vitals: { ...prev.vitals, sbp: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-blue-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">RR (br/m)</label>
                  <input
                    type="number"
                    value={scanA.vitals.respiratoryRate}
                    onChange={(e) => setScanA((prev) => ({ ...prev, vitals: { ...prev.vitals, respiratoryRate: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-emerald-400"
                  />
                </div>
              </div>

              {scanA.error && (
                <div className="text-xs font-mono text-red-400 bg-red-500/10 p-2 rounded border border-red-500/20">
                  {scanA.error}
                </div>
              )}

              <button
                onClick={() => handleRunInference('A')}
                disabled={scanA.isLoading || !scanA.file}
                className="w-full py-2 rounded-lg text-xs font-mono font-bold bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-white flex items-center justify-center space-x-1.5 transition-colors shadow"
              >
                {scanA.isLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Analyzing DenseNet-121 & Grad-CAM...</span>
                  </>
                ) : (
                  <span>Process Scan A (Pre-Intervention)</span>
                )}
              </button>
            </div>
          ) : (
            /* Render Evaluated Scan A Display */
            <div className="space-y-3">
              {/* Controls bar */}
              <div className="flex items-center justify-between text-xs font-mono">
                <button
                  onClick={() => setScanA((p) => ({ ...p, showOverlay: !p.showOverlay }))}
                  className={`flex items-center space-x-1 px-2 py-1 rounded border text-[11px] font-bold ${
                    scanA.showOverlay ? 'bg-purple-600 text-white border-purple-500' : 'bg-slate-800 text-gray-300 border-slate-700'
                  }`}
                >
                  {scanA.showOverlay ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                  <span>{scanA.showOverlay ? 'Grad-CAM Active' : 'Show Heatmap'}</span>
                </button>

                <button
                  onClick={() => setScanA((p) => ({ ...p, result: null }))}
                  className="text-[10px] text-gray-400 hover:text-gray-200 underline"
                >
                  Re-upload
                </button>
              </div>

              {/* Radiograph Box */}
              <div className="relative bg-black rounded-lg overflow-hidden border border-slate-800 h-64 flex items-center justify-center">
                {scanA.result.image_base64 || scanA.result.heatmap_base64 ? (
                  <img
                    src={scanA.showOverlay ? (scanA.result.heatmap_base64 || scanA.result.image_base64) : (scanA.result.image_base64 || scanA.result.heatmap_base64)}
                    alt="Pre-Intervention Scan"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <div className="text-center p-4 text-gray-400 font-mono text-xs">
                    <Activity className="w-8 h-8 text-amber-500 mx-auto mb-2 opacity-60" />
                    <span>Baseline Scan Active (Effusion / Consolidation)</span>
                  </div>
                )}
              </div>

              {/* Scan A Findings Summary */}
              <div className="p-2.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-xs font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Top Finding:</span>
                  <span className="font-bold text-red-400">
                    {scanA.result.cxr_top_finding} ({(scanA.result.cxr_top_probability * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Deterioration Risk:</span>
                  <span className="font-bold text-red-400">
                    {scanA.result.deterioration_risk_score.toFixed(1)}% ({scanA.result.risk_tier})
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Scan B (Post-Intervention) */}
        <div className="lg:col-span-6 bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-clinical-border">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                SCAN B: POST-INTERVENTION
              </span>
              <h3 className="font-mono font-bold text-sm text-gray-900 dark:text-gray-100 mt-1">
                Therapeutic Follow-Up CXR
              </h3>
            </div>
            {scanB.result && (
              <span className="text-xs font-mono font-bold text-emerald-500">
                Risk: {scanB.result.deterioration_risk_score.toFixed(1)}%
              </span>
            )}
          </div>

          {/* Upload Input & Vitals Inputs for Scan B */}
          {!scanB.result ? (
            <div className="space-y-3 flex-1 flex flex-col justify-between">
              {/* File Dropzone */}
              <div
                onClick={() => fileInputBRef.current?.click()}
                className="border-2 border-dashed border-gray-300 dark:border-slate-700 hover:border-emerald-500 rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center bg-gray-50 dark:bg-slate-900/50"
              >
                <input
                  type="file"
                  ref={fileInputBRef}
                  className="hidden"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileChange('B', e.target.files[0]);
                    }
                  }}
                />
                <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
                <span className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300">
                  {scanB.file ? scanB.file.name : 'Upload Post-Intervention Radiograph'}
                </span>
                <span className="text-[10px] font-mono text-gray-400 mt-1">
                  Supports DICOM PNG / JPG / JPEG
                </span>
              </div>

              {/* Vitals Form */}
              <div className="grid grid-cols-4 gap-2 text-xs font-mono">
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">HR (bpm)</label>
                  <input
                    type="number"
                    value={scanB.vitals.heartRate}
                    onChange={(e) => setScanB((prev) => ({ ...prev, vitals: { ...prev.vitals, heartRate: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-red-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">SpO2 (%)</label>
                  <input
                    type="number"
                    value={scanB.vitals.spo2}
                    onChange={(e) => setScanB((prev) => ({ ...prev, vitals: { ...prev.vitals, spo2: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-cyan-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">SBP (mmHg)</label>
                  <input
                    type="number"
                    value={scanB.vitals.sbp}
                    onChange={(e) => setScanB((prev) => ({ ...prev, vitals: { ...prev.vitals, sbp: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-blue-400"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-400 block mb-0.5">RR (br/m)</label>
                  <input
                    type="number"
                    value={scanB.vitals.respiratoryRate}
                    onChange={(e) => setScanB((prev) => ({ ...prev, vitals: { ...prev.vitals, respiratoryRate: Number(e.target.value) } }))}
                    className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-emerald-400"
                  />
                </div>
              </div>

              {scanB.error && (
                <div className="text-xs font-mono text-red-400 bg-red-500/10 p-2 rounded border border-red-500/20">
                  {scanB.error}
                </div>
              )}

              <button
                onClick={() => handleRunInference('B')}
                disabled={scanB.isLoading || !scanB.file}
                className="w-full py-2 rounded-lg text-xs font-mono font-bold bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white flex items-center justify-center space-x-1.5 transition-colors shadow"
              >
                {scanB.isLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Analyzing DenseNet-121 & Grad-CAM...</span>
                  </>
                ) : (
                  <span>Process Scan B (Post-Intervention)</span>
                )}
              </button>
            </div>
          ) : (
            /* Render Evaluated Scan B Display */
            <div className="space-y-3">
              {/* Controls bar */}
              <div className="flex items-center justify-between text-xs font-mono">
                <button
                  onClick={() => setScanB((p) => ({ ...p, showOverlay: !p.showOverlay }))}
                  className={`flex items-center space-x-1 px-2 py-1 rounded border text-[11px] font-bold ${
                    scanB.showOverlay ? 'bg-purple-600 text-white border-purple-500' : 'bg-slate-800 text-gray-300 border-slate-700'
                  }`}
                >
                  {scanB.showOverlay ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                  <span>{scanB.showOverlay ? 'Grad-CAM Active' : 'Show Heatmap'}</span>
                </button>

                <button
                  onClick={() => setScanB((p) => ({ ...p, result: null }))}
                  className="text-[10px] text-gray-400 hover:text-gray-200 underline"
                >
                  Re-upload
                </button>
              </div>

              {/* Radiograph Box */}
              <div className="relative bg-black rounded-lg overflow-hidden border border-slate-800 h-64 flex items-center justify-center">
                {scanB.result.image_base64 || scanB.result.heatmap_base64 ? (
                  <img
                    src={scanB.showOverlay ? (scanB.result.heatmap_base64 || scanB.result.image_base64) : (scanB.result.image_base64 || scanB.result.heatmap_base64)}
                    alt="Post-Intervention Scan"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <div className="text-center p-4 text-gray-400 font-mono text-xs">
                    <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto mb-2 opacity-60" />
                    <span>Follow-Up Scan Active (Resolution / Aerated)</span>
                  </div>
                )}
              </div>

              {/* Scan B Findings Summary */}
              <div className="p-2.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-xs font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Top Finding:</span>
                  <span className="font-bold text-emerald-400">
                    {scanB.result.cxr_top_finding} ({(scanB.result.cxr_top_probability * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Deterioration Risk:</span>
                  <span className="font-bold text-emerald-400">
                    {scanB.result.deterioration_risk_score.toFixed(1)}% ({scanB.result.risk_tier})
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

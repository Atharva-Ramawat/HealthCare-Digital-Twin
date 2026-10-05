import { useState, useRef } from 'react';
import { 
  AlertTriangle, 
  FlaskConical, 
  UploadCloud, 
  RefreshCw, 
  Sparkles, 
  RotateCcw, 
  Heart,
  Droplets,
  Wind,
  Activity,
  ShieldAlert,
  X
} from 'lucide-react';
import { adHocInferDigitalTwin } from '../services/api';
import { VitalsTelemetryGrid } from './VitalsTelemetryGrid';
import { CXRFusionViewer } from './CXRFusionViewer';
import { UnifiedRiskGauge } from './UnifiedRiskGauge';
import { UploadIntakeModal } from './UploadIntakeModal';
import type { DigitalTwinResponse, PatientProfile } from '../types/digitalTwin';

interface SandboxViewProps {
  isDarkMode: boolean;
}

export const SandboxView: React.FC<SandboxViewProps> = ({ isDarkMode }) => {
  const [twinData, setTwinData] = useState<DigitalTwinResponse | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState<string>('SANDBOX-901');
  const [patientName, setPatientName] = useState<string>('Simulation Candidate');
  const [age, setAge] = useState<number>(62);
  const [gender, setGender] = useState<string>('M');

  // Bedside Vitals
  const [heartRate, setHeartRate] = useState<number>(118);
  const [spo2, setSpo2] = useState<number>(88);
  const [sbp, setSbp] = useState<number>(86);
  const [respiratoryRate, setRespiratoryRate] = useState<number>(28);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [modalityRejected, setModalityRejected] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const applyPreset = (preset: 'critical' | 'moderate' | 'normal') => {
    if (preset === 'critical') {
      setHeartRate(134);
      setSpo2(83);
      setSbp(78);
      setRespiratoryRate(32);
      setPatientName('Acute ARDS / Septic Deterioration');
    } else if (preset === 'moderate') {
      setHeartRate(106);
      setSpo2(91);
      setSbp(102);
      setRespiratoryRate(22);
      setPatientName('Post-Thoracic Monitoring');
    } else {
      setHeartRate(72);
      setSpo2(98);
      setSbp(120);
      setRespiratoryRate(15);
      setPatientName('Stable Physiological Assessment');
    }
  };

  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setErrorMessage('Please select a valid image file (PNG, JPG, JPEG).');
      return;
    }
    setErrorMessage(null);
    setModalityRejected(false);
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleRunInference = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setErrorMessage(null);
    setModalityRejected(false);

    if (!file) {
      setErrorMessage('Please upload or drag & drop a chest radiograph image file.');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('patient_id', patientId.trim() || 'SANDBOX-001');
      formData.append('heart_rate', heartRate.toString());
      formData.append('spo2', spo2.toString());
      formData.append('sbp', sbp.toString());
      formData.append('respiratory_rate', respiratoryRate.toString());
      formData.append('age', age.toString());
      formData.append('gender', gender);

      const res = await adHocInferDigitalTwin(formData);
      setTwinData(res.data);
    } catch (err: any) {
      const msg = err.message || '';
      if (
        msg.toLowerCase().includes('invalid image modality') ||
        msg.toLowerCase().includes('chest radiograph') ||
        msg.toLowerCase().includes('not a chest x-ray')
      ) {
        setModalityRejected(true);
        setErrorMessage('File rejected: Input image is not a valid Chest Radiograph.');
      } else {
        setErrorMessage(msg || 'Inference failed. Check image and values.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalSuccess = (data: DigitalTwinResponse, _profile: PatientProfile) => {
    setTwinData(data);
  };

  const handleReset = () => {
    setTwinData(null);
    setFile(null);
    setPreviewUrl(null);
    setErrorMessage(null);
    setModalityRejected(false);
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* 1. Mandatory Amber / Yellow Warning Banner */}
      <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 dark:bg-amber-950/30 p-4 shadow-sm">
        <div className="flex items-start space-x-3">
          <div className="p-2 rounded-lg bg-amber-500/20 text-amber-500 border border-amber-500/30 flex-shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xs md:text-sm font-extrabold font-mono uppercase tracking-wide text-amber-600 dark:text-amber-400">
              SIMULATION ENVIRONMENT: Synthetic trajectory generation active. No historical clinical data present.
            </h2>
            <p className="text-xs font-mono text-amber-700/90 dark:text-amber-300/80 mt-1 leading-relaxed">
              All inferences, temporal vital trajectories, and Grad-CAM visualizations generated here are synthesized for ad-hoc exploration. Verified MIMIC ICU records remain isolated under "Live Ward".
            </p>
          </div>
        </div>
      </div>

      {/* 1b. OOD / Non-Radiograph Rejection Banner (HTTP 400 Modality Gatekeeper) */}
      {modalityRejected && (
        <div className="rounded-xl border-2 border-red-500/70 bg-red-500/10 dark:bg-red-950/40 p-4 shadow-lg shadow-red-500/10 flex items-start justify-between gap-3 animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-start space-x-3">
            <div className="p-2 rounded-lg bg-red-500/20 text-red-500 border border-red-500/30 flex-shrink-0">
              <ShieldAlert className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="text-xs md:text-sm font-extrabold font-mono uppercase tracking-wide text-red-600 dark:text-red-400 flex items-center gap-2">
                <span>NON-RADIOGRAPH IMAGE REJECTED BY OOD GATEKEEPER</span>
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/20 text-red-400 border border-red-500/30 font-mono">
                  HTTP 400
                </span>
              </div>
              <p className="text-xs font-mono text-red-700/90 dark:text-red-300/90 mt-1 leading-relaxed">
                The uploaded file was rejected because it is not a valid Chest Radiograph. DenseNet-121 inference was safely halted to prevent hallucinated pathology scores. Please upload a genuine thoracic X-ray (PNG/JPEG/DICOM).
              </p>
            </div>
          </div>
          <button
            onClick={() => setModalityRejected(false)}
            className="p-1 rounded text-red-400 hover:text-red-200 hover:bg-red-500/20 transition-colors"
            title="Dismiss warning"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* 2. Top Controls & Header */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h1 className="text-sm font-bold font-mono uppercase tracking-wider text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-purple-500" />
            Ad-Hoc Manual Intake & Simulation Core
          </h1>
          <p className="text-xs font-mono text-gray-500 dark:text-gray-400 mt-0.5">
            Execute DenseNet-121 forward pass, Grad-CAM attribution, and vitals synthesis on ungrounded clinical data
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {twinData && (
            <button
              onClick={handleReset}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-700 border border-gray-300 dark:border-slate-700 transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Sandbox</span>
            </button>
          )}

          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-purple-600/10 hover:bg-purple-600/20 text-purple-600 dark:text-purple-400 border border-purple-500/30 transition-all shadow-sm"
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Open Intake Modal</span>
          </button>
        </div>
      </div>

      {/* 3. Inline Upload & Simulation Form (if no active twinData) */}
      {!twinData ? (
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-5 shadow-sm space-y-4">
          {/* Presets Strip */}
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-gray-100 dark:border-clinical-border">
            <span className="text-[11px] font-mono text-gray-400 uppercase font-bold flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-purple-400" /> Quick Simulation Presets:
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => applyPreset('critical')}
                className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/30 transition-colors"
              >
                Critical ARDS
              </button>
              <button
                type="button"
                onClick={() => applyPreset('moderate')}
                className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 border border-amber-500/30 transition-colors"
              >
                Moderate Effusion
              </button>
              <button
                type="button"
                onClick={() => applyPreset('normal')}
                className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 border border-emerald-500/30 transition-colors"
              >
                Stable Baseline
              </button>
            </div>
          </div>

          <form onSubmit={handleRunInference} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
              {/* Left Column: Image Upload Dropzone (5 cols) */}
              <div className="md:col-span-5 flex flex-col">
                <label className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300 mb-1 flex items-center gap-1.5">
                  <UploadCloud className="w-3.5 h-3.5 text-purple-500" />
                  Chest Radiograph File
                </label>
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 min-h-[200px] border-2 border-dashed border-gray-300 dark:border-slate-700 hover:border-purple-500 rounded-xl p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center bg-gray-50 dark:bg-slate-900/50 group"
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFileChange(e.target.files[0]);
                      }
                    }}
                  />
                  {previewUrl ? (
                    <div className="relative w-full h-44 flex items-center justify-center">
                      <img src={previewUrl} alt="Upload Preview" className="max-h-full max-w-full object-contain rounded" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded text-white font-mono text-xs">
                        Click to change image
                      </div>
                    </div>
                  ) : (
                    <>
                      <UploadCloud className="w-10 h-10 text-gray-400 group-hover:text-purple-500 transition-colors mb-2" />
                      <span className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300">
                        Drop radiograph or click to browse
                      </span>
                      <span className="text-[10px] font-mono text-gray-400 mt-1">
                        PNG, JPG, or DICOM-exported format
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Right Column: Vitals & Demographics (7 cols) */}
              <div className="md:col-span-7 space-y-3">
                {/* Demographics Strip */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-0.5">Subject Tag</label>
                    <input
                      type="text"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 font-bold text-gray-800 dark:text-gray-200"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-0.5">Simulation Name</label>
                    <input
                      type="text"
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 font-bold text-gray-800 dark:text-gray-200"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-0.5">Age</label>
                    <input
                      type="number"
                      value={age}
                      onChange={(e) => setAge(Number(e.target.value))}
                      className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-gray-800 dark:text-gray-200"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-0.5">Gender</label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="w-full p-1.5 rounded bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-bold text-gray-800 dark:text-gray-200"
                    >
                      <option value="M">Male (M)</option>
                      <option value="F">Female (F)</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                {/* Vitals Inputs */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-red-500/5 border border-red-500/20">
                    <label className="text-[10px] text-red-400 flex items-center gap-1 font-bold mb-1">
                      <Heart className="w-3 h-3" /> Heart Rate
                    </label>
                    <input
                      type="number"
                      value={heartRate}
                      onChange={(e) => setHeartRate(Number(e.target.value))}
                      className="w-full p-1.5 rounded bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-extrabold text-red-500 text-sm"
                    />
                    <span className="text-[9px] text-gray-400 block text-center mt-0.5">20 - 250 bpm</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-cyan-500/5 border border-cyan-500/20">
                    <label className="text-[10px] text-cyan-400 flex items-center gap-1 font-bold mb-1">
                      <Droplets className="w-3 h-3" /> SpO2 Sat
                    </label>
                    <input
                      type="number"
                      value={spo2}
                      onChange={(e) => setSpo2(Number(e.target.value))}
                      className="w-full p-1.5 rounded bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-extrabold text-cyan-400 text-sm"
                    />
                    <span className="text-[9px] text-gray-400 block text-center mt-0.5">50 - 100 %</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-blue-500/5 border border-blue-500/20">
                    <label className="text-[10px] text-blue-400 flex items-center gap-1 font-bold mb-1">
                      <Activity className="w-3 h-3" /> Systolic BP
                    </label>
                    <input
                      type="number"
                      value={sbp}
                      onChange={(e) => setSbp(Number(e.target.value))}
                      className="w-full p-1.5 rounded bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-extrabold text-blue-400 text-sm"
                    />
                    <span className="text-[9px] text-gray-400 block text-center mt-0.5">40 - 260 mmHg</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                    <label className="text-[10px] text-emerald-400 flex items-center gap-1 font-bold mb-1">
                      <Wind className="w-3 h-3" /> Resp Rate
                    </label>
                    <input
                      type="number"
                      value={respiratoryRate}
                      onChange={(e) => setRespiratoryRate(Number(e.target.value))}
                      className="w-full p-1.5 rounded bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-800 text-center font-extrabold text-emerald-400 text-sm"
                    />
                    <span className="text-[9px] text-gray-400 block text-center mt-0.5">4 - 70 br/m</span>
                  </div>
                </div>

                {errorMessage && (
                  <div className="p-2.5 rounded-lg bg-red-500/10 border border-red-500/30 text-xs font-mono text-red-400">
                    {errorMessage}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isLoading || !file}
                  className="w-full py-2.5 rounded-lg text-xs font-mono font-bold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 text-white flex items-center justify-center space-x-2 transition-all shadow-md shadow-purple-600/20"
                >
                  {isLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Executing Real-Time Vision & Multimodal Fusion...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Run Isolated Ad-Hoc Evaluation</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        </div>
      ) : (
        /* 4. Evaluated Sandbox Digital Twin View */
        <div className="space-y-4">
          {/* Telemetry Grid (Synthesized 24h Vitals Trajectory) */}
          <section>
            <VitalsTelemetryGrid
              trajectory={twinData.vitals_trajectory_24h || []}
              currentVitals={twinData.current_vitals}
              isDarkMode={isDarkMode}
            />
          </section>

          {/* Lower Grid: CXR Fusion Core + Unified Risk Gauge */}
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
            <div className="lg:col-span-7">
              <CXRFusionViewer
                studyId={twinData.study_id}
                probabilities={twinData.cxr_probabilities}
                predictions={twinData.cxr_predictions}
                topFinding={twinData.cxr_top_finding}
                topProbability={twinData.cxr_top_probability}
                heatmapBase64={twinData.heatmap_base64 || twinData.heatmap_image_base64}
                inputImageBase64={twinData.image_base64 || twinData.input_image_base64}
              />
            </div>

            <div className="lg:col-span-5">
              <UnifiedRiskGauge
                score={twinData.deterioration_risk_score}
                tier={twinData.risk_tier}
                visualContribution={twinData.visual_risk_contribution}
                vitalsContribution={twinData.vitals_risk_contribution}
                riskFactors={twinData.risk_factors}
                recommendation={twinData.clinical_recommendation}
              />
            </div>
          </section>
        </div>
      )}

      {/* Modal fallback */}
      <UploadIntakeModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleModalSuccess}
      />
    </div>
  );
};

import { useState, useRef } from 'react';
import { 
  X, 
  UploadCloud, 
  Heart, 
  Droplets, 
  Activity, 
  Wind, 
  Sparkles, 
  RefreshCw, 
  AlertCircle, 
  Image as ImageIcon
} from 'lucide-react';
import { adHocInferDigitalTwin } from '../services/api';
import type { DigitalTwinResponse, PatientProfile } from '../types/digitalTwin';

interface UploadIntakeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (data: DigitalTwinResponse, patient: PatientProfile) => void;
}

export const UploadIntakeModal: React.FC<UploadIntakeModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [patientId, setPatientId] = useState<string>(`CUSTOM-${Math.floor(1000 + Math.random() * 9000)}`);
  const [patientName, setPatientName] = useState<string>('Unregistered ICU Intake');
  const [age, setAge] = useState<number>(58);
  const [gender, setGender] = useState<string>('M');
  const [unit, setUnit] = useState<string>('MICU Bed 12');

  // Vitals
  const [heartRate, setHeartRate] = useState<number>(112);
  const [spo2, setSpo2] = useState<number>(88);
  const [sbp, setSbp] = useState<number>(85);
  const [respiratoryRate, setRespiratoryRate] = useState<number>(28);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setErrorMessage('Please select a valid image file (PNG, JPG, JPEG).');
      return;
    }
    setErrorMessage(null);
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const applyPreset = (preset: 'critical' | 'moderate' | 'normal') => {
    if (preset === 'critical') {
      setHeartRate(132);
      setSpo2(83);
      setSbp(78);
      setRespiratoryRate(32);
      setPatientName('Acute Respiratory Failure');
    } else if (preset === 'moderate') {
      setHeartRate(104);
      setSpo2(91);
      setSbp(105);
      setRespiratoryRate(22);
      setPatientName('Post-Op Monitoring');
    } else {
      setHeartRate(74);
      setSpo2(98);
      setSbp(122);
      setRespiratoryRate(15);
      setPatientName('Stable Assessment');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    // Client-side validations
    if (!file) {
      setErrorMessage('A chest radiograph image is required. Please upload or drag & drop an image.');
      return;
    }
    if (heartRate < 20 || heartRate > 250) {
      setErrorMessage('Heart rate must be between 20 and 250 bpm.');
      return;
    }
    if (spo2 < 50 || spo2 > 100) {
      setErrorMessage('SpO2 must be between 50% and 100%.');
      return;
    }
    if (sbp < 40 || sbp > 260) {
      setErrorMessage('Systolic blood pressure must be between 40 and 260 mmHg.');
      return;
    }
    if (respiratoryRate < 4 || respiratoryRate > 70) {
      setErrorMessage('Respiratory rate must be between 4 and 70 breaths/min.');
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('patient_id', patientId.trim() || 'CUSTOM-001');
      formData.append('heart_rate', heartRate.toString());
      formData.append('spo2', spo2.toString());
      formData.append('sbp', sbp.toString());
      formData.append('respiratory_rate', respiratoryRate.toString());
      formData.append('age', age.toString());
      formData.append('gender', gender);

      const result = await adHocInferDigitalTwin(formData);

      const customProfile: PatientProfile = {
        id: result.data.patient_id,
        study_id: result.data.study_id,
        name: patientName.trim() || 'Manual Intake Subject',
        age: age,
        gender: gender,
        unit: unit,
        bed: unit.split(' ').pop() || 'B-01',
        admission_diagnosis: `Ad-Hoc: ${result.data.cxr_top_finding}`,
        intubated: result.data.deterioration_risk_score > 75,
      };

      onSuccess(result.data, customProfile);
      onClose();
    } catch (err: any) {
      setErrorMessage(err.message || 'Inference failed. Check image format and values.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 md:p-4 bg-black/75 backdrop-blur-sm overflow-y-auto">
      <div 
        className="relative w-full max-w-4xl bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh] animate-in fade-in zoom-in-95 duration-150"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-clinical-border bg-gray-50/50 dark:bg-slate-900/50">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-purple-500/10 text-purple-500 dark:text-purple-400 border border-purple-500/20">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold font-mono text-gray-900 dark:text-gray-100 flex items-center gap-2">
                Manual Patient Intake & CXR Upload
              </h2>
              <p className="text-xs font-mono text-gray-500 dark:text-gray-400">
                Run live DenseNet-121 forward pass, Grad-CAM heatmap, and Multimodal ICU Fusion
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="mx-6 mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 dark:text-red-400 text-xs font-mono flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left Column: Image Upload & Dropzone (5 cols) */}
            <div className="lg:col-span-5 flex flex-col space-y-3">
              <label className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-300">
                1. Chest Radiograph (CXR)
              </label>

              <div
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex-1 min-h-[260px] border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition-all ${
                  isDragOver
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-gray-300 dark:border-slate-700 hover:border-purple-400 bg-gray-50 dark:bg-slate-900/60'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileChange(e.target.files[0]);
                    }
                  }}
                />

                {previewUrl ? (
                  <div className="flex flex-col items-center justify-center space-y-2">
                    <img
                      src={previewUrl}
                      alt="CXR Preview"
                      className="max-h-48 max-w-full object-contain rounded-lg shadow-md border border-gray-200 dark:border-slate-700"
                    />
                    <div className="text-[11px] font-mono text-gray-500 text-center">
                      <span className="font-bold text-gray-800 dark:text-gray-200">{file?.name}</span>
                      <div>{(file!.size / 1024).toFixed(1)} KB • Click or drop to replace</div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center text-center space-y-2.5 p-4">
                    <div className="p-3 rounded-full bg-purple-500/10 text-purple-400">
                      <ImageIcon className="w-8 h-8" />
                    </div>
                    <div className="font-mono text-xs font-bold text-gray-800 dark:text-gray-200">
                      Drag and drop CXR image here
                    </div>
                    <p className="text-[11px] font-mono text-gray-400 max-w-[200px]">
                      Supports PNG, JPEG, JPG (DenseNet 224x224 Auto-Resize)
                    </p>
                    <button
                      type="button"
                      className="px-3 py-1 rounded-lg bg-purple-600 text-white text-xs font-mono font-bold shadow hover:bg-purple-700"
                    >
                      Browse Files
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Demographics & Manual Vitals (7 cols) */}
            <div className="lg:col-span-7 space-y-5">
              
              {/* Demographics Strip */}
              <div>
                <label className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-300">
                  2. Patient Demographics
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-2">
                  <div>
                    <label className="text-[10px] font-mono text-gray-400 uppercase">Patient ID</label>
                    <input
                      type="text"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-gray-50 dark:bg-slate-900 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-mono text-gray-400 uppercase">Age</label>
                    <input
                      type="number"
                      min={0}
                      max={120}
                      value={age}
                      onChange={(e) => setAge(parseInt(e.target.value) || 0)}
                      className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-gray-50 dark:bg-slate-900 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-mono text-gray-400 uppercase">Gender</label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-gray-50 dark:bg-slate-900 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                    >
                      <option value="M">Male (M)</option>
                      <option value="F">Female (F)</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] font-mono text-gray-400 uppercase">ICU Bed</label>
                    <input
                      type="text"
                      value={unit}
                      onChange={(e) => setUnit(e.target.value)}
                      className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-gray-50 dark:bg-slate-900 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Bedside Vitals Section */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-300">
                    3. Bedside Physiological Telemetry
                  </label>
                  
                  {/* Quick Presets */}
                  <div className="flex items-center space-x-1">
                    <span className="text-[10px] font-mono text-gray-400 mr-1 hidden sm:inline">Presets:</span>
                    <button
                      type="button"
                      onClick={() => applyPreset('critical')}
                      className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20"
                    >
                      Severe Shock
                    </button>
                    <button
                      type="button"
                      onClick={() => applyPreset('moderate')}
                      className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-500 border border-amber-500/20 hover:bg-amber-500/20"
                    >
                      Moderate
                    </button>
                    <button
                      type="button"
                      onClick={() => applyPreset('normal')}
                      className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 hover:bg-emerald-500/20"
                    >
                      Stable
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {/* Heart Rate */}
                  <div className="p-3 rounded-xl border border-gray-200 dark:border-slate-800 bg-gray-50/50 dark:bg-slate-900/40">
                    <div className="flex items-center justify-between text-xs font-mono font-bold text-red-500 mb-1">
                      <span className="flex items-center gap-1.5">
                        <Heart className="w-3.5 h-3.5" />
                        Heart Rate
                      </span>
                      <span className="text-[10px] text-gray-400 font-normal">20-250 bpm</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        min={20}
                        max={250}
                        value={heartRate}
                        onChange={(e) => setHeartRate(parseFloat(e.target.value) || 0)}
                        className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm font-mono font-bold text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-red-500"
                      />
                      <span className="text-xs font-mono text-gray-400">BPM</span>
                    </div>
                  </div>

                  {/* SpO2 */}
                  <div className="p-3 rounded-xl border border-gray-200 dark:border-slate-800 bg-gray-50/50 dark:bg-slate-900/40">
                    <div className="flex items-center justify-between text-xs font-mono font-bold text-cyan-500 mb-1">
                      <span className="flex items-center gap-1.5">
                        <Droplets className="w-3.5 h-3.5" />
                        Oxygen Saturation
                      </span>
                      <span className="text-[10px] text-gray-400 font-normal">50-100 %</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        min={50}
                        max={100}
                        value={spo2}
                        onChange={(e) => setSpo2(parseFloat(e.target.value) || 0)}
                        className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm font-mono font-bold text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                      />
                      <span className="text-xs font-mono text-gray-400">%</span>
                    </div>
                  </div>

                  {/* Systolic BP */}
                  <div className="p-3 rounded-xl border border-gray-200 dark:border-slate-800 bg-gray-50/50 dark:bg-slate-900/40">
                    <div className="flex items-center justify-between text-xs font-mono font-bold text-blue-500 mb-1">
                      <span className="flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5" />
                        Systolic BP (NIBP)
                      </span>
                      <span className="text-[10px] text-gray-400 font-normal">40-260 mmHg</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        min={40}
                        max={260}
                        value={sbp}
                        onChange={(e) => setSbp(parseFloat(e.target.value) || 0)}
                        className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm font-mono font-bold text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <span className="text-xs font-mono text-gray-400">mmHg</span>
                    </div>
                  </div>

                  {/* Respiratory Rate */}
                  <div className="p-3 rounded-xl border border-gray-200 dark:border-slate-800 bg-gray-50/50 dark:bg-slate-900/40">
                    <div className="flex items-center justify-between text-xs font-mono font-bold text-emerald-500 mb-1">
                      <span className="flex items-center gap-1.5">
                        <Wind className="w-3.5 h-3.5" />
                        Respiratory Rate
                      </span>
                      <span className="text-[10px] text-gray-400 font-normal">4-70 br/min</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        min={4}
                        max={70}
                        value={respiratoryRate}
                        onChange={(e) => setRespiratoryRate(parseFloat(e.target.value) || 0)}
                        className="w-full px-2.5 py-1.5 rounded-lg border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm font-mono font-bold text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                      />
                      <span className="text-xs font-mono text-gray-400">BR/MIN</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Footer */}
          <div className="pt-4 border-t border-gray-100 dark:border-clinical-border flex items-center justify-between">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-mono text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={isLoading || !file}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-mono text-xs font-bold shadow-lg shadow-purple-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing DenseNet-121 & Fusion...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Run Diagnostic Inference & Twin Fusion</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

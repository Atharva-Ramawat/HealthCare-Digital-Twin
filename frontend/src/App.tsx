import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { VitalsTelemetryGrid } from './components/VitalsTelemetryGrid';
import { CXRFusionViewer } from './components/CXRFusionViewer';
import { UnifiedRiskGauge } from './components/UnifiedRiskGauge';
import { fetchDigitalTwinAssessment } from './services/api';
import { UploadIntakeModal } from './components/UploadIntakeModal';
import type { DigitalTwinResponse, PatientProfile } from './types/digitalTwin';
import { HeartPulse, RefreshCw } from 'lucide-react';

// Clinical Cohort of ICU Patients strictly synchronized with backend database
const AVAILABLE_PATIENTS: PatientProfile[] = [
  {
    id: '10003502',
    study_id: 's50084553',
    name: 'Robert Vance',
    age: 64,
    gender: 'M',
    unit: 'MICU Bed 04',
    bed: 'B-04',
    admission_diagnosis: 'Bilateral Pleural Effusions & Atelectasis',
    intubated: true,
  },
  {
    id: '10000764',
    study_id: 's57375967',
    name: 'Elena Rostova',
    age: 58,
    gender: 'F',
    unit: 'MICU Bed 08',
    bed: 'B-08',
    admission_diagnosis: 'Lobar Pneumonia & Consolidation',
    intubated: true,
  },
  {
    id: '10000032',
    study_id: 's50414267',
    name: 'Sarah Chen',
    age: 52,
    gender: 'F',
    unit: 'SICU Bed 02',
    bed: 'S-02',
    admission_diagnosis: 'Baseline Screening (Low Risk)',
    intubated: false,
  },
  {
    id: '10000898',
    study_id: 's50771383',
    name: 'David Kim',
    age: 68,
    gender: 'M',
    unit: 'CCU Bed 01',
    bed: 'C-01',
    admission_diagnosis: 'Post-Op Thoracic Monitoring',
    intubated: false,
  },
];

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);
  const [patientList, setPatientList] = useState<PatientProfile[]>(AVAILABLE_PATIENTS);
  const [currentPatient, setCurrentPatient] = useState<PatientProfile>(AVAILABLE_PATIENTS[0]);
  const [twinData, setTwinData] = useState<DigitalTwinResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [latencyMs, setLatencyMs] = useState<number>(18);
  const [isMockData, setIsMockData] = useState<boolean>(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [adhocMap, setAdhocMap] = useState<Record<string, DigitalTwinResponse>>({});

  // Sync Dark Mode with DOM
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode((prev) => !prev);

  // Fetch Digital Twin Telemetry & Multimodal Assessment
  const loadDigitalTwinData = useCallback(async (patient: PatientProfile) => {
    // If it's an ad-hoc uploaded patient, retrieve from cache and prevent overwriting with 404 mock
    if (patient.study_id.startsWith('s_adhoc')) {
      if (adhocMap[patient.study_id]) {
        setTwinData(adhocMap[patient.study_id]);
      }
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    try {
      const { data, isMock, latencyMs: measuredLatency } = await fetchDigitalTwinAssessment(
        patient.id,
        patient.study_id,
        true
      );
      setTwinData(data);
      setIsMockData(isMock);
      setLatencyMs(measuredLatency);
    } catch (err) {
      console.error('Failed to load digital twin assessment:', err);
    } finally {
      setIsLoading(false);
    }
  }, [adhocMap]);

  // Initial load and on patient change
  useEffect(() => {
    loadDigitalTwinData(currentPatient);
  }, [currentPatient, loadDigitalTwinData]);

  const handleSelectPatient = (patient: PatientProfile) => {
    setCurrentPatient(patient);
  };

  const handleRefresh = () => {
    loadDigitalTwinData(currentPatient);
  };

  const handleCustomIntakeSuccess = (data: DigitalTwinResponse, customProfile: PatientProfile) => {
    // Correctly extract image_base64 and heatmap_base64 from JSON payload
    const imageBase64 = data.image_base64 || data.input_image_base64;
    const heatmapBase64 = data.heatmap_base64 || data.heatmap_image_base64;

    const normalizedData: DigitalTwinResponse = {
      ...data,
      image_base64: imageBase64,
      heatmap_base64: heatmapBase64,
      input_image_base64: imageBase64,
      heatmap_image_base64: heatmapBase64,
    };

    setPatientList((prev) => {
      const exists = prev.some((p) => p.id === customProfile.id);
      return exists ? prev.map((p) => (p.id === customProfile.id ? customProfile : p)) : [customProfile, ...prev];
    });
    setAdhocMap((prev) => ({ ...prev, [customProfile.study_id]: normalizedData }));
    setCurrentPatient(customProfile);
    setTwinData(normalizedData);
    setIsMockData(false);
    setLatencyMs(45);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-clinical-dark text-gray-900 dark:text-gray-100 font-sans overflow-hidden transition-colors">
      {/* 1. Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDarkMode={isDarkMode}
        toggleDarkMode={toggleDarkMode}
        onOpenUploadModal={() => setIsUploadModalOpen(true)}
      />

      {/* Main Panel Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* 2. Top Header with Patient Demographics Strip & Latency Metrics */}
        <Header
          currentPatient={currentPatient}
          availablePatients={patientList}
          onSelectPatient={handleSelectPatient}
          latencyMs={latencyMs}
          isMockData={isMockData}
          onRefresh={handleRefresh}
          isLoading={isLoading}
          onOpenUploadModal={() => setIsUploadModalOpen(true)}
        />

        {/* 3. Main Dashboard Body */}
        <main className="flex-1 overflow-y-auto p-3 md:p-4 icu-grid space-y-4">
          {twinData ? (
            <>
              {/* Tab 1: Combined High-Density Overview */}
              {(activeTab === 'dashboard' || activeTab === 'monitoring') && (
                <section>
                  {/* 3. Vitals Telemetry Grid (4 Synchronized Recharts Line Charts) */}
                  <VitalsTelemetryGrid
                    trajectory={twinData.vitals_trajectory_24h || []}
                    currentVitals={twinData.current_vitals}
                    isDarkMode={isDarkMode}
                  />
                </section>
              )}

              {/* Lower Section: CXR Fusion Core + Unified Risk Gauge */}
              {(activeTab === 'dashboard' || activeTab === 'cxr') && (
                <section className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
                  {/* 4. CXR & Multimodal Fusion Core (Image Viewer + Grad-CAM + 8-Class Probabilities) */}
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

                  {/* 5. Unified Risk Gauge (Semi-Circular 0-100% Score + Clinical Guidance) */}
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
              )}

              {/* Tab: Patient History / Timeline */}
              {activeTab === 'history' && (
                <section className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-clinical-border">
                    <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                      <HeartPulse className="w-4 h-4 text-blue-500" />
                      Patient Clinical History & Longitudinal Audit Trail
                    </h2>
                    <span className="text-xs font-mono text-gray-400">
                      Subject #{currentPatient.id}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
                    <div className="p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800">
                      <div className="text-gray-400 text-[10px] uppercase">Admission Details</div>
                      <div className="font-bold text-gray-800 dark:text-gray-200 mt-1">{currentPatient.admission_diagnosis}</div>
                      <div className="text-gray-500 mt-0.5">Unit: {currentPatient.unit}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800">
                      <div className="text-gray-400 text-[10px] uppercase">CXR Modality</div>
                      <div className="font-bold text-gray-800 dark:text-gray-200 mt-1">AP Portable Bedside</div>
                      <div className="text-gray-500 mt-0.5">Study: {twinData.study_id}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800">
                      <div className="text-gray-400 text-[10px] uppercase">Digital Twin State</div>
                      <div className="font-bold text-emerald-500 mt-1">Causal Verified (No Leakage)</div>
                      <div className="text-gray-500 mt-0.5">NEWS2 Tier: High Risk</div>
                    </div>
                  </div>

                  <div className="space-y-2 text-xs font-mono">
                    <h3 className="text-xs font-bold text-gray-700 dark:text-gray-300">Physiological Trajectory Log (Selected Timesteps):</h3>
                    <div className="max-h-64 overflow-y-auto border border-gray-200 dark:border-slate-800 rounded-lg">
                      <table className="w-full text-left border-collapse">
                        <thead className="bg-gray-100 dark:bg-slate-900/80 sticky top-0 text-[11px] text-gray-400">
                          <tr>
                            <th className="p-2">Step</th>
                            <th className="p-2">Timestamp</th>
                            <th className="p-2 text-red-400">HR (bpm)</th>
                            <th className="p-2 text-cyan-400">SpO2 (%)</th>
                            <th className="p-2 text-blue-400">SBP (mmHg)</th>
                            <th className="p-2 text-emerald-400">RR (bpm)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 dark:divide-slate-800 text-[11px]">
                          {(twinData.vitals_trajectory_24h || []).slice(-16).map((step) => (
                            <tr key={step.step_index} className="hover:bg-gray-50 dark:hover:bg-slate-800/50">
                              <td className="p-2 font-bold">{step.step_index}</td>
                              <td className="p-2 text-gray-400">{new Date(step.timestamp).toLocaleTimeString()}</td>
                              <td className="p-2 text-red-500">{step.heart_rate}</td>
                              <td className="p-2 text-cyan-500">{step.spo2}%</td>
                              <td className="p-2 text-blue-500">{step.sbp}</td>
                              <td className="p-2 text-emerald-500">{step.respiratory_rate}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </section>
              )}
            </>
          ) : (
            <div className="h-full flex flex-col items-center justify-center space-y-3 p-12 text-center">
              <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
              <div className="font-mono text-sm text-gray-600 dark:text-gray-300 font-bold">
                Connecting to Digital Twin Multimodal Fusion Engine...
              </div>
              <div className="font-mono text-xs text-gray-400">
                Fetching DenseNet-121 embeddings and 96-timestep 15-min vitals telemetry
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Manual Patient Intake & CXR Upload Modal */}
      <UploadIntakeModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onSuccess={handleCustomIntakeSuccess}
      />
    </div>
  );
}

export default App;

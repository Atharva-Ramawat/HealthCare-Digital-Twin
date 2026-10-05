import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { CommandCenter } from './components/CommandCenter';
import { TreatmentAnalysis } from './components/TreatmentAnalysis';
import { SandboxView } from './components/SandboxView';
import { VitalsTelemetryGrid } from './components/VitalsTelemetryGrid';
import { CXRFusionViewer } from './components/CXRFusionViewer';
import { UnifiedRiskGauge } from './components/UnifiedRiskGauge';
import { fetchDigitalTwinAssessment } from './services/api';
import type { DigitalTwinResponse, PatientProfile, ActiveNavView } from './types/digitalTwin';
import { HeartPulse, RefreshCw } from 'lucide-react';

// Clinical Cohort of verified MIMIC-CXR ICU Patients
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
  // Primary Navigation View: 'dashboard' | 'ward' | 'treatment' | 'sandbox'
  const [activeView, setActiveView] = useState<ActiveNavView>('dashboard');
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);

  // Live Ward states
  const [wardSubTab, setWardSubTab] = useState<'monitoring' | 'history'>('monitoring');
  const [patientList, setPatientList] = useState<PatientProfile[]>(AVAILABLE_PATIENTS);
  const [currentPatient, setCurrentPatient] = useState<PatientProfile>(AVAILABLE_PATIENTS[0]);
  const [twinData, setTwinData] = useState<DigitalTwinResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [latencyMs, setLatencyMs] = useState<number>(18);
  const [isMockData, setIsMockData] = useState<boolean>(false);

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

  // Fetch Digital Twin Telemetry & Multimodal Assessment for verified cohort patient
  const loadDigitalTwinData = useCallback(async (patient: PatientProfile) => {
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
  }, []);

  // Load when active patient changes
  useEffect(() => {
    loadDigitalTwinData(currentPatient);
  }, [currentPatient, loadDigitalTwinData]);

  const handleSelectPatient = (patient: PatientProfile) => {
    setCurrentPatient(patient);
  };

  const handleRefresh = () => {
    loadDigitalTwinData(currentPatient);
  };

  // Called when clicking an Active Bed or Alert in the Command Center
  const handleSelectPatientFromCommandCenter = (patient: PatientProfile) => {
    const existing = patientList.find((p) => p.id === patient.id);
    if (existing) {
      setCurrentPatient(existing);
    } else {
      setPatientList((prev) => [...prev, patient]);
      setCurrentPatient(patient);
    }
    setActiveView('ward');
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-clinical-dark text-gray-900 dark:text-gray-100 font-sans overflow-hidden transition-colors">
      {/* 1. Redesigned Sidebar Navigation with 4 Primary Views */}
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
        isDarkMode={isDarkMode}
        toggleDarkMode={toggleDarkMode}
      />

      {/* Main Panel Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* VIEW 1: Main Dashboard - ICU Command Center */}
        {activeView === 'dashboard' && (
          <CommandCenter
            onSelectPatient={handleSelectPatientFromCommandCenter}
            isDarkMode={isDarkMode}
          />
        )}

        {/* VIEW 2: Live Ward (Verified MIMIC Database Patients) */}
        {activeView === 'ward' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            {/* Top Header with Patient Demographics Strip & Latency Metrics */}
            <Header
              currentPatient={currentPatient}
              availablePatients={patientList}
              onSelectPatient={handleSelectPatient}
              latencyMs={latencyMs}
              isMockData={isMockData}
              onRefresh={handleRefresh}
              isLoading={isLoading}
              onNavigateToSandbox={() => setActiveView('sandbox')}
            />

            {/* Sub-navigation inside Live Ward */}
            <div className="px-4 py-2 border-b border-gray-200 dark:border-clinical-border bg-white/50 dark:bg-clinical-panel/50 flex items-center justify-between text-xs font-mono">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setWardSubTab('monitoring')}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    wardSubTab === 'monitoring'
                      ? 'bg-blue-600/15 text-blue-500 font-bold border border-blue-500/30'
                      : 'text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'
                  }`}
                >
                  Multimodal Telemetry & Vision
                </button>
                <button
                  onClick={() => setWardSubTab('history')}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    wardSubTab === 'history'
                      ? 'bg-blue-600/15 text-blue-500 font-bold border border-blue-500/30'
                      : 'text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'
                  }`}
                >
                  Longitudinal Audit Trail
                </button>
              </div>

              <div className="text-[11px] text-gray-400">
                Ground-Truth 15-min Intervals (96 Timesteps)
              </div>
            </div>

            {/* Main Live Ward Content */}
            <main className="flex-1 overflow-y-auto p-3 md:p-4 icu-grid space-y-4">
              {twinData ? (
                <>
                  {wardSubTab === 'monitoring' && (
                    <>
                      {/* Vitals Telemetry Grid (4 Synchronized Recharts Line Charts) */}
                      <section>
                        <VitalsTelemetryGrid
                          trajectory={twinData.vitals_trajectory_24h || []}
                          currentVitals={twinData.current_vitals}
                          isDarkMode={isDarkMode}
                        />
                      </section>

                      {/* Lower Section: CXR Fusion Core + Unified Risk Gauge */}
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
                    </>
                  )}

                  {/* Tab: Patient Clinical History / Timeline */}
                  {wardSubTab === 'history' && (
                    <section className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-5 shadow-sm space-y-4">
                      <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-clinical-border">
                        <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                          <HeartPulse className="w-4 h-4 text-blue-500" />
                          Patient Clinical History & Longitudinal Audit Trail
                        </h2>
                        <span className="text-xs font-mono text-gray-400">
                          Subject #{currentPatient.id} • Verified Cohort
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
                          <div className="text-gray-400 text-[10px] uppercase">Ground Truth Reliability</div>
                          <div className="font-bold text-emerald-500 mt-1">PhysioNet MIMIC-IV Verified</div>
                          <div className="text-gray-500 mt-0.5">Risk Tier: {twinData.risk_tier}</div>
                        </div>
                      </div>

                      <div className="space-y-2 text-xs font-mono">
                        <h3 className="text-xs font-bold text-gray-700 dark:text-gray-300">Physiological Trajectory Log (Last 16 Steps):</h3>
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
        )}

        {/* VIEW 3: Longitudinal Treatment Response Tracker */}
        {activeView === 'treatment' && (
          <TreatmentAnalysis />
        )}

        {/* VIEW 4: Ad-Hoc Simulation Sandbox (Isolated from MIMIC) */}
        {activeView === 'sandbox' && (
          <SandboxView isDarkMode={isDarkMode} />
        )}
      </div>
    </div>
  );
}

export default App;

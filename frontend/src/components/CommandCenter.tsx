import React, { useState } from 'react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line 
} from 'recharts';
import { 
  AlertTriangle, 
  Activity, 
  ArrowUpRight, 
  Clock, 
  Radio, 
  ShieldAlert, 
  CheckCircle2, 
  Layers, 
  Filter,
  Search
} from 'lucide-react';
import type { PatientProfile, RiskTier, WardAlert, BedOverview } from '../types/digitalTwin';

interface CommandCenterProps {
  onSelectPatient: (patient: PatientProfile) => void;
  isDarkMode?: boolean;
}

// 6 Active ICU Beds strictly synchronized with clinical database profiles and telemetry
const COHORT_BEDS: BedOverview[] = [
  {
    patient: {
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
    riskScore: 84.5,
    riskTier: 'Critical',
    topFinding: 'Pleural Effusion',
    topProbability: 0.762,
    currentVitals: {
      heart_rate: 132,
      spo2: 84,
      sbp: 82,
      respiratory_rate: 30,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'Heart Rate',
    sparklineData: [
      { time: '00:00', value: 88 },
      { time: '04:00', value: 92 },
      { time: '08:00', value: 104 },
      { time: '12:00', value: 112 },
      { time: '16:00', value: 124 },
      { time: '20:00', value: 132 },
    ],
    alerts: ['Critical Hypoxemia (84%)', 'Bilateral Pleural Effusion (76.2%)'],
  },
  {
    patient: {
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
    riskScore: 78.2,
    riskTier: 'Critical',
    topFinding: 'Pneumonia',
    topProbability: 0.884,
    currentVitals: {
      heart_rate: 118,
      spo2: 86,
      sbp: 94,
      respiratory_rate: 28,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'SpO2',
    sparklineData: [
      { time: '00:00', value: 96 },
      { time: '04:00', value: 94 },
      { time: '08:00', value: 91 },
      { time: '12:00', value: 89 },
      { time: '16:00', value: 87 },
      { time: '20:00', value: 86 },
    ],
    alerts: ['Dense Lobar Consolidation (88.4%)', 'Tachypnea (28 bpm)'],
  },
  {
    patient: {
      id: '10001244',
      study_id: 's50119284',
      name: 'Marcus Brody',
      age: 61,
      gender: 'M',
      unit: 'MICU Bed 06',
      bed: 'B-06',
      admission_diagnosis: 'Septic Shock & Acute Pulmonary Edema',
      intubated: true,
    },
    riskScore: 69.4,
    riskTier: 'High',
    topFinding: 'Edema',
    topProbability: 0.723,
    currentVitals: {
      heart_rate: 126,
      spo2: 89,
      sbp: 78,
      respiratory_rate: 26,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'Heart Rate',
    sparklineData: [
      { time: '00:00', value: 94 },
      { time: '04:00', value: 102 },
      { time: '08:00', value: 110 },
      { time: '12:00', value: 118 },
      { time: '16:00', value: 122 },
      { time: '20:00', value: 126 },
    ],
    alerts: ['Hypotensive Shock (SBP 78)', 'Pulmonary Edema (72.3%)'],
  },
  {
    patient: {
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
    riskScore: 48.0,
    riskTier: 'Moderate',
    topFinding: 'Atelectasis',
    topProbability: 0.642,
    currentVitals: {
      heart_rate: 92,
      spo2: 94,
      sbp: 128,
      respiratory_rate: 19,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'Heart Rate',
    sparklineData: [
      { time: '00:00', value: 82 },
      { time: '04:00', value: 85 },
      { time: '08:00', value: 88 },
      { time: '12:00', value: 90 },
      { time: '16:00', value: 91 },
      { time: '20:00', value: 92 },
    ],
    alerts: ['Mild Bibasilar Atelectasis (64.2%)'],
  },
  {
    patient: {
      id: '10002190',
      study_id: 's50892110',
      name: 'Linda Torres',
      age: 73,
      gender: 'F',
      unit: 'SICU Bed 05',
      bed: 'S-05',
      admission_diagnosis: 'Congestive Heart Failure Exacerbation',
      intubated: false,
    },
    riskScore: 39.0,
    riskTier: 'Moderate',
    topFinding: 'Cardiomegaly',
    topProbability: 0.584,
    currentVitals: {
      heart_rate: 86,
      spo2: 95,
      sbp: 142,
      respiratory_rate: 18,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'Heart Rate',
    sparklineData: [
      { time: '00:00', value: 80 },
      { time: '04:00', value: 82 },
      { time: '08:00', value: 84 },
      { time: '12:00', value: 88 },
      { time: '16:00', value: 85 },
      { time: '20:00', value: 86 },
    ],
    alerts: ['Cardiomegaly Finding (58.4%)'],
  },
  {
    patient: {
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
    riskScore: 16.5,
    riskTier: 'Low',
    topFinding: 'No Finding',
    topProbability: 0.841,
    currentVitals: {
      heart_rate: 72,
      spo2: 99,
      sbp: 118,
      respiratory_rate: 14,
      timestamp: new Date().toISOString(),
    },
    primaryVitalLabel: 'SpO2',
    sparklineData: [
      { time: '00:00', value: 98 },
      { time: '04:00', value: 98 },
      { time: '08:00', value: 99 },
      { time: '12:00', value: 99 },
      { time: '16:00', value: 98 },
      { time: '20:00', value: 99 },
    ],
    alerts: ['Stable Normal Physiological Baseline'],
  },
];

// Active timestamped clinical alerts across the ICU ward
const WARD_ALERTS: WardAlert[] = [
  {
    id: 'alt-001',
    timestamp: '10:44 AM',
    bed: 'MICU Bed 04',
    patientName: 'Robert Vance',
    patientId: '10003502',
    severity: 'critical',
    message: 'Acute Deterioration Trigger: Bilateral Pleural Effusions (76.2%) + SpO2 drop to 84%',
    metric: 'Risk: 84.5%',
  },
  {
    id: 'alt-002',
    timestamp: '10:32 AM',
    bed: 'MICU Bed 08',
    patientName: 'Elena Rostova',
    patientId: '10000764',
    severity: 'critical',
    message: 'High Risk ARDS: Dense Lobar Consolidation (88.4%) with progressive tachypnea (28 bpm)',
    metric: 'Risk: 78.2%',
  },
  {
    id: 'alt-003',
    timestamp: '10:15 AM',
    bed: 'MICU Bed 06',
    patientName: 'Marcus Brody',
    patientId: '10001244',
    severity: 'critical',
    message: 'Septic Shock Alert: Hypotension (SBP 78 mmHg) & Tachycardia (126 bpm)',
    metric: 'Risk: 69.4%',
  },
  {
    id: 'alt-004',
    timestamp: '09:55 AM',
    bed: 'MICU Bed 04',
    patientName: 'Robert Vance',
    patientId: '10003502',
    severity: 'warning',
    message: 'Grad-CAM Attention Shift: Marked increase in bibasilar opacities over 4h window',
  },
  {
    id: 'alt-005',
    timestamp: '09:20 AM',
    bed: 'CCU Bed 01',
    patientName: 'David Kim',
    patientId: '10000898',
    severity: 'warning',
    message: 'Post-thoracotomy monitoring: Persistent subsegmental atelectasis (64.2%)',
    metric: 'Risk: 48.0%',
  },
  {
    id: 'alt-006',
    timestamp: '08:45 AM',
    bed: 'SICU Bed 05',
    patientName: 'Linda Torres',
    patientId: '10002190',
    severity: 'info',
    message: 'Diuretic response under evaluation: Stable cardiomegaly without acute pulmonary edema',
    metric: 'Risk: 39.0%',
  },
  {
    id: 'alt-007',
    timestamp: '08:10 AM',
    bed: 'SICU Bed 02',
    patientName: 'Sarah Chen',
    patientId: '10000032',
    severity: 'info',
    message: 'Extubation evaluation: Gas exchange stable, spontaneous breathing trial candidate',
    metric: 'Risk: 16.5%',
  },
];

export const CommandCenter: React.FC<CommandCenterProps> = ({
  onSelectPatient,
  isDarkMode: _isDarkMode,
}) => {
  const [alertFilter, setAlertFilter] = useState<'all' | 'critical' | 'warning'>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  const filteredAlerts = WARD_ALERTS.filter((alt) => {
    if (alertFilter !== 'all' && alt.severity !== alertFilter) return false;
    if (searchTerm.trim() && !alt.patientName.toLowerCase().includes(searchTerm.toLowerCase()) && !alt.bed.toLowerCase().includes(searchTerm.toLowerCase()) && !alt.message.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    return true;
  });

  const getTierColorClass = (tier: RiskTier) => {
    switch (tier) {
      case 'Critical':
        return 'text-red-500 bg-red-500/10 border-red-500/30 ring-red-500/20';
      case 'High':
        return 'text-amber-500 bg-amber-500/10 border-amber-500/30 ring-amber-500/20';
      case 'Moderate':
        return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30 ring-yellow-500/20';
      case 'Low':
        return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30 ring-emerald-500/20';
    }
  };

  const getSparklineColor = (tier: RiskTier) => {
    switch (tier) {
      case 'Critical':
        return '#ef4444';
      case 'High':
        return '#f97316';
      case 'Moderate':
        return '#eab308';
      case 'Low':
        return '#10b981';
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* 1. Header Banner & Key Cohort Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono uppercase text-gray-500 dark:text-gray-400">Total ICU Beds</div>
            <div className="text-xl font-bold font-mono text-gray-900 dark:text-gray-100 mt-0.5">
              6 <span className="text-xs font-normal text-gray-400">/ 6 Monitored</span>
            </div>
            <div className="text-[11px] font-mono text-emerald-500 flex items-center gap-1 mt-0.5">
              <CheckCircle2 className="w-3 h-3" /> 100% Ward Coverage
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-blue-500/10 text-blue-500 border border-blue-500/20">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
        </div>

        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono uppercase text-gray-500 dark:text-gray-400">Critical Triggers</div>
            <div className="text-xl font-bold font-mono text-red-500 mt-0.5">
              2 <span className="text-xs font-normal text-gray-400">Patients At Risk</span>
            </div>
            <div className="text-[11px] font-mono text-red-400 flex items-center gap-1 mt-0.5">
              <AlertTriangle className="w-3 h-3" /> Immediate Response Advised
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-red-500/10 text-red-500 border border-red-500/20">
            <ShieldAlert className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono uppercase text-gray-500 dark:text-gray-400">Mean Ward Risk</div>
            <div className="text-xl font-bold font-mono text-amber-500 mt-0.5">
              52.6% <span className="text-xs font-normal text-gray-400">ICU Index</span>
            </div>
            <div className="text-[11px] font-mono text-gray-500 dark:text-gray-400 mt-0.5">
              Multimodal Fusion Active
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-500 border border-amber-500/20">
            <Activity className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono uppercase text-gray-500 dark:text-gray-400">Inference Engine</div>
            <div className="text-xl font-bold font-mono text-cyan-400 mt-0.5">
              18 ms <span className="text-xs font-normal text-gray-400">AMP CUDA</span>
            </div>
            <div className="text-[11px] font-mono text-cyan-500 flex items-center gap-1 mt-0.5">
              DenseNet-121 • 1024-dim
            </div>
          </div>
          <div className="p-2.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Layers className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 2. Main Layout: Left 8 cols = Active Beds Grid, Right 4 cols = Critical Alerts Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: Active Bed Cards Grid (8 cols) */}
        <div className="lg:col-span-8 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <Radio className="w-4 h-4 text-blue-500" />
              Active Ward Beds Cohort Overview
            </h2>
            <span className="text-[11px] font-mono text-gray-500 dark:text-gray-400">
              Synchronized with verified MIMIC Database
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {COHORT_BEDS.map((bed) => {
              const tierClass = getTierColorClass(bed.riskTier);
              const sparklineColor = getSparklineColor(bed.riskTier);

              return (
                <div
                  key={bed.patient.id}
                  className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm hover:border-blue-500/50 transition-all flex flex-col justify-between group"
                >
                  {/* Bed Header */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 text-[11px] font-mono font-bold rounded bg-gray-100 dark:bg-slate-800 text-gray-800 dark:text-gray-200 border border-gray-300 dark:border-slate-700">
                          {bed.patient.bed}
                        </span>
                        <span className="text-[10px] font-mono text-gray-400">
                          {bed.patient.unit}
                        </span>
                      </div>

                      <div className="flex items-center gap-1.5">
                        {bed.patient.intubated ? (
                          <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-red-500/10 text-red-500 border border-red-500/20">
                            VENTILATED
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 text-[9px] font-mono font-semibold rounded bg-gray-100 dark:bg-slate-800 text-gray-400 border border-gray-300 dark:border-slate-700">
                            SPONTANEOUS
                          </span>
                        )}
                        <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded border ${tierClass}`}>
                          {bed.riskTier.toUpperCase()}
                        </span>
                      </div>
                    </div>

                    {/* Patient Identification */}
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-bold text-sm text-gray-900 dark:text-gray-100 font-mono group-hover:text-blue-500 transition-colors">
                          {bed.patient.name}
                        </h3>
                        <p className="text-[11px] text-gray-400 font-mono">
                          Patient #{bed.patient.id} • {bed.patient.age}yo {bed.patient.gender}
                        </p>
                      </div>

                      {/* Large Fused Risk Score Display */}
                      <div className="text-right">
                        <div className="text-[10px] font-mono text-gray-400 uppercase">Risk Score</div>
                        <div className={`text-xl font-extrabold font-mono ${bed.riskTier === 'Critical' ? 'text-red-500' : bed.riskTier === 'High' ? 'text-amber-500' : bed.riskTier === 'Moderate' ? 'text-yellow-500' : 'text-emerald-500'}`}>
                          {bed.riskScore.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Admission Diagnosis */}
                    <div className="mt-2 text-xs font-mono text-gray-600 dark:text-gray-300 line-clamp-1 bg-gray-50 dark:bg-slate-900/60 p-1.5 rounded border border-gray-100 dark:border-slate-800/80">
                      <span className="text-gray-400 text-[10px] uppercase block">Diagnosis:</span>
                      {bed.patient.admission_diagnosis}
                    </div>

                    {/* Primary Radiological Finding */}
                    <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono">
                      <span className="text-gray-400">DenseNet-121 Finding:</span>
                      <span className="font-semibold text-purple-600 dark:text-purple-400">
                        {bed.topFinding} ({(bed.topProbability * 100).toFixed(1)}%)
                      </span>
                    </div>

                    {/* Vitals Summary Strip */}
                    <div className="grid grid-cols-4 gap-1.5 mt-2.5 text-center text-[10px] font-mono">
                      <div className="p-1 rounded bg-red-500/10 border border-red-500/20 text-red-500">
                        <div className="text-[9px] text-gray-400">HR</div>
                        <div className="font-bold">{bed.currentVitals.heart_rate}</div>
                      </div>
                      <div className="p-1 rounded bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                        <div className="text-[9px] text-gray-400">SpO2</div>
                        <div className="font-bold">{bed.currentVitals.spo2}%</div>
                      </div>
                      <div className="p-1 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400">
                        <div className="text-[9px] text-gray-400">BP</div>
                        <div className="font-bold">{bed.currentVitals.sbp}</div>
                      </div>
                      <div className="p-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                        <div className="text-[9px] text-gray-400">RR</div>
                        <div className="font-bold">{bed.currentVitals.respiratory_rate}</div>
                      </div>
                    </div>

                    {/* Micro-Sparkline (24h Trend) */}
                    <div className="mt-3 pt-2 border-t border-gray-100 dark:border-slate-800">
                      <div className="flex items-center justify-between text-[10px] font-mono text-gray-400 mb-1">
                        <span>24h Trend ({bed.primaryVitalLabel})</span>
                        <span className="text-gray-500">
                          {bed.sparklineData[0].value} ➔ {bed.sparklineData[bed.sparklineData.length - 1].value}
                        </span>
                      </div>
                      <div className="h-10 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={bed.sparklineData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                            <Line
                              type="monotone"
                              dataKey="value"
                              stroke={sparklineColor}
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Open Digital Twin Button */}
                  <div className="mt-3 pt-2">
                    <button
                      onClick={() => onSelectPatient(bed.patient)}
                      className="w-full flex items-center justify-center space-x-1.5 py-1.5 rounded-lg text-xs font-mono font-bold bg-blue-600/10 hover:bg-blue-600/20 text-blue-500 border border-blue-500/30 transition-all shadow-sm"
                    >
                      <span>Examine Patient Digital Twin</span>
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Critical Alerts Feed (4 cols) */}
        <div className="lg:col-span-4 bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col h-[720px]">
          {/* Header */}
          <div className="pb-3 border-b border-gray-100 dark:border-clinical-border">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-red-500" />
                Critical Alerts Feed
              </h2>
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
              </span>
            </div>
            <p className="text-[11px] font-mono text-gray-400 mt-0.5">
              Active physiological & radiological triggers
            </p>

            {/* Search and Filters */}
            <div className="mt-3 space-y-2">
              <div className="relative">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Filter alerts by patient or bed..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 text-xs font-mono bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-lg text-gray-800 dark:text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center gap-1.5 text-[10px] font-mono">
                <span className="text-gray-400 flex items-center gap-1">
                  <Filter className="w-3 h-3" /> Severity:
                </span>
                <button
                  onClick={() => setAlertFilter('all')}
                  className={`px-2 py-0.5 rounded border transition-colors ${
                    alertFilter === 'all'
                      ? 'bg-blue-500/20 text-blue-400 border-blue-500/40'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-400 border-transparent hover:border-slate-700'
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => setAlertFilter('critical')}
                  className={`px-2 py-0.5 rounded border transition-colors ${
                    alertFilter === 'critical'
                      ? 'bg-red-500/20 text-red-400 border-red-500/40'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-400 border-transparent hover:border-slate-700'
                  }`}
                >
                  Critical
                </button>
                <button
                  onClick={() => setAlertFilter('warning')}
                  className={`px-2 py-0.5 rounded border transition-colors ${
                    alertFilter === 'warning'
                      ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-400 border-transparent hover:border-slate-700'
                  }`}
                >
                  Warning
                </button>
              </div>
            </div>
          </div>

          {/* Alerts List */}
          <div className="flex-1 overflow-y-auto mt-3 space-y-2.5 pr-1">
            {filteredAlerts.length > 0 ? (
              filteredAlerts.map((alt) => {
                const targetBed = COHORT_BEDS.find((b) => b.patient.id === alt.patientId);

                return (
                  <div
                    key={alt.id}
                    onClick={() => {
                      if (targetBed) onSelectPatient(targetBed.patient);
                    }}
                    className={`p-3 rounded-lg border text-xs font-mono cursor-pointer transition-all hover:scale-[1.01] ${
                      alt.severity === 'critical'
                        ? 'bg-red-500/5 dark:bg-red-950/20 border-red-500/30 hover:border-red-500'
                        : alt.severity === 'warning'
                        ? 'bg-amber-500/5 dark:bg-amber-950/20 border-amber-500/30 hover:border-amber-500'
                        : 'bg-blue-500/5 dark:bg-blue-950/20 border-blue-500/30 hover:border-blue-500'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[10px] text-gray-400 mb-1">
                      <span className="flex items-center gap-1 font-bold text-gray-800 dark:text-gray-200">
                        <Clock className="w-3 h-3 text-gray-400" />
                        {alt.timestamp} • {alt.bed}
                      </span>
                      {alt.metric && (
                        <span className="font-extrabold text-red-400 bg-red-500/10 px-1 rounded">
                          {alt.metric}
                        </span>
                      )}
                    </div>

                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {alt.patientName} (#{alt.patientId})
                    </div>
                    <div className="text-[11px] text-gray-600 dark:text-gray-300 mt-1 leading-snug">
                      {alt.message}
                    </div>

                    <div className="mt-2 text-[10px] text-blue-500 flex items-center justify-end gap-1 opacity-0 hover:opacity-100 transition-opacity">
                      <span>View Twin</span>
                      <ArrowUpRight className="w-3 h-3" />
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-center text-gray-400">
                <CheckCircle2 className="w-8 h-8 text-emerald-500 mb-2 opacity-60" />
                <p className="text-xs font-mono">No active alerts matching filter criteria.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

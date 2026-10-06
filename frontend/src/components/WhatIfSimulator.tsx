import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from 'recharts';
import {
  SlidersHorizontal,
  TrendingUp,
  RotateCcw,
  Sparkles,
  ShieldAlert,
  Activity,
  Wind,
  Droplets,
  Heart,
} from 'lucide-react';
import type { WhatIfVitals, RiskTier } from '../types/digitalTwin';

interface WhatIfSimulatorProps {
  isDarkMode?: boolean;
}

interface PresetScenario {
  id: string;
  name: string;
  description: string;
  vitals: WhatIfVitals;
}

const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'baseline',
    name: 'Normal Homeostasis',
    description: 'Physiologically stable, normal respiratory mechanics and hemodynamics.',
    vitals: { heartRate: 76, spo2: 98, sbp: 122, respiratoryRate: 16 },
  },
  {
    id: 'acute_ards',
    name: 'Acute Hypoxemia (ARDS)',
    description: 'Rapid alveolar collapse with refractory desaturation and severe tachypnea.',
    vitals: { heartRate: 124, spo2: 83, sbp: 108, respiratoryRate: 32 },
  },
  {
    id: 'septic_shock',
    name: 'Distributive Septic Shock',
    description: 'Profound vasodilation, refractory hypotension, compensatory tachycardia.',
    vitals: { heartRate: 138, spo2: 88, sbp: 74, respiratoryRate: 28 },
  },
  {
    id: 'hypercapnic',
    name: 'Exacerbation (Hypercapnic)',
    description: 'Severe respiratory muscle fatigue with rising ventilatory demand.',
    vitals: { heartRate: 106, spo2: 86, sbp: 145, respiratoryRate: 34 },
  },
  {
    id: 'hemodynamic',
    name: 'Cardiogenic Instability',
    description: 'Low cardiac output syndrome with hypoperfusion and pulmonary congestion.',
    vitals: { heartRate: 142, spo2: 90, sbp: 68, respiratoryRate: 26 },
  },
];

export const WhatIfSimulator: React.FC<WhatIfSimulatorProps> = ({ isDarkMode = true }) => {
  // Current Adjustable Vitals
  const [vitals, setVitals] = useState<WhatIfVitals>({
    heartRate: 98,
    spo2: 91,
    sbp: 104,
    respiratoryRate: 24,
  });

  // Selected Active Preset
  const [activePreset, setActivePreset] = useState<string>('custom');

  // Simulated Counterfactual Interventions
  const [interventions, setInterventions] = useState<{
    nivSupport: boolean;
    vasopressor: boolean;
    sedation: boolean;
  }>({
    nivSupport: false,
    vasopressor: false,
    sedation: false,
  });

  const handleVitalChange = (field: keyof WhatIfVitals, value: number) => {
    setVitals((prev) => ({ ...prev, [field]: value }));
    setActivePreset('custom');
  };

  const handleSelectPreset = (preset: PresetScenario) => {
    setVitals({ ...preset.vitals });
    setActivePreset(preset.id);
  };

  const handleReset = () => {
    setVitals({
      heartRate: 76,
      spo2: 98,
      sbp: 122,
      respiratoryRate: 16,
    });
    setActivePreset('baseline');
    setInterventions({ nivSupport: false, vasopressor: false, sedation: false });
  };

  // Trajectory Modeling Function
  const trajectoryData = useMemo(() => {
    // 1. Calculate Base Decompensation Points from vitals
    let rawRisk = 15; // baseline low risk

    // SpO2 Impact
    if (vitals.spo2 < 85) rawRisk += 45;
    else if (vitals.spo2 < 90) rawRisk += 32;
    else if (vitals.spo2 < 93) rawRisk += 18;
    else if (vitals.spo2 < 95) rawRisk += 8;

    // Heart Rate Impact
    if (vitals.heartRate > 140) rawRisk += 30;
    else if (vitals.heartRate > 120) rawRisk += 20;
    else if (vitals.heartRate > 100) rawRisk += 10;
    else if (vitals.heartRate < 50) rawRisk += 25;

    // Blood Pressure (SBP) Impact
    if (vitals.sbp < 80) rawRisk += 35;
    else if (vitals.sbp < 90) rawRisk += 22;
    else if (vitals.sbp < 100) rawRisk += 12;
    else if (vitals.sbp > 180) rawRisk += 18;

    // Respiratory Rate Impact
    if (vitals.respiratoryRate > 32) rawRisk += 32;
    else if (vitals.respiratoryRate > 26) rawRisk += 20;
    else if (vitals.respiratoryRate > 20) rawRisk += 10;
    else if (vitals.respiratoryRate < 10) rawRisk += 25;

    const baseRisk = Math.min(Math.max(Math.round(rawRisk), 5), 98);

    // Compute trajectory slope based on severity
    const escalationRate = baseRisk > 55 ? 1.18 : baseRisk > 35 ? 1.08 : 0.96;

    // Counterfactual intervention impact
    let interventionDelta = 0;
    if (interventions.nivSupport) interventionDelta -= 22;
    if (interventions.vasopressor) interventionDelta -= 18;
    if (interventions.sedation) interventionDelta -= 12;

    const computeHorizonRisk = (hours: number, applyIntervention: boolean) => {
      let risk = baseRisk;
      if (hours === 1) risk = Math.round(baseRisk * (1 + (escalationRate - 1) * 0.45));
      if (hours === 3) risk = Math.round(baseRisk * (1 + (escalationRate - 1) * 1.15));
      if (hours === 6) risk = Math.round(baseRisk * (1 + (escalationRate - 1) * 1.95));

      if (applyIntervention) {
        // Interventions take effect over time (stronger effect at 3h and 6h)
        const timeFactor = hours === 0 ? 0 : hours === 1 ? 0.6 : hours === 3 ? 0.9 : 1.0;
        risk = Math.round(risk + interventionDelta * timeFactor);
      }

      return Math.min(Math.max(risk, 4), 99);
    };

    return [
      {
        horizon: '0h (Current)',
        hours: 0,
        unmanagedRisk: baseRisk,
        counterfactualRisk: computeHorizonRisk(0, true),
      },
      {
        horizon: '1h Horizon',
        hours: 1,
        unmanagedRisk: computeHorizonRisk(1, false),
        counterfactualRisk: computeHorizonRisk(1, true),
      },
      {
        horizon: '3h Horizon',
        hours: 3,
        unmanagedRisk: computeHorizonRisk(3, false),
        counterfactualRisk: computeHorizonRisk(3, true),
      },
      {
        horizon: '6h Horizon',
        hours: 6,
        unmanagedRisk: computeHorizonRisk(6, false),
        counterfactualRisk: computeHorizonRisk(6, true),
      },
    ];
  }, [vitals, interventions]);

  const currentScore = trajectoryData[0].unmanagedRisk;
  const projected6hScore = trajectoryData[3].unmanagedRisk;
  const counterfactual6hScore = trajectoryData[3].counterfactualRisk;

  const currentTier: RiskTier =
    currentScore >= 70 ? 'Critical' : currentScore >= 45 ? 'High' : currentScore >= 25 ? 'Moderate' : 'Low';

  const projectedTier: RiskTier =
    projected6hScore >= 70 ? 'Critical' : projected6hScore >= 45 ? 'High' : projected6hScore >= 25 ? 'Moderate' : 'Low';

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto bg-gray-50 dark:bg-clinical-dark p-4 md:p-6 space-y-5">
      {/* Top Banner & Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-gray-200 dark:border-clinical-border gap-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-500 border border-blue-500/30">
              <SlidersHorizontal className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold font-mono tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-2">
                What-If Physiological Trajectory Simulator
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">
                  Forecasting Engine
                </span>
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-mono mt-0.5">
                Simulate dynamic physiological perturbations and counterfactual clinical interventions across 1h, 3h, and 6h horizons.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-clinical-border text-xs font-mono text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-clinical-card transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Homeostasis
          </button>
        </div>
      </div>

      {/* Preset Scenario Selectors */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm">
        <div className="text-xs font-mono uppercase tracking-wider text-gray-400 mb-2.5 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-blue-400" />
          Clinical Perturbation Scenarios
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {PRESET_SCENARIOS.map((scenario) => {
            const isSelected = activePreset === scenario.id;
            return (
              <button
                key={scenario.id}
                onClick={() => handleSelectPreset(scenario)}
                className={`text-left p-2.5 rounded-lg border transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500/10 text-blue-400 shadow-sm'
                    : 'border-gray-200 dark:border-slate-800 hover:border-gray-400 dark:hover:border-slate-700 bg-gray-50 dark:bg-slate-900/60 text-gray-700 dark:text-gray-300'
                }`}
              >
                <div className="text-xs font-bold font-mono tracking-tight">{scenario.name}</div>
                <div className="text-[10px] text-gray-400 line-clamp-1 mt-0.5">{scenario.description}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Interactive Controls & Forecasting Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Sliders & Intervention Toggles (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-5">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <span className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-200 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-cyan-500" />
                Physiological Parameter Sliders
              </span>
              <span className="text-[10px] font-mono text-gray-400">Continuous Inputs</span>
            </div>

            {/* Slider 1: Heart Rate */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-red-500 font-bold">
                  <Heart className="w-3.5 h-3.5" />
                  Heart Rate (HR)
                </span>
                <span className="font-bold text-gray-900 dark:text-gray-100 text-sm">
                  {vitals.heartRate} <span className="text-[10px] text-gray-400">bpm</span>
                </span>
              </div>
              <input
                type="range"
                min="40"
                max="180"
                step="1"
                value={vitals.heartRate}
                onChange={(e) => handleVitalChange('heartRate', Number(e.target.value))}
                className="w-full accent-red-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-gray-400">
                <span>40 Bradycardia</span>
                <span className="text-emerald-500">Normal (60-100)</span>
                <span>180 Tachycardia</span>
              </div>
            </div>

            {/* Slider 2: SpO2 */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-cyan-500 font-bold">
                  <Wind className="w-3.5 h-3.5" />
                  Oxygen Saturation (SpO2)
                </span>
                <span className="font-bold text-gray-900 dark:text-gray-100 text-sm">
                  {vitals.spo2} <span className="text-[10px] text-gray-400">%</span>
                </span>
              </div>
              <input
                type="range"
                min="70"
                max="100"
                step="1"
                value={vitals.spo2}
                onChange={(e) => handleVitalChange('spo2', Number(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-gray-400">
                <span className="text-red-400">70 Hypoxia</span>
                <span className="text-amber-400">90 Caution</span>
                <span className="text-emerald-500">100 Optimal</span>
              </div>
            </div>

            {/* Slider 3: SBP */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-blue-500 font-bold">
                  <Droplets className="w-3.5 h-3.5" />
                  Systolic Blood Pressure (SBP)
                </span>
                <span className="font-bold text-gray-900 dark:text-gray-100 text-sm">
                  {vitals.sbp} <span className="text-[10px] text-gray-400">mmHg</span>
                </span>
              </div>
              <input
                type="range"
                min="60"
                max="200"
                step="1"
                value={vitals.sbp}
                onChange={(e) => handleVitalChange('sbp', Number(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-gray-400">
                <span className="text-red-400">60 Shock</span>
                <span className="text-emerald-500">120 Eutensive</span>
                <span className="text-amber-400">200 Hypertensive</span>
              </div>
            </div>

            {/* Slider 4: Respiratory Rate */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-emerald-500 font-bold">
                  <Activity className="w-3.5 h-3.5" />
                  Respiratory Rate (RR)
                </span>
                <span className="font-bold text-gray-900 dark:text-gray-100 text-sm">
                  {vitals.respiratoryRate} <span className="text-[10px] text-gray-400">bpm</span>
                </span>
              </div>
              <input
                type="range"
                min="8"
                max="44"
                step="1"
                value={vitals.respiratoryRate}
                onChange={(e) => handleVitalChange('respiratoryRate', Number(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-gray-200 dark:bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-gray-400">
                <span>8 Hypoventilation</span>
                <span className="text-emerald-500">16 Normal</span>
                <span className="text-red-400">44 Tachypnea</span>
              </div>
            </div>
          </div>

          {/* Counterfactual Intervention Checkboxes */}
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <span className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-200 flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-emerald-500" />
                Counterfactual Intervention Simulation
              </span>
              <span className="text-[10px] font-mono text-gray-400">What-If Branch</span>
            </div>

            <div className="space-y-2">
              <label className="flex items-start gap-2.5 p-2 rounded-lg border border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-slate-900/40 cursor-pointer hover:border-emerald-500/40">
                <input
                  type="checkbox"
                  checked={interventions.nivSupport}
                  onChange={(e) => setInterventions((prev) => ({ ...prev, nivSupport: e.target.checked }))}
                  className="mt-0.5 accent-emerald-500"
                />
                <div className="text-xs font-mono">
                  <div className="font-bold text-gray-800 dark:text-gray-200">Non-Invasive Ventilation (NIV / CPAP)</div>
                  <div className="text-[10px] text-gray-400">Alveolar recruitment, +6% SpO2 delta, ventilatory offloading.</div>
                </div>
              </label>

              <label className="flex items-start gap-2.5 p-2 rounded-lg border border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-slate-900/40 cursor-pointer hover:border-blue-500/40">
                <input
                  type="checkbox"
                  checked={interventions.vasopressor}
                  onChange={(e) => setInterventions((prev) => ({ ...prev, vasopressor: e.target.checked }))}
                  className="mt-0.5 accent-blue-500"
                />
                <div className="text-xs font-mono">
                  <div className="font-bold text-gray-800 dark:text-gray-200">Vasopressor Infusion (Norepinephrine)</div>
                  <div className="text-[10px] text-gray-400">Restores SVR and MAP, elevates SBP by +20 mmHg.</div>
                </div>
              </label>

              <label className="flex items-start gap-2.5 p-2 rounded-lg border border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-slate-900/40 cursor-pointer hover:border-purple-500/40">
                <input
                  type="checkbox"
                  checked={interventions.sedation}
                  onChange={(e) => setInterventions((prev) => ({ ...prev, sedation: e.target.checked }))}
                  className="mt-0.5 accent-purple-500"
                />
                <div className="text-xs font-mono">
                  <div className="font-bold text-gray-800 dark:text-gray-200">Targeted Sedation / Anxiolysis</div>
                  <div className="text-[10px] text-gray-400">Blunts autonomic drive, reduces sympathetic tachycardia (-15 bpm).</div>
                </div>
              </label>
            </div>
          </div>
        </div>

        {/* Right Column: Recharts LineChart & Projected Trajectory Analysis (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Risk Metrics Summary Strip */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3.5 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
              <div className="text-[10px] font-mono uppercase text-gray-400">Current Risk (0h)</div>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100">{currentScore}%</span>
                <span
                  className={`text-[11px] font-mono px-1.5 py-0.5 rounded font-bold ${
                    currentTier === 'Critical'
                      ? 'bg-red-500/20 text-red-500'
                      : currentTier === 'High'
                      ? 'bg-amber-500/20 text-amber-500'
                      : currentTier === 'Moderate'
                      ? 'bg-blue-500/20 text-blue-500'
                      : 'bg-emerald-500/20 text-emerald-500'
                  }`}
                >
                  {currentTier}
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
              <div className="text-[10px] font-mono uppercase text-gray-400">Projected 6h (Unmanaged)</div>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-2xl font-bold font-mono text-red-500">{projected6hScore}%</span>
                <span className="text-[11px] font-mono text-gray-400">
                  Δ {projected6hScore - currentScore >= 0 ? `+${projected6hScore - currentScore}%` : `${projected6hScore - currentScore}%`}
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
              <div className="text-[10px] font-mono uppercase text-gray-400">With Interventions (6h)</div>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-2xl font-bold font-mono text-emerald-500">{counterfactual6hScore}%</span>
                <span className="text-[11px] font-mono text-emerald-400">
                  Δ {counterfactual6hScore - projected6hScore}%
                </span>
              </div>
            </div>
          </div>

          {/* Recharts Trajectory Line Chart */}
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <div>
                <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-500" />
                  Multi-Horizon Deterioration Trajectory (0h → 1h → 3h → 6h)
                </h3>
                <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Comparative projection: Baseline Natural Course vs Counterfactual Managed Path
                </p>
              </div>

              <div className="text-[11px] font-mono text-gray-400">
                P(Escalation): <span className="font-bold text-gray-800 dark:text-gray-200">{Math.min(projected6hScore + 4, 99)}%</span>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trajectoryData} margin={{ top: 10, right: 25, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#1e293b' : '#e2e8f0'} />
                  <XAxis
                    dataKey="horizon"
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 11, fontFamily: 'monospace' }}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDarkMode ? '#0f172a' : '#ffffff',
                      borderColor: isDarkMode ? '#334155' : '#cbd5e1',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      borderRadius: '8px',
                    }}
                    formatter={(val: any) => [`${val}%`, '']}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '8px' }}
                  />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Critical (70%)', fill: '#ef4444', fontSize: 10 }} />
                  <ReferenceLine y={45} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'High (45%)', fill: '#f59e0b', fontSize: 10 }} />

                  {/* Unmanaged Natural Path */}
                  <Line
                    type="monotone"
                    dataKey="unmanagedRisk"
                    name="Natural Progression"
                    stroke="#ef4444"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#ef4444' }}
                    activeDot={{ r: 6 }}
                  />

                  {/* Counterfactual Path */}
                  <Line
                    type="monotone"
                    dataKey="counterfactualRisk"
                    name="Counterfactual Intervention Path"
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={{ r: 4, fill: '#10b981' }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Clinical Alert & Interpretation */}
            <div className="p-3 rounded-lg border border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-slate-900/60 flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div className="text-xs font-mono space-y-1">
                <div className="font-bold text-gray-800 dark:text-gray-200">
                  Clinical Forecast Synthesis: {projectedTier} Trajectory
                </div>
                <div className="text-gray-500 dark:text-gray-400">
                  {currentScore >= 70
                    ? 'Patient displays overt respiratory or circulatory failure. Unmanaged trajectory indicates high likelihood of urgent intubation or resuscitation within 1-3 hours.'
                    : currentScore >= 45
                    ? 'Elevated physiological volatility detected. Targeted ventilatory and hemodynamic stabilization reduces 6-hour risk progression significantly.'
                    : 'Hemodynamic stability maintained. Vital perturbations remain within compensatory limits with low probability of acute escalation.'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

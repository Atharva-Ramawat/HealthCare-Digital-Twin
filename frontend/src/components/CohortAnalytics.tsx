import React from 'react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import {
  PieChart as PieIcon,
  BarChart3,
  Users,
  AlertTriangle,
  TrendingDown,
  Activity,
  BedDouble,
  CheckCircle,
} from 'lucide-react';
import type { PathologyDistributionItem, DailyAlertFrequency } from '../types/digitalTwin';

interface CohortAnalyticsProps {
  isDarkMode?: boolean;
}

// Pathology Distribution Mock Data across 48 Ward Cases
const PATHOLOGY_DATA: PathologyDistributionItem[] = [
  { name: 'Pneumonia', count: 14, percentage: 29.2, color: '#ef4444' },
  { name: 'Pulmonary Edema', count: 11, percentage: 22.9, color: '#f59e0b' },
  { name: 'Atelectasis', count: 9, percentage: 18.8, color: '#3b82f6' },
  { name: 'Pleural Effusion', count: 7, percentage: 14.6, color: '#06b6d4' },
  { name: 'Cardiomegaly', count: 4, percentage: 8.3, color: '#a855f7' },
  { name: 'Pneumothorax', count: 3, percentage: 6.2, color: '#ec4899' },
];

// Daily Critical Alert Frequencies over the Last 7 Days
const DAILY_ALERTS: DailyAlertFrequency[] = [
  { date: 'Mon Sep 29', critical: 6, warning: 14, info: 22, total: 42 },
  { date: 'Tue Sep 30', critical: 8, warning: 17, info: 19, total: 44 },
  { date: 'Wed Oct 01', critical: 5, warning: 12, info: 25, total: 42 },
  { date: 'Thu Oct 02', critical: 9, warning: 19, info: 18, total: 46 },
  { date: 'Fri Oct 03', critical: 7, warning: 15, info: 21, total: 43 },
  { date: 'Sat Oct 04', critical: 4, warning: 10, info: 16, total: 30 },
  { date: 'Sun Oct 05', critical: 3, warning: 8, info: 14, total: 25 },
];

// ICU Units Breakdown Data
const UNIT_BREAKDOWN = [
  { unit: 'MICU (Medical ICU)', bedsOccupied: 11, totalBeds: 12, avgRisk: 48.4, criticalCount: 3 },
  { unit: 'SICU (Surgical ICU)', bedsOccupied: 7, totalBeds: 8, avgRisk: 34.2, criticalCount: 1 },
  { unit: 'CCU (Cardiac Care Unit)', bedsOccupied: 5, totalBeds: 6, avgRisk: 42.1, criticalCount: 1 },
  { unit: 'Neuro-ICU', bedsOccupied: 4, totalBeds: 6, avgRisk: 28.6, criticalCount: 0 },
];

export const CohortAnalytics: React.FC<CohortAnalyticsProps> = ({ isDarkMode = true }) => {
  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto bg-gray-50 dark:bg-clinical-dark p-4 md:p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-gray-200 dark:border-clinical-border gap-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/30">
            <PieIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-mono tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-2">
              Ward Cohort Analytics & Epidemiological Trends
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                Macro-Level Surveillance
              </span>
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono mt-0.5">
              Population-level pathology distribution, daily alarm frequencies, and bed occupancy telemetry.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-3 py-1 rounded bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-slate-700 flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5 text-blue-400" />
            Active Cohort: <strong className="text-gray-900 dark:text-gray-100">27 Patients</strong>
          </span>
        </div>
      </div>

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="p-4 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
          <div className="flex items-center justify-between text-gray-400 text-[10px] font-mono uppercase">
            <span>Ward Occupancy</span>
            <BedDouble className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100">84.4%</span>
            <span className="text-xs font-mono text-cyan-400">27 / 32 Beds</span>
          </div>
          <div className="text-[10px] text-gray-400 font-mono mt-1">5 available step-down beds</div>
        </div>

        <div className="p-4 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
          <div className="flex items-center justify-between text-gray-400 text-[10px] font-mono uppercase">
            <span>Mean Deterioration Risk</span>
            <Activity className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-amber-500">41.8%</span>
            <span className="text-xs font-mono text-emerald-400 flex items-center">
              <TrendingDown className="w-3 h-3 mr-0.5" /> -3.2%
            </span>
          </div>
          <div className="text-[10px] text-gray-400 font-mono mt-1">Stabilizing across medical ICU</div>
        </div>

        <div className="p-4 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
          <div className="flex items-center justify-between text-gray-400 text-[10px] font-mono uppercase">
            <span>Critical Tier Patients</span>
            <AlertTriangle className="w-4 h-4 text-red-500" />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-red-500">5 Beds</span>
            <span className="text-xs font-mono text-gray-400">+ 8 High</span>
          </div>
          <div className="text-[10px] text-gray-400 font-mono mt-1">Priority surveillance queue</div>
        </div>

        <div className="p-4 rounded-xl border border-gray-200 dark:border-clinical-border bg-white dark:bg-clinical-panel">
          <div className="flex items-center justify-between text-gray-400 text-[10px] font-mono uppercase">
            <span>Telemetry Health</span>
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-emerald-500">100%</span>
            <span className="text-xs font-mono text-gray-400">Zero packet loss</span>
          </div>
          <div className="text-[10px] text-gray-400 font-mono mt-1">28 MIMIC ingestion streams active</div>
        </div>
      </div>

      {/* Main Charts: Pathology Pie on Left, 7-Day Alert Frequencies on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Recharts PieChart (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <div>
                <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <PieIcon className="w-4 h-4 text-emerald-500" />
                  Pathology Distribution in Monitored Ward
                </h3>
                <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Proportion of radiographic findings classified by DenseNet-121
                </p>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={PATHOLOGY_DATA}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="count"
                    label={(entry: any) => `${entry.name} (${entry.payload?.percentage}%)`}
                    labelLine={false}
                  >
                    {PATHOLOGY_DATA.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDarkMode ? '#0f172a' : '#ffffff',
                      borderColor: isDarkMode ? '#334155' : '#cbd5e1',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      borderRadius: '8px',
                    }}
                    formatter={(val: any, name: any, item: any) => [
                      `${val} cases (${item.payload.percentage}%)`,
                      name,
                    ]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Pathology legend tags */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-100 dark:border-slate-800 text-xs font-mono">
              {PATHOLOGY_DATA.map((item) => (
                <div key={item.name} className="flex items-center justify-between p-1.5 rounded bg-gray-50 dark:bg-slate-900/60 border border-gray-200 dark:border-slate-800">
                  <div className="flex items-center gap-1.5 truncate">
                    <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }}></span>
                    <span className="truncate text-gray-700 dark:text-gray-300">{item.name}</span>
                  </div>
                  <span className="font-bold text-gray-900 dark:text-gray-100">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Recharts BarChart of Daily Critical Alerts (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-gray-100 dark:border-slate-800">
              <div>
                <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-blue-500" />
                  Daily Alert Frequencies (Last 7 Days)
                </h3>
                <p className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Severity breakdown of automated physiological alarms across all ICU units
                </p>
              </div>

              <div className="flex items-center gap-2 text-[10px] font-mono">
                <span className="flex items-center gap-1 text-red-400">
                  <span className="w-2 h-2 rounded-full bg-red-500"></span> Critical
                </span>
                <span className="flex items-center gap-1 text-amber-400">
                  <span className="w-2 h-2 rounded-full bg-amber-500"></span> Warning
                </span>
                <span className="flex items-center gap-1 text-blue-400">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span> Info
                </span>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={DAILY_ALERTS} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#1e293b' : '#e2e8f0'} />
                  <XAxis
                    dataKey="date"
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <YAxis
                    stroke={isDarkMode ? '#94a3b8' : '#64748b'}
                    tick={{ fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDarkMode ? '#0f172a' : '#ffffff',
                      borderColor: isDarkMode ? '#334155' : '#cbd5e1',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="critical" name="Critical Alert" fill="#ef4444" stackId="a" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="warning" name="Warning" fill="#f59e0b" stackId="a" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="info" name="Clinical Info" fill="#3b82f6" stackId="a" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Unit Breakdown Mini-Table */}
            <div className="border border-gray-200 dark:border-slate-800 rounded-lg overflow-hidden text-xs font-mono">
              <table className="w-full text-left">
                <thead className="bg-gray-100 dark:bg-slate-900/80 text-[10px] text-gray-400 uppercase">
                  <tr>
                    <th className="p-2">Unit</th>
                    <th className="p-2">Occupancy</th>
                    <th className="p-2">Mean Risk</th>
                    <th className="p-2 text-right">Critical Beds</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-slate-800 text-[11px]">
                  {UNIT_BREAKDOWN.map((u) => (
                    <tr key={u.unit} className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                      <td className="p-2 font-bold text-gray-800 dark:text-gray-200">{u.unit}</td>
                      <td className="p-2 text-gray-400">
                        {u.bedsOccupied} / {u.totalBeds} ({Math.round((u.bedsOccupied / u.totalBeds) * 100)}%)
                      </td>
                      <td className="p-2">
                        <span
                          className={`font-bold ${
                            u.avgRisk > 45 ? 'text-amber-500' : u.avgRisk > 30 ? 'text-blue-400' : 'text-emerald-400'
                          }`}
                        >
                          {u.avgRisk}%
                        </span>
                      </td>
                      <td className="p-2 text-right font-bold text-red-400">{u.criticalCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

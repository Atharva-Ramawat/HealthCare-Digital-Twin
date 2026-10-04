import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { Heart, Activity, Droplets, Wind } from 'lucide-react';
import type { VitalReadingItem, VitalSignSnapshot } from '../types/digitalTwin';

interface VitalsTelemetryGridProps {
  trajectory?: VitalReadingItem[] | null;
  currentVitals?: VitalSignSnapshot | null;
  isDarkMode: boolean;
}

export const VitalsTelemetrySkeleton: React.FC<{ isDarkMode?: boolean }> = () => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400/60 animate-pulse" />
          <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-500 dark:text-gray-400">
            Synchronized Vitals Telemetry (Buffering 24h Trajectory...)
          </h2>
        </div>
        <span className="text-[11px] font-mono text-gray-400 animate-pulse">
          Resolution: 15-min intervals
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {[
          { label: 'Heart Rate (HR)', unit: 'BPM', icon: Heart, color: 'text-red-500/50', badge: 'BUFFERING' },
          { label: 'Oxygen Saturation (SpO₂)', unit: '%', icon: Droplets, color: 'text-cyan-500/50', badge: 'BUFFERING' },
          { label: 'Systolic BP (NIBP)', unit: 'mmHg', icon: Activity, color: 'text-blue-500/50', badge: 'BUFFERING' },
          { label: 'Respiratory Rate (RR)', unit: 'BR/MIN', icon: Wind, color: 'text-emerald-500/50', badge: 'BUFFERING' },
        ].map((item, idx) => {
          const Icon = item.icon;
          return (
            <div
              key={idx}
              className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm space-y-3"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2">
                  <div className="p-1.5 rounded-lg bg-gray-100 dark:bg-slate-800 text-gray-400">
                    <Icon className={`w-4 h-4 ${item.color} animate-pulse`} />
                  </div>
                  <div>
                    <div className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300 uppercase tracking-tight">
                      {item.label}
                    </div>
                    <div className="h-2.5 w-28 bg-gray-200 dark:bg-slate-800 rounded animate-pulse mt-1" />
                  </div>
                </div>

                <div className="text-right flex flex-col items-end space-y-1">
                  <div className="flex items-baseline space-x-1 justify-end">
                    <div className="h-7 w-12 bg-gray-200 dark:bg-slate-800 rounded animate-pulse" />
                    <span className="text-[10px] font-mono text-gray-400 font-semibold">{item.unit}</span>
                  </div>
                  <span className="inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-gray-100 dark:bg-slate-800 text-gray-400">
                    {item.badge}
                  </span>
                </div>
              </div>

              {/* Shimmer Placeholder for Recharts AreaChart */}
              <div className="h-28 w-full bg-gray-50/80 dark:bg-slate-900/40 rounded-lg flex items-end justify-between px-3 py-2 space-x-1 border border-dashed border-gray-200 dark:border-slate-800 overflow-hidden">
                {Array.from({ length: 24 }).map((_, barIdx) => {
                  const h = 25 + Math.sin(barIdx * 0.4) * 20 + ((barIdx % 3) * 10);
                  return (
                    <div
                      key={barIdx}
                      className="flex-1 bg-gray-200 dark:bg-slate-800 rounded-t transition-all"
                      style={{
                        height: `${Math.max(15, Math.min(95, h))}%`,
                        opacity: 0.3 + (barIdx % 5) * 0.1,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const VitalsTelemetryGrid: React.FC<VitalsTelemetryGridProps> = ({
  trajectory,
  currentVitals,
  isDarkMode,
}) => {
  // Defensive null-checks & skeleton rendering if telemetry is absent, null, or empty
  if (!trajectory || !Array.isArray(trajectory) || trajectory.length === 0 || !currentVitals) {
    return <VitalsTelemetrySkeleton isDarkMode={isDarkMode} />;
  }

  // Format data for Recharts
  const chartData = trajectory.map((reading) => {
    const time = new Date(reading.timestamp);
    const timeLabel = !isNaN(time.getTime())
      ? time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
      : `T-${reading.step_index}`;
    return {
      step: reading.step_index,
      time: timeLabel,
      hr: reading.heart_rate,
      spo2: reading.spo2,
      sbp: reading.sbp,
      rr: reading.respiratory_rate,
    };
  });

  // Calculate 24h Min & Max for quick bedside reference defensively
  const hrVals = trajectory.map((t) => t.heart_rate);
  const spo2Vals = trajectory.map((t) => t.spo2);
  const sbpVals = trajectory.map((t) => t.sbp);
  const rrVals = trajectory.map((t) => t.respiratory_rate);

  const hrStats = { min: hrVals.length ? Math.min(...hrVals) : 0, max: hrVals.length ? Math.max(...hrVals) : 0 };
  const spo2Stats = { min: spo2Vals.length ? Math.min(...spo2Vals) : 0, max: spo2Vals.length ? Math.max(...spo2Vals) : 0 };
  const sbpStats = { min: sbpVals.length ? Math.min(...sbpVals) : 0, max: sbpVals.length ? Math.max(...sbpVals) : 0 };
  const rrStats = { min: rrVals.length ? Math.min(...rrVals) : 0, max: rrVals.length ? Math.max(...rrVals) : 0 };

  // Status evaluators
  const getHrStatus = (v: number) => {
    if (v >= 130) return { label: 'CRITICAL HIGH', color: 'bg-red-500/20 text-red-400 border-red-500/40' };
    if (v >= 100) return { label: 'TACHYCARDIA', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    if (v < 50) return { label: 'BRADYCARDIA', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    return { label: 'NORMAL', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' };
  };

  const getSpo2Status = (v: number) => {
    if (v <= 88) return { label: 'SEVERE HYPOXIA', color: 'bg-red-500/20 text-red-400 border-red-500/40' };
    if (v < 92) return { label: 'DESATURATION', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    return { label: 'OPTIMAL', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' };
  };

  const getSbpStatus = (v: number) => {
    if (v <= 85) return { label: 'SHOCK / HYPOTENSION', color: 'bg-red-500/20 text-red-400 border-red-500/40' };
    if (v < 95) return { label: 'BORDERLINE LOW', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    if (v > 160) return { label: 'HYPERTENSIVE', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    return { label: 'NORMOTENSIVE', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' };
  };

  const getRrStatus = (v: number) => {
    if (v >= 28) return { label: 'TACHYPNEA CRITICAL', color: 'bg-red-500/20 text-red-400 border-red-500/40' };
    if (v >= 22) return { label: 'ELEVATED RR', color: 'bg-amber-500/20 text-amber-400 border-amber-500/40' };
    return { label: 'NORMAL', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' };
  };

  const hrStatus = getHrStatus(currentVitals.heart_rate);
  const spo2Status = getSpo2Status(currentVitals.spo2);
  const sbpStatus = getSbpStatus(currentVitals.sbp);
  const rrStatus = getRrStatus(currentVitals.respiratory_rate);

  // Custom high-density tooltip for Recharts
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0];
      return (
        <div className="bg-gray-900/95 dark:bg-slate-950/95 border border-slate-700 p-2 rounded shadow-2xl text-[11px] font-mono text-white pointer-events-none z-50">
          <div className="text-gray-400 pb-1 border-b border-slate-800">{item.payload.time} (T-{95 - item.payload.step} * 15m)</div>
          <div className="pt-1 flex items-center gap-1.5 font-bold" style={{ color: item.color }}>
            <span>{item.name}:</span>
            <span>{item.value}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  const gridLineColor = isDarkMode ? 'rgba(51, 65, 85, 0.4)' : 'rgba(226, 232, 240, 0.8)';
  const tickColor = isDarkMode ? '#64748b' : '#94a3b8';

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
          Synchronized Vitals Telemetry (Last 24 Hours • 96 Timesteps)
        </h2>
        <span className="text-[11px] font-mono text-gray-500 dark:text-gray-400">
          Resolution: 15-min intervals
        </span>
      </div>

      {/* Grid of 4 Synchronized Vitals Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* 1. HEART RATE (RED) */}
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-lg bg-red-500/10 text-red-500">
                <Heart className="w-4 h-4 animate-pulse text-red-500" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-gray-800 dark:text-gray-200 uppercase tracking-tight">
                  Heart Rate (HR)
                </div>
                <div className="text-[10px] font-mono text-gray-400">
                  24h Min: {hrStats.min} • Max: {hrStats.max}
                </div>
              </div>
            </div>

            <div className="text-right">
              <div className="flex items-baseline space-x-1 justify-end">
                <span className="text-2xl font-black font-mono tracking-tight text-red-500 dark:text-red-400">
                  {Math.round(currentVitals.heart_rate)}
                </span>
                <span className="text-[10px] font-mono text-gray-400 font-semibold">BPM</span>
              </div>
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border ${hrStatus.color}`}>
                {hrStatus.label}
              </span>
            </div>
          </div>

          <div className="h-28 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} syncId="vitals-sync" margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                <defs>
                  <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[40, 160]} tick={{ fontSize: 9, fill: tickColor }} stroke={gridLineColor} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" opacity={0.6} />
                <ReferenceLine y={60} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />
                <Area type="monotone" dataKey="hr" name="HR (bpm)" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#hrGrad)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. SpO2 (CYAN) */}
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-500">
                <Droplets className="w-4 h-4 text-cyan-400" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-gray-800 dark:text-gray-200 uppercase tracking-tight">
                  Oxygen Saturation (SpO₂)
                </div>
                <div className="text-[10px] font-mono text-gray-400">
                  24h Min: {spo2Stats.min}% • Max: {spo2Stats.max}%
                </div>
              </div>
            </div>

            <div className="text-right">
              <div className="flex items-baseline space-x-1 justify-end">
                <span className="text-2xl font-black font-mono tracking-tight text-cyan-500 dark:text-cyan-400">
                  {Math.round(currentVitals.spo2)}
                </span>
                <span className="text-[10px] font-mono text-gray-400 font-semibold">%</span>
              </div>
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border ${spo2Status.color}`}>
                {spo2Status.label}
              </span>
            </div>
          </div>

          <div className="h-28 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} syncId="vitals-sync" margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                <defs>
                  <linearGradient id="spo2Grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[70, 100]} tick={{ fontSize: 9, fill: tickColor }} stroke={gridLineColor} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={90} stroke="#ef4444" strokeDasharray="3 3" opacity={0.7} label={{ value: 'Hypoxia', fill: '#ef4444', fontSize: 8, position: 'insideTopLeft' }} />
                <Area type="monotone" dataKey="spo2" name="SpO2 (%)" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#spo2Grad)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 3. NIBP / SBP (BLUE) */}
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-500">
                <Activity className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-gray-800 dark:text-gray-200 uppercase tracking-tight">
                  Systolic BP (NIBP)
                </div>
                <div className="text-[10px] font-mono text-gray-400">
                  24h Min: {sbpStats.min} • Max: {sbpStats.max}
                </div>
              </div>
            </div>

            <div className="text-right">
              <div className="flex items-baseline space-x-1 justify-end">
                <span className="text-2xl font-black font-mono tracking-tight text-blue-500 dark:text-blue-400">
                  {Math.round(currentVitals.sbp)}
                </span>
                <span className="text-[10px] font-mono text-gray-400 font-semibold">mmHg</span>
              </div>
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border ${sbpStatus.color}`}>
                {sbpStatus.label}
              </span>
            </div>
          </div>

          <div className="h-28 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} syncId="vitals-sync" margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                <defs>
                  <linearGradient id="sbpGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[60, 160]} tick={{ fontSize: 9, fill: tickColor }} stroke={gridLineColor} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={90} stroke="#ef4444" strokeDasharray="3 3" opacity={0.6} label={{ value: 'Shock <90', fill: '#ef4444', fontSize: 8, position: 'insideTopLeft' }} />
                <Area type="monotone" dataKey="sbp" name="SBP (mmHg)" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#sbpGrad)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. RESPIRATORY RATE (GREEN) */}
        <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-500">
                <Wind className="w-4 h-4 text-emerald-400" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-gray-800 dark:text-gray-200 uppercase tracking-tight">
                  Respiratory Rate (RR)
                </div>
                <div className="text-[10px] font-mono text-gray-400">
                  24h Min: {rrStats.min} • Max: {rrStats.max}
                </div>
              </div>
            </div>

            <div className="text-right">
              <div className="flex items-baseline space-x-1 justify-end">
                <span className="text-2xl font-black font-mono tracking-tight text-emerald-500 dark:text-emerald-400">
                  {Math.round(currentVitals.respiratory_rate)}
                </span>
                <span className="text-[10px] font-mono text-gray-400 font-semibold">BR/MIN</span>
              </div>
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border ${rrStatus.color}`}>
                {rrStatus.label}
              </span>
            </div>
          </div>

          <div className="h-28 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} syncId="vitals-sync" margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                <defs>
                  <linearGradient id="rrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide />
                <YAxis domain={[8, 36]} tick={{ fontSize: 9, fill: tickColor }} stroke={gridLineColor} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={24} stroke="#ef4444" strokeDasharray="3 3" opacity={0.6} label={{ value: 'Tachypnea >24', fill: '#ef4444', fontSize: 8, position: 'insideTopLeft' }} />
                <Area type="monotone" dataKey="rr" name="RR (br/min)" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#rrGrad)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

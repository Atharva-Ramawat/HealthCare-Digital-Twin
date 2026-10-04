import { 
  ShieldAlert, 
  AlertTriangle, 
  Zap, 
  Activity,
  Layers
} from 'lucide-react';
import type { RiskTier } from '../types/digitalTwin';

interface UnifiedRiskGaugeProps {
  score: number;
  tier: RiskTier;
  visualContribution: number;
  vitalsContribution: number;
  riskFactors: string[];
  recommendation: string;
}

export const UnifiedRiskGauge: React.FC<UnifiedRiskGaugeProps> = ({
  score,
  tier,
  visualContribution,
  vitalsContribution,
  riskFactors,
  recommendation,
}) => {
  // Dynamic color coding based on risk score (0-100%)
  // Green < 25%, Yellow 25-50%, Orange 50-75%, Red > 75%
  const getTheme = (val: number) => {
    if (val >= 75) {
      return {
        color: '#ef4444',
        name: 'CRITICAL',
        bgColor: 'bg-red-500/10',
        borderColor: 'border-red-500/30',
        textColor: 'text-red-500 dark:text-red-400',
        badgeColor: 'bg-red-500 text-white',
        pulse: true,
      };
    }
    if (val >= 50) {
      return {
        color: '#f97316',
        name: 'HIGH RISK',
        bgColor: 'bg-orange-500/10',
        borderColor: 'border-orange-500/30',
        textColor: 'text-orange-500 dark:text-orange-400',
        badgeColor: 'bg-orange-500 text-white',
        pulse: false,
      };
    }
    if (val >= 25) {
      return {
        color: '#eab308',
        name: 'MODERATE',
        bgColor: 'bg-yellow-500/10',
        borderColor: 'border-yellow-500/30',
        textColor: 'text-yellow-500 dark:text-yellow-400',
        badgeColor: 'bg-yellow-500 text-black',
        pulse: false,
      };
    }
    return {
      color: '#22c55e',
      name: 'LOW RISK',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/30',
      textColor: 'text-emerald-500 dark:text-emerald-400',
      badgeColor: 'bg-emerald-500 text-white',
      pulse: false,
    };
  };

  const theme = getTheme(score);

  // SVG Gauge calculations (Semi-circle radius = 80, circumference = pi * r = 251.32)
  const radius = 80;
  const strokeWidth = 14;
  const arcLength = Math.PI * radius; // 251.32
  const progressOffset = arcLength - (Math.min(Math.max(score, 0), 100) / 100) * arcLength;

  return (
    <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-4 shadow-sm flex flex-col justify-between h-full">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-clinical-border">
        <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-gray-800 dark:text-gray-200 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-red-500" />
          Unified ICU Deterioration Risk Score
        </h2>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider ${theme.badgeColor} ${theme.pulse ? 'animate-pulse' : ''}`}>
          {tier}
        </span>
      </div>

      {/* Central Semi-Circular Gauge */}
      <div className="relative flex flex-col items-center justify-center my-2">
        <div className="relative w-56 h-32 flex items-center justify-center overflow-hidden">
          <svg className="w-56 h-56 transform -rotate-180" viewBox="0 0 200 200">
            {/* Background Arc Tracks with Clinical Color Segments */}
            {/* 1. Green Segment (0-25%) */}
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke="#22c55e"
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength * 0.25} ${arcLength * 0.75}`}
              strokeDashoffset="0"
              opacity="0.25"
            />
            {/* 2. Yellow Segment (25-50%) */}
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke="#eab308"
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength * 0.25} ${arcLength * 0.75}`}
              strokeDashoffset={`-${arcLength * 0.25}`}
              opacity="0.25"
            />
            {/* 3. Orange Segment (50-75%) */}
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke="#f97316"
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength * 0.25} ${arcLength * 0.75}`}
              strokeDashoffset={`-${arcLength * 0.50}`}
              opacity="0.25"
            />
            {/* 4. Red Segment (75-100%) */}
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke="#ef4444"
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength * 0.25} ${arcLength * 0.75}`}
              strokeDashoffset={`-${arcLength * 0.75}`}
              opacity="0.25"
            />

            {/* Dynamic Value Arc */}
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke={theme.color}
              strokeWidth={strokeWidth}
              strokeDasharray={arcLength}
              strokeDashoffset={progressOffset}
              strokeLinecap="round"
              className="transition-all duration-700 ease-out"
            />
          </svg>

          {/* Central Score Text Overlay */}
          <div className="absolute top-14 flex flex-col items-center justify-center text-center">
            <span className={`text-4xl font-black font-mono tracking-tight ${theme.textColor}`}>
              {score.toFixed(1)}%
            </span>
            <span className="text-[10px] font-mono text-gray-400 font-semibold uppercase tracking-wider mt-0.5">
              Deterioration Probability
            </span>
          </div>
        </div>

        {/* 4-Zone Scale Indicator Bar */}
        <div className="w-full max-w-xs flex justify-between text-[10px] font-mono text-gray-400 px-4 -mt-2">
          <span className="text-emerald-500 font-semibold">0% Low</span>
          <span className="text-yellow-500 font-semibold">25%</span>
          <span className="text-orange-500 font-semibold">50%</span>
          <span className="text-red-500 font-semibold">75% Critical</span>
        </div>
      </div>

      {/* Multimodal Contribution Breakdown */}
      <div className="bg-gray-50 dark:bg-slate-900/60 rounded-lg p-2.5 border border-gray-200 dark:border-slate-800 text-xs font-mono my-2">
        <div className="flex items-center justify-between text-[11px] text-gray-500 dark:text-gray-400 mb-1.5">
          <span className="flex items-center gap-1">
            <Layers className="w-3 h-3 text-purple-400" />
            Visual Findings: {visualContribution.toFixed(0)}%
          </span>
          <span className="flex items-center gap-1">
            <Activity className="w-3 h-3 text-cyan-400" />
            Vitals Dynamics: {vitalsContribution.toFixed(0)}%
          </span>
        </div>
        {/* Dual Progress Bar */}
        <div className="w-full h-2 rounded-full overflow-hidden flex bg-gray-200 dark:bg-slate-800">
          <div
            className="h-full bg-purple-500 transition-all duration-500"
            style={{ width: `${visualContribution}%` }}
            title={`Visual contribution: ${visualContribution}%`}
          />
          <div
            className="h-full bg-cyan-500 transition-all duration-500"
            style={{ width: `${vitalsContribution}%` }}
            title={`Vitals contribution: ${vitalsContribution}%`}
          />
        </div>
      </div>

      {/* Active Clinical Deterioration Triggers */}
      <div className="my-1">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-400 mb-1.5 flex items-center gap-1">
          <Zap className="w-3 h-3 text-amber-400" />
          Active Physiological & Radiographic Triggers
        </div>
        <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
          {riskFactors.map((factor, idx) => (
            <span
              key={idx}
              className="text-[10px] font-mono px-2 py-0.5 rounded bg-gray-100 dark:bg-slate-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-slate-700 flex items-center gap-1"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
              {factor}
            </span>
          ))}
        </div>
      </div>

      {/* Clinical Recommendation Action Box */}
      <div className={`mt-2 p-2.5 rounded-lg border text-xs font-mono ${theme.bgColor} ${theme.borderColor}`}>
        <div className={`font-bold flex items-center gap-1.5 mb-1 ${theme.textColor}`}>
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>CLINICAL PROTOCOL GUIDELINE</span>
        </div>
        <p className="text-[11px] text-gray-800 dark:text-gray-200 leading-relaxed">
          {recommendation}
        </p>
      </div>
    </div>
  );
};

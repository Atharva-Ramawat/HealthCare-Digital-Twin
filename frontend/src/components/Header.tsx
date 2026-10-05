import React, { useState, useEffect } from 'react';
import { 
  User, 
  Cpu, 
  Clock, 
  RefreshCw, 
  ChevronDown,
  Bed,
  CheckCircle2, 
  Wifi,
  FlaskConical,
  Database
} from 'lucide-react';
import type { PatientProfile } from '../types/digitalTwin';

interface HeaderProps {
  currentPatient: PatientProfile;
  availablePatients: PatientProfile[];
  onSelectPatient: (patient: PatientProfile) => void;
  latencyMs: number;
  isMockData: boolean;
  onRefresh: () => void;
  isLoading: boolean;
  onNavigateToSandbox?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentPatient,
  availablePatients,
  onSelectPatient,
  latencyMs,
  isMockData,
  onRefresh,
  isLoading,
  onNavigateToSandbox,
}) => {
  const [currentTime, setCurrentTime] = useState<string>('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setCurrentTime(
        now.toISOString().substring(11, 19) + ' UTC'
      );
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-white dark:bg-clinical-panel border-b border-gray-200 dark:border-clinical-border px-4 md:px-6 flex items-center justify-between transition-colors z-20 select-none">
      {/* Left: Patient Demographics Strip */}
      <div className="flex items-center space-x-3 md:space-x-4">
        {/* Patient Selector Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-slate-800/80 hover:bg-gray-200 dark:hover:bg-slate-700/80 border border-gray-300 dark:border-slate-700 text-xs font-mono transition-colors"
          >
            <User className="w-3.5 h-3.5 text-blue-500" />
            <span className="font-bold text-gray-900 dark:text-gray-100">
              Patient #{currentPatient.id}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
          </button>

          {/* Dropdown Menu */}
          {isDropdownOpen && (
            <div className="absolute left-0 mt-2 w-72 bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-700 rounded-lg shadow-xl py-1 z-50">
              <div className="px-3 py-1.5 text-[10px] font-mono uppercase text-gray-400 border-b border-gray-100 dark:border-slate-800 flex items-center justify-between">
                <span>Verified MIMIC ICU Patients</span>
                <Database className="w-3 h-3 text-emerald-500" />
              </div>
              {availablePatients.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    onSelectPatient(p);
                    setIsDropdownOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-xs font-mono flex items-center justify-between hover:bg-blue-50 dark:hover:bg-slate-800/80 transition-colors ${
                    p.id === currentPatient.id ? 'bg-blue-50/80 dark:bg-blue-900/20 text-blue-500' : 'text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <div>
                    <div className="font-bold">{p.name} (#{p.id})</div>
                    <div className="text-[10px] text-gray-400">{p.unit} • {p.age}yo {p.gender} • Study #{p.study_id}</div>
                  </div>
                  {p.id === currentPatient.id && <CheckCircle2 className="w-3.5 h-3.5 text-blue-500" />}
                </button>
              ))}

              {/* Jump to Sandbox option */}
              {onNavigateToSandbox && (
                <div className="p-1 border-t border-gray-100 dark:border-slate-800">
                  <button
                    onClick={() => {
                      setIsDropdownOpen(false);
                      onNavigateToSandbox();
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded text-xs font-mono text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-950/40 flex items-center space-x-2 font-bold transition-colors"
                  >
                    <FlaskConical className="w-3.5 h-3.5 text-purple-500" />
                    <span>Open Ad-Hoc Sandbox</span>
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Verified Cohort Badge */}
        <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-500 text-[11px] font-mono font-bold">
          <Database className="w-3 h-3" />
          <span>Verified MIMIC Database</span>
        </div>

        {/* Demographics Badges */}
        <div className="hidden lg:flex items-center space-x-2 text-xs font-mono text-gray-600 dark:text-gray-300">
          <span className="text-gray-400">•</span>
          <span className="px-2 py-0.5 rounded text-[11px] bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/30 font-bold">
            Study #{currentPatient.study_id}
          </span>
          <span className="text-gray-400">•</span>
          <span className="flex items-center gap-1 font-semibold text-gray-900 dark:text-gray-100">
            <Bed className="w-3.5 h-3.5 text-purple-400" />
            {currentPatient.unit} ({currentPatient.bed})
          </span>
          <span className="text-gray-400">•</span>
          <span>{currentPatient.age}yo {currentPatient.gender}</span>
          <span className="text-gray-400">•</span>
          <span className="px-2 py-0.5 rounded text-[11px] bg-amber-500/10 text-amber-500 dark:text-amber-400 border border-amber-500/30">
            {currentPatient.admission_diagnosis}
          </span>
          {currentPatient.intubated && (
            <span className="px-2 py-0.5 rounded text-[11px] bg-red-500/10 text-red-500 border border-red-500/30 animate-pulse font-bold">
              Ventilated
            </span>
          )}
        </div>
      </div>

      {/* Right: Latency & Telemetry Status Metrics */}
      <div className="flex items-center space-x-3 md:space-x-4">
        {/* Latency Pill */}
        <div className="hidden sm:flex items-center space-x-2 px-2.5 py-1 rounded-lg bg-gray-100 dark:bg-slate-800/80 border border-gray-300 dark:border-slate-700 text-xs font-mono">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-gray-500 dark:text-gray-400">DenseNet-121:</span>
          <span className="font-bold text-gray-900 dark:text-gray-100">
            {latencyMs}ms
          </span>
        </div>

        {/* Real-time Clock */}
        <div className="hidden md:flex items-center space-x-1.5 text-xs font-mono text-gray-500 dark:text-gray-400">
          <Clock className="w-3.5 h-3.5" />
          <span>{currentTime || '00:00:00 UTC'}</span>
        </div>

        {/* Live Pulse Indicator */}
        <div className="flex items-center space-x-1.5 px-2 py-1 rounded bg-green-500/10 border border-green-500/20 text-green-500 text-[11px] font-mono">
          <Wifi className="w-3 h-3" />
          <span className="hidden sm:inline">{isMockData ? 'DATABASE RECOVERY' : 'LIVE TELEMETRY'}</span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-2 rounded-lg bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 text-gray-700 dark:text-gray-300 transition-colors disabled:opacity-50"
          title="Refresh Multimodal Assessment"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-blue-500' : ''}`} />
        </button>
      </div>
    </header>
  );
};

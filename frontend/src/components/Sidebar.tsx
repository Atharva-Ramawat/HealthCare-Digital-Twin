import { 
  Activity, 
  LayoutDashboard, 
  Radio, 
  Scan, 
  History, 
  Moon, 
  Sun,
  UploadCloud
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  onOpenUploadModal?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  isDarkMode,
  toggleDarkMode,
  onOpenUploadModal,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'monitoring', label: 'Live Monitoring', icon: Radio },
    { id: 'cxr', label: 'CXR Analysis', icon: Scan },
    { id: 'history', label: 'Patient History', icon: History },
  ];

  return (
    <aside className="w-16 md:w-60 bg-white dark:bg-clinical-panel border-r border-gray-200 dark:border-clinical-border flex flex-col justify-between transition-colors duration-200 z-30 select-none">
      {/* Brand Header */}
      <div>
        <div className="h-16 flex items-center px-4 border-b border-gray-200 dark:border-clinical-border">
          <div className="flex items-center space-x-3">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500">
              <Activity className="w-6 h-6 animate-pulse" />
              <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
              </span>
            </div>
            <div className="hidden md:block">
              <h1 className="text-sm font-bold tracking-wide uppercase text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                PULSE-DT
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  v2.0
                </span>
              </h1>
              <p className="text-[11px] text-gray-500 dark:text-gray-400 font-mono">ICU Twin Engine</p>
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600/10 dark:bg-blue-500/15 text-blue-600 dark:text-blue-400 border border-blue-500/30 shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800/60 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
                title={item.label}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-blue-500' : ''}`} />
                <span className="hidden md:inline font-mono tracking-tight">{item.label}</span>
                {isActive && (
                  <span className="hidden md:inline-block ml-auto w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                )}
              </button>
            );
          })}

          {/* Quick Action: New Patient Intake */}
          <div className="pt-3 border-t border-gray-100 dark:border-slate-800">
            <button
              onClick={onOpenUploadModal}
              className="w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-mono font-bold bg-gradient-to-r from-purple-600/10 to-indigo-600/10 hover:from-purple-600/20 hover:to-indigo-600/20 text-purple-600 dark:text-purple-400 border border-purple-500/30 transition-all shadow-sm"
              title="Manual Patient Intake & CXR Upload"
            >
              <UploadCloud className="w-5 h-5 flex-shrink-0 text-purple-500" />
              <span className="hidden md:inline tracking-tight">Intake / Upload</span>
            </button>
          </div>
        </nav>
      </div>

      {/* Footer Controls & Mode Toggle */}
      <div className="p-3 border-t border-gray-200 dark:border-clinical-border space-y-2">
        <div className="hidden md:block px-3 py-2 rounded-lg bg-gray-50 dark:bg-slate-900/50 border border-gray-200 dark:border-slate-800 text-[11px]">
          <div className="flex items-center justify-between text-gray-500 dark:text-gray-400 font-mono">
            <span>DEVICE</span>
            <span className="text-emerald-500 font-semibold">CUDA GPU</span>
          </div>
          <div className="flex items-center justify-between text-gray-500 dark:text-gray-400 font-mono mt-1">
            <span>MODEL</span>
            <span className="text-gray-700 dark:text-gray-300">DenseNet-121</span>
          </div>
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleDarkMode}
          className="w-full flex items-center justify-center md:justify-start space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800/60 transition-colors"
          title="Toggle Light / Dark Mode"
        >
          {isDarkMode ? (
            <>
              <Sun className="w-5 h-5 text-amber-400" />
              <span className="hidden md:inline font-mono">Light Mode</span>
            </>
          ) : (
            <>
              <Moon className="w-5 h-5 text-indigo-500" />
              <span className="hidden md:inline font-mono">Dark Mode</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};

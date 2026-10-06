import React, { useState, useMemo } from 'react';
import {
  Archive,
  Search,
  Filter,
  Download,
  User,
  ChevronRight,
  X,
  FileText,
} from 'lucide-react';
import type { ArchivedPatientRecord } from '../types/digitalTwin';

interface PatientArchiveProps {
  isDarkMode?: boolean;
}

const MOCK_ARCHIVE_DATA: ArchivedPatientRecord[] = [
  {
    id: '10041289',
    mrn: 'MRN-849102',
    name: 'Walter Hayes',
    age: 72,
    gender: 'M',
    admissionDate: '2026-09-18',
    dischargeDate: '2026-09-26',
    lengthOfStayDays: 8.2,
    primaryDiagnosis: 'Severe Bacterial Pneumonia with ARDS',
    primaryCXRFinding: 'Multilobar Consolidation',
    peakRiskScore: 84,
    finalRiskScore: 28,
    dischargeDisposition: 'Step-down Unit',
    icuUnit: 'MICU Bed 03',
  },
  {
    id: '10038812',
    mrn: 'MRN-671239',
    name: 'Grace Morrison',
    age: 63,
    gender: 'F',
    admissionDate: '2026-09-20',
    dischargeDate: '2026-09-25',
    lengthOfStayDays: 5.1,
    primaryDiagnosis: 'Acute Cardiogenic Pulmonary Edema',
    primaryCXRFinding: 'Diffuse Interstitial Edema',
    peakRiskScore: 78,
    finalRiskScore: 19,
    dischargeDisposition: 'Discharged Home',
    icuUnit: 'CCU Bed 02',
  },
  {
    id: '10029941',
    mrn: 'MRN-551093',
    name: 'Arthur Pendelton',
    age: 79,
    gender: 'M',
    admissionDate: '2026-09-12',
    dischargeDate: '2026-09-22',
    lengthOfStayDays: 10.4,
    primaryDiagnosis: 'Refractory Septic Shock & Hypoxemia',
    primaryCXRFinding: 'Bilateral Pleural Effusion',
    peakRiskScore: 94,
    finalRiskScore: 88,
    dischargeDisposition: 'Deceased',
    icuUnit: 'MICU Bed 01',
  },
  {
    id: '10031024',
    mrn: 'MRN-902144',
    name: 'Clara Sterling',
    age: 54,
    gender: 'F',
    admissionDate: '2026-09-22',
    dischargeDate: '2026-09-27',
    lengthOfStayDays: 4.8,
    primaryDiagnosis: 'Post-Operative Thoracotomy Monitoring',
    primaryCXRFinding: 'Basilar Atelectasis',
    peakRiskScore: 46,
    finalRiskScore: 14,
    dischargeDisposition: 'Discharged Home',
    icuUnit: 'SICU Bed 05',
  },
  {
    id: '10045582',
    mrn: 'MRN-190342',
    name: 'Samuel Thorne',
    age: 67,
    gender: 'M',
    admissionDate: '2026-09-15',
    dischargeDate: '2026-09-28',
    lengthOfStayDays: 13.0,
    primaryDiagnosis: 'Aspiration Pneumonitis / Tracheostomy',
    primaryCXRFinding: 'Dense Right Infiltrate',
    peakRiskScore: 82,
    finalRiskScore: 41,
    dischargeDisposition: 'Long-term Care',
    icuUnit: 'MICU Bed 06',
  },
  {
    id: '10018843',
    mrn: 'MRN-784012',
    name: 'Beatrice Vance',
    age: 59,
    gender: 'F',
    admissionDate: '2026-09-24',
    dischargeDate: '2026-09-29',
    lengthOfStayDays: 5.2,
    primaryDiagnosis: 'Acute Exacerbation of COPD',
    primaryCXRFinding: 'Hyperinflation & Bullae',
    peakRiskScore: 68,
    finalRiskScore: 22,
    dischargeDisposition: 'Step-down Unit',
    icuUnit: 'MICU Bed 04',
  },
  {
    id: '10052219',
    mrn: 'MRN-449102',
    name: 'Leonid Volkov',
    age: 70,
    gender: 'M',
    admissionDate: '2026-09-19',
    dischargeDate: '2026-09-27',
    lengthOfStayDays: 7.6,
    primaryDiagnosis: 'Hospital-Acquired Pneumonia',
    primaryCXRFinding: 'Consolidation & Effusion',
    peakRiskScore: 76,
    finalRiskScore: 24,
    dischargeDisposition: 'Discharged Home',
    icuUnit: 'MICU Bed 07',
  },
];

export const PatientArchive: React.FC<PatientArchiveProps> = ({ isDarkMode: _isDarkMode = true }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [dispositionFilter, setDispositionFilter] = useState('ALL');
  const [selectedRecord, setSelectedRecord] = useState<ArchivedPatientRecord | null>(null);

  // Filtered Archive Records
  const filteredRecords = useMemo(() => {
    return MOCK_ARCHIVE_DATA.filter((record) => {
      const matchesSearch =
        record.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        record.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        record.primaryDiagnosis.toLowerCase().includes(searchTerm.toLowerCase()) ||
        record.mrn.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesDisposition =
        dispositionFilter === 'ALL' || record.dischargeDisposition === dispositionFilter;

      return matchesSearch && matchesDisposition;
    });
  }, [searchTerm, dispositionFilter]);

  const handleExportCSV = () => {
    const headers = ['Patient ID', 'MRN', 'Name', 'Age', 'Gender', 'LOS (days)', 'Diagnosis', 'CXR Finding', 'Peak Risk (%)', 'Final Risk (%)', 'Disposition'];
    const rows = filteredRecords.map((r) => [
      r.id,
      r.mrn,
      r.name,
      r.age,
      r.gender,
      r.lengthOfStayDays,
      `"${r.primaryDiagnosis}"`,
      `"${r.primaryCXRFinding}"`,
      r.peakRiskScore,
      r.finalRiskScore,
      r.dischargeDisposition,
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'patient_archive_export.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto bg-gray-50 dark:bg-clinical-dark p-4 md:p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-gray-200 dark:border-clinical-border gap-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-500 border border-indigo-500/30">
            <Archive className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-mono tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-2">
              Patient Archive & Longitudinal Outcome Log
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-400 border border-indigo-500/30">
                Discharge Registry
              </span>
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono mt-0.5">
              Historical records of discharged and transferred ICU patients, trajectory audits, and final clinical outcomes.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-clinical-border text-xs font-mono text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-clinical-card transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl p-3.5 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Text Search Input */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by ID, Name, or Diagnosis..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:border-indigo-500"
          />
        </div>

        {/* Disposition Filter Select */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-3.5 h-3.5 text-gray-400" />
          <span className="text-xs font-mono text-gray-400">Disposition:</span>
          <select
            value={dispositionFilter}
            onChange={(e) => setDispositionFilter(e.target.value)}
            className="bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-gray-800 dark:text-gray-200 focus:outline-none focus:border-indigo-500"
          >
            <option value="ALL">All Dispositions</option>
            <option value="Discharged Home">Discharged Home</option>
            <option value="Step-down Unit">Step-down Unit</option>
            <option value="Long-term Care">Long-term Care</option>
            <option value="Deceased">Deceased</option>
          </select>

          <span className="text-xs font-mono text-gray-400 ml-2">
            Showing <strong className="text-gray-800 dark:text-gray-200">{filteredRecords.length}</strong> records
          </span>
        </div>
      </div>

      {/* Main Data Table */}
      <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-xl shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse font-mono text-xs">
            <thead className="bg-gray-100 dark:bg-slate-900/80 text-[10px] text-gray-400 uppercase tracking-wider border-b border-gray-200 dark:border-slate-800">
              <tr>
                <th className="p-3">Patient / MRN</th>
                <th className="p-3">Admission - Discharge</th>
                <th className="p-3">Length of Stay</th>
                <th className="p-3">Primary Diagnosis</th>
                <th className="p-3">Primary CXR Finding</th>
                <th className="p-3">Risk Peak → Final</th>
                <th className="p-3">Disposition</th>
                <th className="p-3 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-slate-800 text-[11px]">
              {filteredRecords.map((record) => {
                const isDeceased = record.dischargeDisposition === 'Deceased';
                return (
                  <tr
                    key={record.id}
                    onClick={() => setSelectedRecord(record)}
                    className="hover:bg-gray-50 dark:hover:bg-slate-800/40 cursor-pointer transition-colors"
                  >
                    <td className="p-3">
                      <div className="font-bold text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
                        <User className="w-3.5 h-3.5 text-indigo-400" />
                        {record.name}
                      </div>
                      <div className="text-[10px] text-gray-400 mt-0.5">
                        #{record.id} • {record.age}{record.gender} • {record.mrn}
                      </div>
                    </td>

                    <td className="p-3 text-gray-600 dark:text-gray-300">
                      <div>{record.admissionDate}</div>
                      <div className="text-[10px] text-gray-400 mt-0.5">→ {record.dischargeDate}</div>
                    </td>

                    <td className="p-3 font-bold text-gray-800 dark:text-gray-200">
                      {record.lengthOfStayDays} days
                    </td>

                    <td className="p-3">
                      <div className="text-gray-900 dark:text-gray-100 font-bold max-w-xs truncate">
                        {record.primaryDiagnosis}
                      </div>
                      <div className="text-[10px] text-gray-400">{record.icuUnit}</div>
                    </td>

                    <td className="p-3 text-cyan-400">
                      {record.primaryCXRFinding}
                    </td>

                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <span className="text-red-400 font-bold">{record.peakRiskScore}%</span>
                        <span className="text-gray-400">→</span>
                        <span
                          className={`font-bold ${
                            record.finalRiskScore >= 70
                              ? 'text-red-500'
                              : record.finalRiskScore >= 45
                              ? 'text-amber-500'
                              : record.finalRiskScore >= 25
                              ? 'text-blue-400'
                              : 'text-emerald-400'
                          }`}
                        >
                          {record.finalRiskScore}%
                        </span>
                      </div>
                    </td>

                    <td className="p-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isDeceased
                            ? 'bg-red-500/20 text-red-500 border border-red-500/30'
                            : record.dischargeDisposition === 'Discharged Home'
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                            : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                        }`}
                      >
                        {record.dischargeDisposition}
                      </span>
                    </td>

                    <td className="p-3 text-right">
                      <ChevronRight className="w-4 h-4 text-gray-400 ml-auto" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Record Inspection Modal / Drawer */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white dark:bg-clinical-panel border border-gray-200 dark:border-clinical-border rounded-2xl max-w-lg w-full p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-gray-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-bold font-mono text-gray-900 dark:text-gray-100">
                  Archived Clinical Record: #{selectedRecord.id}
                </h3>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="p-1 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800">
                <div>
                  <span className="text-gray-400 text-[10px]">Patient Name:</span>
                  <div className="font-bold text-gray-800 dark:text-gray-200">{selectedRecord.name}</div>
                </div>
                <div>
                  <span className="text-gray-400 text-[10px]">Demographics:</span>
                  <div className="font-bold text-gray-800 dark:text-gray-200">
                    {selectedRecord.age} yrs • {selectedRecord.gender === 'M' ? 'Male' : 'Female'}
                  </div>
                </div>
                <div>
                  <span className="text-gray-400 text-[10px]">Length of Stay:</span>
                  <div className="font-bold text-gray-800 dark:text-gray-200">{selectedRecord.lengthOfStayDays} Days</div>
                </div>
                <div>
                  <span className="text-gray-400 text-[10px]">Disposition:</span>
                  <div className="font-bold text-indigo-400">{selectedRecord.dischargeDisposition}</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 space-y-1">
                <span className="text-gray-400 text-[10px]">Primary Admission Diagnosis:</span>
                <div className="font-bold text-gray-800 dark:text-gray-200">{selectedRecord.primaryDiagnosis}</div>
              </div>

              <div className="p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 space-y-1">
                <span className="text-gray-400 text-[10px]">Radiographic Saliency & CXR Finding:</span>
                <div className="font-bold text-cyan-400">{selectedRecord.primaryCXRFinding}</div>
              </div>

              <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800">
                <div>
                  <span className="text-gray-400 text-[10px]">Peak Deterioration Risk:</span>
                  <div className="font-bold text-red-400 text-sm">{selectedRecord.peakRiskScore}%</div>
                </div>
                <div>
                  <span className="text-gray-400 text-[10px]">Discharge Final Risk:</span>
                  <div className="font-bold text-emerald-400 text-sm">{selectedRecord.finalRiskScore}%</div>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-gray-100 dark:border-slate-800">
              <button
                onClick={() => setSelectedRecord(null)}
                className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-mono transition-colors"
              >
                Close Record
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * TypeScript interfaces strictly matching the Python Pydantic schemas
 * (DigitalTwinResponse, VitalReadingItem, VitalSignSnapshot) from src/schemas/twin_schema.py.
 */

export interface VitalSignSnapshot {
  heart_rate: number;
  spo2: number;
  sbp: number;
  respiratory_rate: number;
  timestamp: string;
}

export interface VitalReadingItem {
  step_index: number;
  timestamp: string;
  heart_rate: number;
  spo2: number;
  sbp: number;
  respiratory_rate: number;
}

export type RiskTier = 'Low' | 'Moderate' | 'High' | 'Critical';

export interface DigitalTwinResponse {
  patient_id: string;
  study_id: string;
  deterioration_risk_score: number; // 0 to 100%
  risk_tier: RiskTier;
  cxr_probabilities: Record<string, number>;
  cxr_predictions: Record<string, boolean>;
  cxr_top_finding: string;
  cxr_top_probability: number;
  current_vitals: VitalSignSnapshot;
  vitals_trajectory_24h?: VitalReadingItem[];
  visual_risk_contribution: number;
  vitals_risk_contribution: number;
  risk_factors: string[];
  clinical_recommendation: string;
  heatmap_base64?: string;
  image_base64?: string;
  heatmap_image_base64?: string;
  input_image_base64?: string;
  timestamp: string;
}

export interface PatientProfile {
  id: string;
  study_id: string;
  name: string;
  age: number;
  gender: string;
  unit: string;
  bed: string;
  admission_diagnosis: string;
  intubated: boolean;
}

export interface SystemMetrics {
  inference_latency_ms: number;
  api_status: 'online' | 'degraded' | 'offline';
  device: string;
  last_sync: string;
}

export type ActiveNavView = 
  | 'dashboard' 
  | 'ward' 
  | 'treatment' 
  | 'sandbox'
  | 'what-if'
  | 'xai-diagnostics'
  | 'cohort-analytics'
  | 'mlops-telemetry'
  | 'patient-archive';

export interface WardAlert {
  id: string;
  timestamp: string;
  bed: string;
  patientName: string;
  patientId: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  metric?: string;
}

export interface BedOverview {
  patient: PatientProfile;
  riskScore: number;
  riskTier: RiskTier;
  topFinding: string;
  topProbability: number;
  currentVitals: VitalSignSnapshot;
  sparklineData: { time: string; value: number }[];
  primaryVitalLabel: 'Heart Rate' | 'SpO2';
  alerts: string[];
}

// 1. What-If Trajectory Simulator Types
export interface WhatIfVitals {
  heartRate: number;
  spo2: number;
  sbp: number;
  respiratoryRate: number;
}

export interface WhatIfHorizonPoint {
  horizon: string;
  hours: number;
  riskScore: number;
  baselineScore: number;
  tier: RiskTier;
  primaryStress: string;
}

// 2. XAI Diagnostics Types
export interface ShapFeatureImportance {
  feature: string;
  category: 'vitals' | 'imaging' | 'history' | 'labs';
  impactPercentage: number; // positive increases risk, negative decreases
  description: string;
  baselineValue: string;
}

export interface GradCAMSettings {
  selectedPathology: string;
  threshold: number; // 0.0 to 1.0
  opacity: number; // 0.0 to 1.0
  palette: 'jet' | 'viridis' | 'inferno';
}

// 3. Cohort Analytics Types
export interface PathologyDistributionItem {
  name: string;
  count: number;
  percentage: number;
  color: string;
}

export interface DailyAlertFrequency {
  date: string;
  critical: number;
  warning: number;
  info: number;
  total: number;
}

// 4. MLOps Telemetry Types
export interface MLOpsMetricCard {
  title: string;
  value: string;
  unit?: string;
  subtext: string;
  status: 'healthy' | 'warning' | 'critical';
  trend?: string;
}

export interface MemoryBreakdown {
  total_bytes: number;
  allocated_bytes: number;
  reserved_bytes: number;
  free_bytes: number;
  usage_percent: number;
  total_gb: number;
  allocated_gb: number;
  reserved_gb: number;
  free_gb: number;
}

export interface SystemTelemetryData {
  device_name: string;
  device_type: 'cuda' | 'cpu';
  cuda_available: boolean;
  cuda_version?: string;
  pytorch_version: string;
  gpu_memory: MemoryBreakdown;
  system_memory: MemoryBreakdown;
  cpu_percent: number;
  cpu_count_logical: number;
  cpu_count_physical: number;
  active_model: string;
  model_loaded: boolean;
  status: string;
  timestamp: string;
}

export interface ThroughputDataPoint {
  time: string;
  inferencesPerMin: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  gpuUtilization: number;
}

// 5. Patient Archive Types
export interface ArchivedPatientRecord {
  id: string;
  mrn: string;
  name: string;
  age: number;
  gender: 'M' | 'F';
  admissionDate: string;
  dischargeDate: string;
  lengthOfStayDays: number;
  primaryDiagnosis: string;
  primaryCXRFinding: string;
  peakRiskScore: number;
  finalRiskScore: number;
  dischargeDisposition: 'Discharged Home' | 'Step-down Unit' | 'Long-term Care' | 'Deceased';
  icuUnit: string;
}

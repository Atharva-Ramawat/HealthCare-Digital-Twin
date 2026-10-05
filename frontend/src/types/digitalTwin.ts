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

export type ActiveNavView = 'dashboard' | 'ward' | 'treatment' | 'sandbox';

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

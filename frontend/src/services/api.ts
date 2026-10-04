/**
 * API Service for AI-Driven Pulmonary ICU Digital Twin.
 * Connects to:
 * - GET /api/digital-twin/{patient_id}/{study_id}?include_trajectory=true
 * - GET /api/cxr/studies/{study_id}/heatmap?pathology={pathology}
 * - GET /api/health
 * 
 * Provides fallback mock generation for offline development and testing.
 */

import axios from 'axios';
import type { DigitalTwinResponse, VitalReadingItem, RiskTier } from '../types/digitalTwin';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

/**
 * Fetch complete multimodal digital twin assessment including 24h vitals trajectory.
 */
export async function fetchDigitalTwinAssessment(
  patientId: string,
  studyId: string,
  includeTrajectory: boolean = true
): Promise<{ data: DigitalTwinResponse; isMock: boolean; latencyMs: number }> {
  const start = performance.now();
  try {
    const response = await apiClient.get<DigitalTwinResponse>(
      `/api/digital-twin/${patientId}/${studyId}`,
      {
        params: { include_trajectory: includeTrajectory },
      }
    );
    const latencyMs = Math.round(performance.now() - start);
    return { data: response.data, isMock: false, latencyMs };
  } catch (error) {
    console.warn(`[Digital Twin API] Backend unavailable (${error}). Falling back to simulated clinical data.`);
    const latencyMs = Math.round(performance.now() - start);
    const mockData = generateRealisticMockDigitalTwin(patientId, studyId);
    return { data: mockData, isMock: true, latencyMs: Math.max(latencyMs, 14) };
  }
}

/**
 * Construct URL to retrieve Grad-CAM heatmap PNG image stream directly.
 */
export function getGradCAMHeatmapUrl(studyId: string, pathology: string): string {
  const cleanPathology = encodeURIComponent(pathology.trim());
  return `${API_BASE_URL}/api/cxr/studies/${studyId}/heatmap?pathology=${cleanPathology}`;
}

/**
 * Fetch Grad-CAM heatmap binary PNG image as a Blob with explicit responseType: 'blob'.
 */
export async function fetchGradCAMHeatmapBlob(
  studyId: string,
  pathology: string
): Promise<Blob> {
  const cleanPathology = encodeURIComponent(pathology.trim());
  const response = await apiClient.get<Blob>(
    `/api/cxr/studies/${studyId}/heatmap?pathology=${cleanPathology}`,
    {
      responseType: 'blob',
    }
  );
  return response.data;
}

/**
 * Submit manual patient intake with chest radiograph and vitals for real-time multimodal inference.
 */
export async function adHocInferDigitalTwin(
  formData: FormData
): Promise<{ data: DigitalTwinResponse; latencyMs: number; isMock: boolean }> {
  const start = performance.now();
  try {
    const response = await apiClient.post<DigitalTwinResponse>(
      '/api/digital-twin/ad-hoc-infer',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    const latencyMs = Math.round(performance.now() - start);
    return { data: response.data, latencyMs, isMock: false };
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    console.warn(`[Digital Twin API] Backend ad-hoc infer unavailable (${error}). Generating simulated response.`);
    const latencyMs = Math.round(performance.now() - start);
    const patientId = (formData.get('patient_id') as string) || 'CUSTOM-001';
    const mockData = generateRealisticMockDigitalTwin(patientId, `s_adhoc_${Date.now()}`);
    return { data: mockData, latencyMs: Math.max(latencyMs, 45), isMock: true };
  }
}

/**
 * Generate high-fidelity realistic clinical mock data for a patient & study.
 */
export function generateRealisticMockDigitalTwin(
  patientId: string,
  studyId: string
): DigitalTwinResponse {
  const now = new Date();
  const trajectory: VitalReadingItem[] = [];

  // Generate 96 15-minute readings spanning the last 24 hours
  // Patient is in acute deteriorating respiratory state (Pneumonia + ARDS + Septic Shock onset)
  for (let i = 0; i < 96; i++) {
    const stepTime = new Date(now.getTime() - (95 - i) * 15 * 60 * 1000);
    const progress = i / 95; // 0.0 to 1.0

    // Progressive tachycardia (76 -> 134 bpm)
    const hr = Math.round(76 + Math.pow(progress, 1.6) * 58 + (Math.sin(i * 0.4) * 3));
    // Progressive hypoxemia (98 -> 84%)
    const spo2 = Math.round(Math.max(78, 98 - Math.pow(progress, 1.8) * 14 + (Math.cos(i * 0.3) * 1.2)));
    // Progressive hypotension (126 -> 82 mmHg)
    const sbp = Math.round(Math.max(70, 126 - Math.pow(progress, 1.5) * 44 + (Math.sin(i * 0.5) * 4)));
    // Progressive tachypnea (15 -> 32 bpm)
    const rr = Math.round(15 + Math.pow(progress, 1.4) * 17 + (Math.cos(i * 0.6) * 1.5));

    trajectory.push({
      step_index: i,
      timestamp: stepTime.toISOString(),
      heart_rate: hr,
      spo2: spo2,
      sbp: sbp,
      respiratory_rate: rr,
    });
  }

  const latest = trajectory[trajectory.length - 1];

  const probabilities: Record<string, number> = {
    'Pneumonia': 0.8842,
    'Pleural Effusion': 0.7615,
    'Atelectasis': 0.6420,
    'Consolidation': 0.8190,
    'Edema': 0.7230,
    'Cardiomegaly': 0.4120,
    'Pneumothorax': 0.0810,
    'No Finding': 0.0340,
  };

  const predictions: Record<string, boolean> = {
    'Pneumonia': true,
    'Pleural Effusion': true,
    'Atelectasis': true,
    'Consolidation': true,
    'Edema': true,
    'Cardiomegaly': false,
    'Pneumothorax': false,
    'No Finding': false,
  };

  const riskScore = 84.5;
  const riskTier: RiskTier = 'Critical';

  return {
    patient_id: patientId,
    study_id: studyId,
    deterioration_risk_score: riskScore,
    risk_tier: riskTier,
    cxr_probabilities: probabilities,
    cxr_predictions: predictions,
    cxr_top_finding: 'Pneumonia',
    cxr_top_probability: 0.8842,
    current_vitals: {
      heart_rate: latest.heart_rate,
      spo2: latest.spo2,
      sbp: latest.sbp,
      respiratory_rate: latest.respiratory_rate,
      timestamp: latest.timestamp,
    },
    vitals_trajectory_24h: trajectory,
    visual_risk_contribution: 42.0,
    vitals_risk_contribution: 58.0,
    risk_factors: [
      `Severe Tachycardia (HR: ${latest.heart_rate} bpm)`,
      `Severe Hypoxemia (SpO2: ${latest.spo2}%)`,
      `Septic Hypotension / Shock (SBP: ${latest.sbp} mmHg)`,
      `Tachypnea / Respiratory Fatigue (RR: ${latest.respiratory_rate} bpm)`,
      'Dense Bilateral Pulmonary Consolidation & Infiltrates',
    ],
    clinical_recommendation:
      'EMERGENCY: Immediate ICU Rapid Response Team activation required. Prepare for endotracheal intubation, invasive arterial line placement, and immediate broad-spectrum antibiotic escalation.',
    timestamp: now.toISOString(),
  };
}

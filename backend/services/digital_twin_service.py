"""
Digital Twin Service - Core Orchestrator for 4-Partition Digital Twin State.
Integrates Observed, Derived, Predicted, and Simulation States for all virtual ICU patients.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.schemas.patient import (
    PatientSummary,
    PatientDetail,
    PatientEvent,
    PatientMedication
)
from backend.schemas.prediction import DigitalTwinStateResponse
from backend.services.replay_service import ClinicalReplayService
from backend.services.prediction_service import PredictionService


class DigitalTwinService:
    """
    Orchestrates patient records, timeline events, medications, and unified Digital Twin State queries.
    """

    def __init__(self, replay_service: ClinicalReplayService, prediction_service: PredictionService):
        self.replay = replay_service
        self.prediction = prediction_service
        self._seed_events_and_medications()

    def _seed_events_and_medications(self):
        """Seed clinical timeline events and medication registries."""
        self.events_db: Dict[str, List[PatientEvent]] = {
            "PAT-101": [
                PatientEvent(
                    event_id="EVT-101-1",
                    patient_id="PAT-101",
                    timestamp="2026-08-20 06:30:00",
                    event_type="admission",
                    title="ICU Admission",
                    description="Admitted post-operatively following elective abdominal surgery for hemodynamic stability monitoring.",
                    severity="normal"
                ),
                PatientEvent(
                    event_id="EVT-101-2",
                    patient_id="PAT-101",
                    timestamp="2026-08-21 10:00:00",
                    event_type="medication_change",
                    title="Analgesic Dosage Adjusted",
                    description="Transitioned from IV PCA morphine to oral acetaminophen/tramadol regimen.",
                    severity="normal"
                )
            ],
            "PAT-102": [
                PatientEvent(
                    event_id="EVT-102-1",
                    patient_id="PAT-102",
                    timestamp="2026-08-21 14:10:00",
                    event_type="admission",
                    title="Emergency ICU Transfer",
                    description="Transferred from emergency ward due to severe bacterial pneumonia, lactic acidosis (4.2 mmol/L), and hypotension.",
                    severity="critical"
                ),
                PatientEvent(
                    event_id="EVT-102-2",
                    patient_id="PAT-102",
                    timestamp="2026-08-22 03:20:00",
                    event_type="vital_anomaly",
                    title="Septic Shock Progression Alert",
                    description="MAP dropped below 65 mmHg; heart rate spiked to 135 bpm with temperature reaching 39.4 °C.",
                    severity="critical"
                ),
                PatientEvent(
                    event_id="EVT-102-3",
                    patient_id="PAT-102",
                    timestamp="2026-08-22 04:00:00",
                    event_type="medication_change",
                    title="Vasopressor (Norepinephrine) Initiated",
                    description="Started continuous IV Norepinephrine infusion at 0.08 mcg/kg/min for hemodynamic support.",
                    severity="warning"
                )
            ],
            "PAT-103": [
                PatientEvent(
                    event_id="EVT-103-1",
                    patient_id="PAT-103",
                    timestamp="2026-08-22 01:45:00",
                    event_type="admission",
                    title="ICU Admission for ARDS",
                    description="Admitted with acute hypoxemic respiratory failure following viral pneumonia.",
                    severity="critical"
                ),
                PatientEvent(
                    event_id="EVT-103-2",
                    patient_id="PAT-103",
                    timestamp="2026-08-22 07:30:00",
                    event_type="vital_anomaly",
                    title="Severe Hypoxia Episode",
                    description="SpO2 declined to 82% despite 60% High-Flow Nasal Cannula (HFNC). Respiratory rate 36 breaths/min.",
                    severity="critical"
                )
            ],
            "PAT-104": [
                PatientEvent(
                    event_id="EVT-104-1",
                    patient_id="PAT-104",
                    timestamp="2026-08-22 09:00:00",
                    event_type="admission",
                    title="Step-Down Monitoring",
                    description="Admitted for 24-hour observation after uncomplicated thoracic wedge resection.",
                    severity="normal"
                )
            ]
        }

        self.medications_db: Dict[str, List[PatientMedication]] = {
            "PAT-101": [
                PatientMedication(medication_id="MED-101-1", patient_id="PAT-101", name="Cefazolin", dosage="2 g", route="IV", frequency="q8h", start_time="2026-08-20 07:00:00", is_active=True),
                PatientMedication(medication_id="MED-101-2", patient_id="PAT-101", name="Acetaminophen", dosage="1000 mg", route="Oral", frequency="q6h PRN", start_time="2026-08-21 10:00:00", is_active=True),
                PatientMedication(medication_id="MED-101-3", patient_id="PAT-101", name="Enoxaparin", dosage="40 mg", route="SubQ", frequency="Daily", start_time="2026-08-20 20:00:00", is_active=True)
            ],
            "PAT-102": [
                PatientMedication(medication_id="MED-102-1", patient_id="PAT-102", name="Norepinephrine", dosage="0.08 mcg/kg/min", route="IV", frequency="Continuous", start_time="2026-08-22 04:00:00", is_active=True),
                PatientMedication(medication_id="MED-102-2", patient_id="PAT-102", name="Piperacillin/Tazobactam", dosage="4.5 g", route="IV", frequency="q6h", start_time="2026-08-21 14:30:00", is_active=True),
                PatientMedication(medication_id="MED-102-3", patient_id="PAT-102", name="Vancomycin", dosage="1.5 g", route="IV", frequency="q12h", start_time="2026-08-21 14:30:00", is_active=True),
                PatientMedication(medication_id="MED-102-4", patient_id="PAT-102", name="Hydrocortisone", dosage="50 mg", route="IV", frequency="q6h", start_time="2026-08-22 05:00:00", is_active=True)
            ],
            "PAT-103": [
                PatientMedication(medication_id="MED-103-1", patient_id="PAT-103", name="Dexamethasone", dosage="6 mg", route="IV", frequency="Daily", start_time="2026-08-22 02:00:00", is_active=True),
                PatientMedication(medication_id="MED-103-2", patient_id="PAT-103", name="Furosemide", dosage="40 mg", route="IV", frequency="q12h", start_time="2026-08-22 06:00:00", is_active=True),
                PatientMedication(medication_id="MED-103-3", patient_id="PAT-103", name="Albuterol/Ipratropium", dosage="3 mL", route="Inhalation", frequency="q4h", start_time="2026-08-22 02:30:00", is_active=True)
            ],
            "PAT-104": [
                PatientMedication(medication_id="MED-104-1", patient_id="PAT-104", name="Cefepime", dosage="1 g", route="IV", frequency="q8h", start_time="2026-08-22 09:30:00", is_active=True),
                PatientMedication(medication_id="MED-104-2", patient_id="PAT-104", name="Hydromorphone", dosage="0.5 mg", route="IV", frequency="q3h PRN", start_time="2026-08-22 09:30:00", is_active=True)
            ]
        }

    def list_patients(self) -> List[PatientSummary]:
        """Retrieve list of all active ICU patients with real-time risk scores."""
        summaries = []
        for pid, meta in self.replay.patients_meta.items():
            latest = self.replay.get_latest_vitals(pid)
            window_mat = self.replay.get_sliding_window_matrix(pid)
            pred = self.prediction.predict_risk(pid, window_mat)

            vitals_dict = latest.dict() if latest else {}
            summaries.append(
                PatientSummary(
                    patient_id=pid,
                    name=meta["name"],
                    age=meta["age"],
                    gender=meta["gender"],
                    bed=meta["bed"],
                    admission_time=meta["admission_time"],
                    primary_diagnosis=meta["primary_diagnosis"],
                    mode=self.replay.mode,
                    trajectory_template=meta.get("trajectory"),
                    latest_vitals=vitals_dict,
                    news2_score=pred.news2.total_score,
                    news2_tier=pred.news2.risk_tier,
                    health_risk_score=pred.health_risk_score,
                    is_model_available=pred.is_model_available
                )
            )
        return summaries

    def get_patient_detail(self, patient_id: str) -> Optional[PatientDetail]:
        """Retrieve detailed profile for a single patient."""
        meta = self.replay.patients_meta.get(patient_id)
        if not meta:
            return None

        latest = self.replay.get_latest_vitals(patient_id)
        window_mat = self.replay.get_sliding_window_matrix(patient_id)
        pred = self.prediction.predict_risk(patient_id, window_mat)
        vitals_dict = latest.dict() if latest else {}

        return PatientDetail(
            patient_id=patient_id,
            name=meta["name"],
            age=meta["age"],
            gender=meta["gender"],
            bed=meta["bed"],
            admission_time=meta["admission_time"],
            primary_diagnosis=meta["primary_diagnosis"],
            mode=self.replay.mode,
            trajectory_template=meta.get("trajectory"),
            latest_vitals=vitals_dict,
            news2_score=pred.news2.total_score,
            news2_tier=pred.news2.risk_tier,
            health_risk_score=pred.health_risk_score,
            is_model_available=pred.is_model_available,
            medical_history=meta.get("medical_history", []),
            allergies=meta.get("allergies", []),
            attending_physician=meta.get("attending_physician", "Dr. A. Sharma (ICU Lead)"),
            icu_stay_id=f"MIMIC-ICU-{patient_id.split('-')[-1]}"
        )

    def get_patient_events(self, patient_id: str) -> List[PatientEvent]:
        """Retrieve chronological clinical events for patient."""
        return self.events_db.get(patient_id, [])

    def get_patient_medications(self, patient_id: str) -> List[PatientMedication]:
        """Retrieve active and historical medications for patient."""
        return self.medications_db.get(patient_id, [])

    def get_digital_twin_state(self, patient_id: str) -> Optional[DigitalTwinStateResponse]:
        """
        Generate full 4-partition Digital Twin state response for patient.
        """
        meta = self.replay.patients_meta.get(patient_id)
        if not meta:
            return None

        latest_vitals = self.replay.get_latest_vitals(patient_id)
        window_mat = self.replay.get_sliding_window_matrix(patient_id)
        pred = self.prediction.predict_risk(patient_id, window_mat)
        sim_state = self.replay.get_simulation_state()

        # 1. Observed State
        obs_state = {
            "patient_id": patient_id,
            "demographics": {"name": meta["name"], "age": meta["age"], "gender": meta["gender"], "bed": meta["bed"]},
            "current_vitals": latest_vitals.dict() if latest_vitals else {},
            "latest_timestamp": latest_vitals.timestamp if latest_vitals else time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": self.replay.mode,
            "step_index": self.replay.current_step
        }

        # 2. Derived State (causally computed)
        latest_d = latest_vitals.dict() if latest_vitals else {}
        derived_state = {
            "mean_arterial_pressure": latest_d.get("mean_arterial_pressure", 93.3),
            "shock_index": latest_d.get("shock_index", 0.62),
            "news2_total": pred.news2.total_score,
            "news2_tier": pred.news2.risk_tier,
            "news2_breakdown": pred.news2.dict(),
            "patient_baseline": {"heart_rate": 75.0, "spo2": 98.0, "systolic_bp": 120.0, "respiratory_rate": 16.0, "body_temperature": 37.0},
            "baseline_deviations": {
                k: round(latest_d.get(k, 0) - v, 2)
                for k, v in {"heart_rate": 75.0, "spo2": 98.0, "systolic_bp": 120.0, "respiratory_rate": 16.0, "body_temperature": 37.0}.items()
            }
        }

        # 3. Predicted State
        predicted_state = {
            "health_risk_score": pred.health_risk_score,
            "multi_horizon_risk": pred.multi_horizon.dict(),
            "forecast_15m": pred.forecast.dict(),
            "xai_attributions": [item.dict() for item in pred.xai_attributions],
            "is_model_available": pred.is_model_available,
            "model_version": pred.model_version
        }

        # 4. Simulation State
        simulation_state = {
            "is_playing": sim_state.is_playing,
            "speed_multiplier": sim_state.speed_multiplier,
            "mode": sim_state.mode,
            "active_anomalies": sim_state.active_anomalies.get(patient_id, []),
            "what_if_overrides": self.replay.what_if_overrides.get(patient_id, {})
        }

        return DigitalTwinStateResponse(
            patient_id=patient_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            mode=self.replay.mode,
            observed_state=obs_state,
            derived_state=derived_state,
            predicted_state=predicted_state,
            simulation_state=simulation_state
        )

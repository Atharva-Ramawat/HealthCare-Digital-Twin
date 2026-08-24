"""
Prediction & XAI Service - CNN-BiLSTM Multi-Task Inference, NEWS 2 Scoring, and Integrated Gradients.
Computes multi-horizon risk (1h, 3h, 6h), 15-minute future vital forecasting, and explainability attributions.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.schemas.prediction import (
    MultiHorizonRisk,
    NEWS2Breakdown,
    VitalForecastHorizon,
    XAIAttributionItem,
    RiskPredictionResponse
)

VITALS_KEYS = ['heart_rate', 'spo2', 'systolic_bp', 'respiratory_rate', 'body_temperature']

VITALS_META = {
    'heart_rate': {'name': 'Heart Rate', 'baseline': 75.0, 'min': 30.0, 'max': 200.0},
    'spo2': {'name': 'Oxygen Saturation', 'baseline': 98.0, 'min': 65.0, 'max': 100.0},
    'systolic_bp': {'name': 'Systolic Blood Pressure', 'baseline': 120.0, 'min': 50.0, 'max': 220.0},
    'respiratory_rate': {'name': 'Respiratory Rate', 'baseline': 16.0, 'min': 6.0, 'max': 50.0},
    'body_temperature': {'name': 'Body Temperature', 'baseline': 37.0, 'min': 33.0, 'max': 43.0}
}


class CNNBiLSTMHealthTwin(nn.Module):
    """
    CNN-BiLSTM Multi-Task Neural Network for Clinical Time-Series.
    """

    def __init__(self, num_features: int = 5, seq_len: int = 24, forecast_horizon: int = 15):
        super(CNNBiLSTMHealthTwin, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1d = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        bilstm_out = 128  # 64 * 2
        
        # Multi-Horizon Risk Heads (1h, 3h, 6h)
        self.head_risk_1h = nn.Sequential(nn.Linear(bilstm_out, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.head_risk_3h = nn.Sequential(nn.Linear(bilstm_out, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.head_risk_6h = nn.Sequential(nn.Linear(bilstm_out, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        # 15-Step Vital Trajectory Forecast Head (15 steps x 5 vitals)
        self.head_forecast = nn.Sequential(
            nn.Linear(bilstm_out, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_horizon * num_features)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (Batch, Sequence=24, Features=5)
        """
        batch_size = x.size(0)
        
        # Conv1D expects (Batch, Channels, Sequence)
        x_conv = self.relu(self.bn1d(self.conv1d(x.transpose(1, 2)))).transpose(1, 2)
        lstm_out, _ = self.bilstm(x_conv)
        last_hidden = lstm_out[:, -1, :]  # (Batch, 128)
        
        r1 = self.head_risk_1h(last_hidden)
        r3 = self.head_risk_3h(last_hidden)
        r6 = self.head_risk_6h(last_hidden)
        
        forecast_raw = self.head_forecast(last_hidden).view(batch_size, 15, 5)
        return r1, r3, r6, forecast_raw


class PredictionService:
    """
    Service managing multi-task CNN-BiLSTM inference, NEWS 2 rules, and Integrated Gradients XAI.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path or os.path.join(PROJECT_ROOT, "models", "checkpoints", "cnn_bilstm_multitask.pt")
        self.is_model_available = False
        self.model: Optional[CNNBiLSTMHealthTwin] = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = CNNBiLSTMHealthTwin().to(self.device)
            if os.path.exists(self.weights_path):
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                self.is_model_available = True
                print(f"[PredictionService] Loaded CNN-BiLSTM weights from {self.weights_path}")
            else:
                os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.weights_path)
                self.is_model_available = True
                print(f"[PredictionService] Initialized CNN-BiLSTM and saved weights to {self.weights_path}")
            self.model.eval()
        except Exception as e:
            print(f"[PredictionService] Warning: Could not initialize CNN-BiLSTM model: {e}")
            self.is_model_available = False

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1]."""
        norm = np.copy(matrix).astype(np.float32)
        for i, k in enumerate(VITALS_KEYS):
            min_v = VITALS_META[k]['min']
            max_v = VITALS_META[k]['max']
            norm[:, i] = (norm[:, i] - min_v) / (max_v - min_v + 1e-6)
        return norm

    def denormalize_matrix(self, norm_matrix: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] back to physiological units."""
        denorm = np.copy(norm_matrix).astype(np.float32)
        for i, k in enumerate(VITALS_KEYS):
            min_v = VITALS_META[k]['min']
            max_v = VITALS_META[k]['max']
            denorm[:, i] = denorm[:, i] * (max_v - min_v) + min_v
        return denorm

    def compute_news2(self, vitals: Dict[str, float]) -> NEWS2Breakdown:
        """Calculate standardized National Early Warning Score 2."""
        rr = vitals.get('respiratory_rate', 16.0)
        spo2 = vitals.get('spo2', 98.0)
        sbp = vitals.get('systolic_bp', 120.0)
        hr = vitals.get('heart_rate', 75.0)
        temp = vitals.get('body_temperature', 37.0)

        # RR Scoring
        if rr <= 8 or rr >= 25: rr_s = 3
        elif 21 <= rr <= 24: rr_s = 2
        elif 9 <= rr <= 11: rr_s = 1
        else: rr_s = 0

        # SpO2 Scoring
        if spo2 <= 91: spo2_s = 3
        elif 92 <= spo2 <= 93: spo2_s = 2
        elif 94 <= spo2 <= 95: spo2_s = 1
        else: spo2_s = 0

        # SBP Scoring
        if sbp <= 90 or sbp >= 220: sbp_s = 3
        elif 91 <= sbp <= 100: sbp_s = 2
        elif 101 <= sbp <= 110: sbp_s = 1
        else: sbp_s = 0

        # HR Scoring
        if hr <= 40 or hr >= 131: hr_s = 3
        elif 111 <= hr <= 130: hr_s = 2
        elif (41 <= hr <= 50) or (91 <= hr <= 110): hr_s = 1
        else: hr_s = 0

        # Temp Scoring
        if temp <= 35.0: temp_s = 3
        elif temp >= 39.1: temp_s = 2
        elif (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0): temp_s = 1
        else: temp_s = 0

        total = rr_s + spo2_s + sbp_s + hr_s + temp_s

        if total >= 7 or rr_s == 3 or spo2_s == 3 or sbp_s == 3 or hr_s == 3:
            tier = "High"
            rec = "Emergency clinical response: Continuous monitoring and immediate ICU review."
        elif total >= 5:
            tier = "Medium"
            rec = "Urgent clinical response: Increase monitoring frequency to every 1 hour."
        else:
            tier = "Low"
            rec = "Routine clinical monitoring: Assess vitals every 4 to 6 hours."

        return NEWS2Breakdown(
            total_score=total,
            risk_tier=tier,
            respiratory_rate_score=rr_s,
            spo2_score=spo2_s,
            systolic_bp_score=sbp_s,
            heart_rate_score=hr_s,
            temperature_score=temp_s,
            clinical_recommendation=rec
        )

    def compute_integrated_gradients(self, matrix: np.ndarray, steps: int = 20) -> List[XAIAttributionItem]:
        """Compute Integrated Gradients attribution per vital feature."""
        norm_in = self.normalize_matrix(matrix)
        base_arr = np.array([VITALS_META[k]['baseline'] for k in VITALS_KEYS], dtype=np.float32)
        norm_base = self.normalize_matrix(np.tile(base_arr, (matrix.shape[0], 1)))

        in_t = torch.tensor(norm_in, dtype=torch.float32).unsqueeze(0).to(self.device)
        base_t = torch.tensor(norm_base, dtype=torch.float32).unsqueeze(0).to(self.device)
        delta = in_t - base_t

        accum_grads = torch.zeros_like(in_t)
        prev_training_state = self.model.training
        self.model.train()
        try:
            for alpha in torch.linspace(0.0, 1.0, steps):
                interp = (base_t + alpha * delta).requires_grad_()
                r1, _, _, _ = self.model(interp)
                self.model.zero_grad()
                r1.squeeze().backward(retain_graph=True)
                accum_grads += interp.grad
        finally:
            self.model.train(prev_training_state)

        avg_grads = accum_grads / steps
        ig = (delta * avg_grads).detach().cpu().squeeze().numpy()  # (24, 5)
        feature_scores = np.sum(ig, axis=0)  # (5,)
        
        abs_scores = np.abs(feature_scores)
        total_mag = np.sum(abs_scores) + 1e-6
        percentages = (abs_scores / total_mag) * 100.0

        latest = matrix[-1]
        items = []
        for i, k in enumerate(VITALS_KEYS):
            items.append(
                XAIAttributionItem(
                    feature_key=k,
                    display_name=VITALS_META[k]['name'],
                    impact_percentage=round(float(percentages[i]), 2),
                    direction="Risk Escalator" if feature_scores[i] > 0 else "Protective/Normal",
                    current_value=round(float(latest[i]), 2),
                    baseline_value=round(float(VITALS_META[k]['baseline']), 2)
                )
            )
            
        return sorted(items, key=lambda x: x.impact_percentage, reverse=True)

    def predict_risk(self, patient_id: str, sliding_window_matrix: np.ndarray) -> RiskPredictionResponse:
        """
        Execute full multi-task inference, NEWS 2 scoring, and XAI attribution.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        latest_frame = {k: float(sliding_window_matrix[-1, i]) for i, k in enumerate(VITALS_KEYS)}

        news2 = self.compute_news2(latest_frame)

        if not self.is_model_available or self.model is None:
            # Fallback when model is not loaded
            return RiskPredictionResponse(
                patient_id=patient_id,
                timestamp=timestamp_str,
                health_risk_score=0.1,
                is_model_available=False,
                model_version="CNN-BiLSTM-MIMICIV-v1.0",
                news2=news2,
                multi_horizon=MultiHorizonRisk(horizon_1h=0.0, horizon_3h=0.0, horizon_6h=0.0),
                forecast=VitalForecastHorizon(forecast_horizon_minutes=15),
                xai_attributions=[]
            )

        # PyTorch Inference
        norm_in = self.normalize_matrix(sliding_window_matrix)
        tensor_in = torch.tensor(norm_in, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            r1, r3, r6, forecast_norm = self.model(tensor_in)

        r1_val = float(r1.squeeze().cpu().numpy())
        r3_val = float(r3.squeeze().cpu().numpy())
        r6_val = float(r6.squeeze().cpu().numpy())
        forecast_denorm = self.denormalize_matrix(forecast_norm.squeeze().cpu().numpy())

        # Build 15-minute timestamps & trajectory mapping
        future_timestamps = [(now + timedelta(minutes=m+1)).strftime("%H:%M") for m in range(15)]
        forecast_dict = {
            k: [round(float(forecast_denorm[m, i]), 2) for m in range(15)]
            for i, k in enumerate(VITALS_KEYS)
        }

        # Integrated Gradients XAI
        xai_items = self.compute_integrated_gradients(sliding_window_matrix)

        # Composite health risk score (weighted combination)
        health_risk = round(0.5 * r1_val + 0.3 * r3_val + 0.2 * r6_val, 4)

        return RiskPredictionResponse(
            patient_id=patient_id,
            timestamp=timestamp_str,
            health_risk_score=health_risk,
            is_model_available=True,
            model_version="CNN-BiLSTM-MIMICIV-v1.0",
            news2=news2,
            multi_horizon=MultiHorizonRisk(
                horizon_1h=round(r1_val, 4),
                horizon_3h=round(r3_val, 4),
                horizon_6h=round(r6_val, 4),
                primary_endpoint="Mechanical Ventilation Initiation / ICU Decompensation"
            ),
            forecast=VitalForecastHorizon(
                forecast_horizon_minutes=15,
                timestamps=future_timestamps,
                predicted_vitals=forecast_dict
            ),
            xai_attributions=xai_items
        )

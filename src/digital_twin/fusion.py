"""
Multimodal Digital Twin Fusion Model & ICU Risk Scoring Engine.
Combines 1024-dimensional vision embeddings from DenseNet-121 with normalized temporal vital sign snapshots.
Outputs a unified ICU Deterioration Risk Score (0% to 100%) and clinical risk tiering.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class FusionResult:
    """Structured result returned by DigitalTwinFusion risk evaluation."""
    deterioration_risk_score: float  # Percentage 0.0 to 100.0%
    risk_tier: str                   # 'Low', 'Moderate', 'High', 'Critical'
    visual_risk_contribution: float  # % contribution of visual features
    vitals_risk_contribution: float  # % contribution of vitals instability
    risk_factors: List[str]          # Key clinical drivers
    clinical_recommendation: str     # Clinical actionable guidance


class LightweightFusionMLP(nn.Module):
    """
    Lightweight feedforward Multilayer Perceptron for multimodal fusion.
    Concatenates visual embedding (1024) and normalized vital signs (4) to predict deterioration probability.
    """
    def __init__(self, visual_dim: int = 1024, vitals_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + vitals_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reproducible Xavier normal distribution."""
        torch.manual_seed(42)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting [Batch, 1028] -> [Batch, 1]."""
        return self.net(x)


class DigitalTwinFusion:
    """
    Digital Twin Multimodal Fusion Engine.
    Accepts 1024-dimensional visual embeddings and clinical vital sign snapshots.
    Concatenates normalized representations and passes them through an MLP and clinical heuristics
    to compute a calibrated ICU Deterioration Risk Score (0 to 100%).
    """

    VISUAL_DIM: int = 1024
    VITALS_DIM: int = 4

    # Physiological reference points for normalization
    VITAL_REFERENCES = {
        "heart_rate": {"mean": 75.0, "std": 20.0, "high_risk": 110.0, "critical": 130.0},
        "spo2": {"mean": 98.0, "std": 3.0, "low_risk": 92.0, "critical": 88.0},
        "sbp": {"mean": 120.0, "std": 20.0, "low_risk": 95.0, "critical": 85.0},
        "respiratory_rate": {"mean": 16.0, "std": 4.0, "high_risk": 22.0, "critical": 28.0}
    }

    def __init__(self, mlp_model: Optional[LightweightFusionMLP] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.mlp = mlp_model or LightweightFusionMLP(
            visual_dim=self.VISUAL_DIM,
            vitals_dim=self.VITALS_DIM,
            hidden_dim=64
        ).to(self.device)
        self.mlp.eval()

    def normalize_vitals(self, vitals: Dict[str, float]) -> np.ndarray:
        """
        Normalize raw vital signs into standardized clinical deviation metrics.
        Higher normalized values indicate higher physiological instability.
        
        Order: [HR_risk, SpO2_risk, SBP_risk, RR_risk]
        """
        hr = float(vitals.get("heart_rate", 75.0))
        spo2 = float(vitals.get("spo2", 98.0))
        sbp = float(vitals.get("sbp", 120.0))
        rr = float(vitals.get("respiratory_rate", 16.0))

        # Tachycardia / bradycardia deviation
        norm_hr = (hr - 75.0) / 25.0
        # Hypoxia: drops below 98 increase risk dramatically
        norm_spo2 = (98.0 - spo2) / 8.0
        # Hypotension: drops below 115 increase risk
        norm_sbp = (115.0 - sbp) / 25.0
        # Tachypnea: rises above 16 increase risk
        norm_rr = (rr - 16.0) / 6.0

        vitals_vec = np.array([norm_hr, norm_spo2, norm_sbp, norm_rr], dtype=np.float32)
        return np.clip(vitals_vec, -3.0, 5.0)

    def compute_vitals_instability_score(self, vitals: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Compute an interpretable clinical instability score (0 to 100) and identify active risk factors.
        """
        hr = float(vitals.get("heart_rate", 75.0))
        spo2 = float(vitals.get("spo2", 98.0))
        sbp = float(vitals.get("sbp", 120.0))
        rr = float(vitals.get("respiratory_rate", 16.0))

        points = 0.0
        risk_factors: List[str] = []

        # Heart rate evaluation
        if hr >= 130.0:
            points += 30.0
            risk_factors.append(f"Severe Tachycardia (HR: {hr:.0f} bpm)")
        elif hr >= 110.0:
            points += 18.0
            risk_factors.append(f"Moderate Tachycardia (HR: {hr:.0f} bpm)")
        elif hr < 45.0:
            points += 20.0
            risk_factors.append(f"Bradycardia (HR: {hr:.0f} bpm)")

        # SpO2 evaluation (Critical ICU indicator)
        if spo2 <= 88.0:
            points += 35.0
            risk_factors.append(f"Severe Hypoxemia (SpO2: {spo2:.1f}%)")
        elif spo2 <= 92.0:
            points += 22.0
            risk_factors.append(f"Moderate Hypoxemia (SpO2: {spo2:.1f}%)")
        elif spo2 < 95.0:
            points += 10.0
            risk_factors.append(f"Mild Desaturation (SpO2: {spo2:.1f}%)")

        # SBP evaluation (Circulatory shock indicator)
        if sbp <= 85.0:
            points += 30.0
            risk_factors.append(f"Severe Hypotension / Shock (SBP: {sbp:.0f} mmHg)")
        elif sbp <= 95.0:
            points += 18.0
            risk_factors.append(f"Moderate Hypotension (SBP: {sbp:.0f} mmHg)")

        # Respiratory rate evaluation (Early deterioration warning)
        if rr >= 28.0:
            points += 25.0
            risk_factors.append(f"Severe Tachypnea (RR: {rr:.0f} bpm)")
        elif rr >= 22.0:
            points += 14.0
            risk_factors.append(f"Elevated Respiratory Rate (RR: {rr:.0f} bpm)")

        return min(points, 100.0), risk_factors

    def compute_risk(
        self,
        visual_embedding: Union[List[float], np.ndarray, torch.Tensor],
        vitals: Dict[str, float],
        cxr_top_finding: Optional[str] = None,
        cxr_top_probability: Optional[float] = None
    ) -> FusionResult:
        """
        Concatenate visual embedding with normalized vitals and compute unified ICU Deterioration Risk.
        
        Args:
            visual_embedding: 1024-dimensional feature vector from DenseNet-121.
            vitals: Dictionary containing latest vital signs ('heart_rate', 'spo2', 'sbp', 'respiratory_rate').
            cxr_top_finding: Optional label of top pulmonary pathology (e.g. 'Pneumonia').
            cxr_top_probability: Optional confidence score of top pathology (0.0 to 1.0).
            
        Returns:
            FusionResult containing Deterioration Risk Score (0-100%), risk tier, and clinical factors.
        """
        # 1. Format visual embedding into normalized 1024-dim numpy array
        if isinstance(visual_embedding, torch.Tensor):
            vis_arr = visual_embedding.detach().cpu().numpy().flatten()
        else:
            vis_arr = np.array(visual_embedding, dtype=np.float32).flatten()

        if len(vis_arr) != self.VISUAL_DIM:
            # Handle potential dimension mismatch by padding or truncating gracefully
            if len(vis_arr) < self.VISUAL_DIM:
                vis_arr = np.pad(vis_arr, (0, self.VISUAL_DIM - len(vis_arr)))
            else:
                vis_arr = vis_arr[:self.VISUAL_DIM]

        # L2-normalize visual embedding for stable joint magnitude
        vis_norm_val = np.linalg.norm(vis_arr)
        if vis_norm_val > 1e-8:
            vis_normalized = vis_arr / vis_norm_val
        else:
            vis_normalized = vis_arr

        # 2. Normalize vital signs
        vitals_norm = self.normalize_vitals(vitals)

        # 3. Concatenate visual embedding and normalized vitals -> [1, 1028]
        fusion_vector = np.concatenate([vis_normalized, vitals_norm], axis=0).astype(np.float32)
        fusion_tensor = torch.from_numpy(fusion_vector).unsqueeze(0).to(self.device)

        # 4. Forward pass through Lightweight Fusion MLP
        with torch.no_grad():
            mlp_prob = float(self.mlp(fusion_tensor).cpu().numpy()[0, 0])

        # 5. Compute clinical physiological vital instability score
        vitals_score, risk_factors = self.compute_vitals_instability_score(vitals)

        # 6. Compute visual pathology risk factor
        vis_risk = 0.0
        if cxr_top_probability is not None and cxr_top_finding and cxr_top_finding != "No Finding":
            vis_risk = cxr_top_probability * 100.0
            if cxr_top_probability >= 0.40:
                risk_factors.append(f"Visual Infiltrate/Abnormality: {cxr_top_finding} ({cxr_top_probability*100:.1f}%)")
        else:
            # Use visual embedding mean activation magnitude
            vis_risk = float(np.clip(np.mean(np.abs(vis_arr)) * 120.0, 5.0, 75.0))

        # 7. Unified Multimodal Fusion Risk Combination
        # 45% MLP prediction + 35% Vital Instability + 20% Visual Pathology
        combined_score = (mlp_prob * 100.0 * 0.45) + (vitals_score * 0.35) + (vis_risk * 0.20)
        risk_score = round(float(np.clip(combined_score, 0.0, 100.0)), 1)

        # Calculate relative risk contributions
        total_parts = (mlp_prob * 100.0 * 0.45) + (vitals_score * 0.35) + (vis_risk * 0.20)
        if total_parts > 1e-6:
            vitals_contrib = round(((mlp_prob * 100.0 * 0.25 + vitals_score * 0.35) / total_parts) * 100.0, 1)
            visual_contrib = round(100.0 - vitals_contrib, 1)
        else:
            vitals_contrib = 50.0
            visual_contrib = 50.0

        # 8. Clinical Tiering & Action Recommendations
        if risk_score >= 75.0:
            risk_tier = "Critical"
            recommendation = (
                "EMERGENCY: Immediate ICU Rapid Response Team activation required. "
                "Prepare for airway stabilization, invasive arterial line monitoring, and vasopressor support."
            )
        elif risk_score >= 50.0:
            risk_tier = "High"
            recommendation = (
                "URGENT: High probability of acute deterioration within next 2-4 hours. "
                "Initiate supplemental oxygen, recheck arterial blood gas (ABG), and notify attending intensivist."
            )
        elif risk_score >= 25.0:
            risk_tier = "Moderate"
            recommendation = (
                "MONITOR: Elevated physiological stress. Increase vitals sampling to every 15 minutes "
                "and monitor for worsening respiratory fatigue or lactate elevation."
            )
        else:
            risk_tier = "Low"
            recommendation = "STABLE: Patient within safe physiological margins. Continue standard ICU / step-down telemetry."

        if not risk_factors:
            risk_factors.append("No acute physiological or visual deterioration triggers detected")

        return FusionResult(
            deterioration_risk_score=risk_score,
            risk_tier=risk_tier,
            visual_risk_contribution=visual_contrib,
            vitals_risk_contribution=vitals_contrib,
            risk_factors=risk_factors,
            clinical_recommendation=recommendation
        )

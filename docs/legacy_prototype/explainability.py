"""
Explainable AI (XAI) Module - Integrated Gradients & Clinical Feature Sensitivity
Computes gradient-based feature importance attributions for the CNN-BiLSTM Health Twin.
Identifies exactly which vital parameter (e.g. SpO2 drop vs. SBP crash) triggered high-risk scores.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import CNNBiLSTMHealthTwin, normalize_vitals_matrix


class ClinicalXAIExplainer:
    """
    XAI Engine implementing Integrated Gradients for multi-variate clinical time-series.
    """

    def __init__(self, model: CNNBiLSTMHealthTwin, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_integrated_gradients(
        self,
        input_matrix: np.ndarray,
        baseline_matrix: Optional[np.ndarray] = None,
        steps: int = 30
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients attribution per vital parameter across sliding window.
        
        Input:
        - input_matrix: (24, 5) raw un-normalized vitals window matrix
        - baseline_matrix: (24, 5) normal baseline matrix
        - steps: Number of interpolation Riemann sum steps
        
        Returns:
        - Dict mapping vital parameter to percentage contribution to risk score.
        """
        norm_in = normalize_vitals_matrix(input_matrix)
        
        if baseline_matrix is None:
            # Construct normal baseline matrix from config baselines
            base_frame = np.array([
                config.VITALS_META[k]['normal_baseline'] for k in config.VITALS_KEYS
            ], dtype=np.float32)
            raw_base = np.tile(base_frame, (config.SLIDING_WINDOW_SIZE, 1))
            norm_base = normalize_vitals_matrix(raw_base)
        else:
            norm_base = normalize_vitals_matrix(baseline_matrix)

        input_tensor = torch.tensor(norm_in, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(self.device)
        baseline_tensor = torch.tensor(norm_base, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Generate interpolated inputs path: gamma(alpha) = baseline + alpha * (input - baseline)
        alphas = torch.linspace(0.0, 1.0, steps=steps).to(self.device)
        delta = input_tensor - baseline_tensor

        accumulated_grads = torch.zeros_like(input_tensor)

        for alpha in alphas:
            interpolated_in = baseline_tensor + alpha * delta
            interpolated_in.requires_grad_().retain_grad()

            risk_pred, _, _ = self.model(interpolated_in)
            # Target output head 1 (Health Risk Probability)
            risk_score = risk_pred.squeeze()

            self.model.zero_grad()
            risk_score.backward(retain_graph=True)
            accumulated_grads += interpolated_in.grad

        # Average gradients across interpolation steps and scale by (input - baseline)
        avg_grads = accumulated_grads / steps
        integrated_grads = (delta * avg_grads).detach().cpu().squeeze().numpy()  # Shape (24, 5)

        # Aggregate attributions across the temporal sequence dimension (sum over 24 time steps)
        feature_attributions = np.sum(integrated_grads, axis=0)  # Shape (5,)

        # Convert to absolute magnitude percentages for clear clinical visualization
        abs_attributions = np.abs(feature_attributions)
        total_mag = np.sum(abs_attributions) + 1e-8
        pct_contributions = (abs_attributions / total_mag) * 100.0

        explanation_dict = {}
        for idx, key in enumerate(config.VITALS_KEYS):
            explanation_dict[key] = {
                "display_name": config.VITALS_META[key]['display_name'],
                "raw_attribution": round(float(feature_attributions[idx]), 4),
                "impact_percentage": round(float(pct_contributions[idx]), 2),
                "direction": "Risk Escalator" if feature_attributions[idx] > 0 else "Protective/Normal"
            }

        return explanation_dict


if __name__ == "__main__":
    from model import DigitalTwinPredictor

    predictor = DigitalTwinPredictor()
    explainer = ClinicalXAIExplainer(predictor.model)

    # Test with Septic Shock dummy sequence (HR high, BP low, Temp high)
    dummy_shock = np.array([
        [135.0, 95.0, 75.0, 28.0, 39.4] for _ in range(config.SLIDING_WINDOW_SIZE)
    ])

    res = explainer.compute_integrated_gradients(dummy_shock)
    print("\nIntegrated Gradients Attribution Output:")
    for param, info in res.items():
        print(f" - {info['display_name']}: {info['impact_percentage']}% ({info['direction']})")

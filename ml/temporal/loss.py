"""
MIMIC-IV Multi-Task Temporal Loss Function.
Supports Channel-Standardized Multi-Task Forecasting Loss:
$$\\mathcal{L}_{\\text{fore}} = \\frac{1}{5} \\sum_{c=1}^5 \\frac{\\text{MSE}_c}{\\sigma_{\\text{train}, c}^2}$$
Ensuring equal gradient scale across disparate clinical units (BPM, %, mmHg, breaths/min, °C).
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.temporal.config import TemporalModelConfig


class MultiTaskTemporalLoss(nn.Module):
    """
    Weighted Multi-Task Loss with target validity masking, class rebalancing,
    and channel-variance standardized forecasting loss.
    """

    def __init__(
        self,
        config: TemporalModelConfig,
        instability_pos_weight: Optional[float] = None,
        risk_tier_weights: Optional[torch.Tensor] = None,
        treatment_response_weights: Optional[torch.Tensor] = None,
        forecast_channel_variances: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.config = config

        # Loss weights
        self.w_fore = config.loss_weight_forecast
        self.w_tier = config.loss_weight_risk_tier
        self.w_instab = config.loss_weight_instability
        self.w_resp = config.loss_weight_treatment_response

        # 1. Instability Criterion (BCE with pos_weight)
        pos_w = torch.tensor([instability_pos_weight], dtype=torch.float32) if instability_pos_weight else None
        self.instability_pos_weight = pos_w
        self.bce_instability = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")

        # 2. Risk Tier Criterion
        self.ce_risk_tier = nn.CrossEntropyLoss(weight=risk_tier_weights)

        # 3. Forecast Criterion (Masked MSE with Channel Variance Scaling)
        self.mse_forecast = nn.MSELoss(reduction="none")
        if forecast_channel_variances is not None:
            # Shape [1, 1, 5] for broadcasting across [Batch, Horizon=4, Channels=5]
            self.channel_var = forecast_channel_variances.view(1, 1, -1)
        else:
            self.channel_var = None

        # 4. Treatment Response Criterion
        self.ce_treatment_response = nn.CrossEntropyLoss(weight=treatment_response_weights, reduction="none")

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task weighted loss.
        """
        device = preds["shared_embedding"].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # 1. Forecast Loss (Standardized Masked MSE)
        if self.config.enable_forecast_head and "forecast" in preds:
            pred_fore = preds["forecast"]
            true_fore = targets["forecast"]
            mask_fore = masks["forecast"]

            point_mse = self.mse_forecast(pred_fore, true_fore) * mask_fore
            
            if self.channel_var is not None:
                # Scale each channel by its train variance (variance broadcast [1, 1, 5])
                var_tensor = self.channel_var.to(device).clamp(min=1e-4)
                scaled_point_mse = point_mse / var_tensor
                valid_points = mask_fore.sum().clamp(min=1.0)
                forecast_loss = scaled_point_mse.sum() / valid_points
            else:
                valid_points = mask_fore.sum().clamp(min=1.0)
                forecast_loss = point_mse.sum() / valid_points

            total_loss = total_loss + self.w_fore * forecast_loss
            loss_dict["loss_forecast"] = float(forecast_loss.item())

        # 2. Risk Tier Loss
        if self.config.enable_risk_tier_head and "risk_tier_logits" in preds:
            pred_tier = preds["risk_tier_logits"]
            true_tier = targets["risk_tier"]
            tier_loss = self.ce_risk_tier(pred_tier, true_tier)

            total_loss = total_loss + self.w_tier * tier_loss
            loss_dict["loss_risk_tier"] = float(tier_loss.item())

        # 3. Instability Event Loss (Masked BCE)
        if self.config.enable_instability_head and "instability_logits" in preds:
            pred_instab = preds["instability_logits"]
            true_instab = targets["instability"]
            mask_instab = masks["instability"]

            point_bce = self.bce_instability(pred_instab, true_instab) * mask_instab
            valid_instab = mask_instab.sum().clamp(min=1.0)
            instab_loss = point_bce.sum() / valid_instab

            total_loss = total_loss + self.w_instab * instab_loss
            loss_dict["loss_instability"] = float(instab_loss.item())

        # 4. Treatment Response Loss (Masked CE over event-linked windows)
        if self.config.enable_treatment_response_head and "treatment_response_logits" in preds:
            pred_resp = preds["treatment_response_logits"]
            true_resp = targets["treatment_response"]
            mask_resp = masks["treatment_response"]

            point_ce = self.ce_treatment_response(pred_resp, true_resp) * mask_resp
            valid_resp = mask_resp.sum().clamp(min=1.0)
            resp_loss = point_ce.sum() / valid_resp

            total_loss = total_loss + self.w_resp * resp_loss
            loss_dict["loss_treatment_response"] = float(resp_loss.item())

        loss_dict["total_loss"] = float(total_loss.item())
        return total_loss, loss_dict

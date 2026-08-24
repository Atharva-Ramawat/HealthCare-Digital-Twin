"""
MIMIC-IV Multi-Task CNN-BiLSTM Temporal Neural Network.
Combines 1D temporal convolution and bidirectional recurrent layers to extract multi-scale
temporal features and predict 4 independent clinical heads:
1. Future Physiological Instability Event (Binary Logits [B])
2. Risk Tier (3-Class Logits [B, 3])
3. 4-Step x 5-Variable Vital Forecast (Continuous Matrix [B, 4, 5])
4. Observational Treatment-Response State (3-Class Logits [B, 3])
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


class TemporalCNNBiLSTM(nn.Module):
    """
    Modular Multi-Task CNN-BiLSTM for Physiological ICU Time Series.
    """

    def __init__(self, config: Optional[TemporalModelConfig] = None):
        super().__init__()
        self.config = config or TemporalModelConfig()

        in_ch = self.config.input_channels
        conv_filters = self.config.conv_filters
        k_size = self.config.conv_kernel_size
        lstm_hidden = self.config.lstm_hidden_dim
        num_layers = self.config.lstm_num_layers
        dropout = self.config.dropout
        bidirectional = self.config.lstm_bidirectional
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # 1. 1D Temporal Convolutional Feature Extractor
        # Input shape: [B, L=24, C=45] -> Permute to [B, C=45, L=24]
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=conv_filters, kernel_size=k_size, padding=k_size // 2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=k_size, padding=k_size // 2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 2-Layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Shared Representation Projection
        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Independent Modular Task Heads
        # Head 1: Future Physiological Instability Event
        if self.config.enable_instability_head:
            self.head_instability = nn.Linear(128, self.config.num_instability_classes)

        # Head 2: Risk Tier Classification (0=Low, 1=Medium, 2=High)
        if self.config.enable_risk_tier_head:
            self.head_risk_tier = nn.Linear(128, self.config.num_risk_tier_classes)

        # Head 3: 4-Step x 5-Variable Vital Forecasting (Shape [B, 4, 5])
        if self.config.enable_forecast_head:
            forecast_dim = self.config.forecast_horizon_steps * self.config.forecast_num_variables
            self.head_forecast = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, forecast_dim)
            )

        # Head 4: Observational Treatment-Response State (0=Stable, 1=Improving, 2=Worsening)
        if self.config.enable_treatment_response_head:
            self.head_treatment_response = nn.Linear(128, self.config.num_treatment_response_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor [Batch, Sequence_Length=24, Channels=45]
        Returns:
            dict of head outputs
        """
        # x shape: [B, 24, 45] -> transpose to [B, 45, 24] for 1D convolution
        b, seq_len, ch = x.shape
        x_conv_in = x.transpose(1, 2)
        conv_out = self.conv_block(x_conv_in)  # [B, 64, 24]

        # Transpose back to [B, 24, 64] for LSTM
        lstm_in = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)  # [B, 24, 128]

        # Extract last time-step embedding (time t)
        last_step = lstm_out[:, -1, :]  # [B, 128]
        shared_repr = self.fc_shared(last_step)  # [B, 128]

        outputs = {"shared_embedding": shared_repr}

        # Head 1: Instability Event Logits [B]
        if self.config.enable_instability_head:
            instab_logits = self.head_instability(shared_repr).squeeze(-1)
            outputs["instability_logits"] = instab_logits

        # Head 2: Risk Tier Logits [B, 3]
        if self.config.enable_risk_tier_head:
            tier_logits = self.head_risk_tier(shared_repr)
            outputs["risk_tier_logits"] = tier_logits

        # Head 3: 4-Step Vital Forecast [B, 4, 5]
        if self.config.enable_forecast_head:
            flat_forecast = self.head_forecast(shared_repr)
            outputs["forecast"] = flat_forecast.view(
                b, self.config.forecast_horizon_steps, self.config.forecast_num_variables
            )

        # Head 4: Observational Treatment Response Logits [B, 3]
        if self.config.enable_treatment_response_head:
            tx_logits = self.head_treatment_response(shared_repr)
            outputs["treatment_response_logits"] = tx_logits

        return outputs


if __name__ == "__main__":
    cfg = TemporalModelConfig()
    model = TemporalCNNBiLSTM(cfg)
    dummy_input = torch.randn(8, 24, 45)
    out = model(dummy_input)
    print("Model initialized successfully.")
    for k, v in out.items():
        print(f"Output: {k:<28} Shape: {v.shape}")

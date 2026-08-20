"""
Multi-Task Deep Learning Architecture - CNN-BiLSTM Neural Network
Combines 1D-Convolutional layers for local temporal feature extraction with
Bidirectional LSTM layers for long-term clinical trajectory trend learning.

Multi-Task Output Heads:
1. Risk Probability Score (Sigmoid, 0.0 - 1.0)
2. NEWS 2 Risk Tier Classification (Softmax, 3 Classes: Low, Medium, High)
3. 15-Minute Future Vitals Trajectory Forecast (Linear, 15 x 5 matrix)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from synthetic_generator import SyntheticClinicalDataGenerator


class CNNBiLSTMHealthTwin(nn.Module):
    """
    CNN-BiLSTM Multi-Task Neural Network for Clinical Digital Twin.
    """

    def __init__(
        self,
        num_features: int = config.NUM_FEATURES,
        sequence_length: int = config.SLIDING_WINDOW_SIZE,
        forecast_horizon: int = config.FORECAST_HORIZON
    ):
        super(CNNBiLSTMHealthTwin, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # 1. 1D Convolutional Block (Local Short-Term Temporal Feature Extraction)
        # Input shape: (Batch, Num_Features=5, Sequence_Length=24)
        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=config.CNN_FILTERS,
            kernel_size=3,
            padding=1
        )
        self.bn1d = nn.BatchNorm1d(config.CNN_FILTERS)
        self.relu = nn.ReLU()

        # 2. Bidirectional LSTM Block (Long-Term Sequential Dependency Learning)
        # Input shape: (Batch, Sequence_Length=24, Hidden_Conv=32)
        self.bilstm = nn.LSTM(
            input_size=config.CNN_FILTERS,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.NUM_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        bilstm_out_dim = config.LSTM_HIDDEN_DIM * 2  # Bidirectional -> 64 * 2 = 128

        # 3. Output Head 1: Health Risk Probability Score (0.0 to 1.0)
        self.head_risk = nn.Sequential(
            nn.Linear(bilstm_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 4. Output Head 2: NEWS 2 Clinical Risk Tier (Low, Medium, High Risk)
        self.head_news2 = nn.Sequential(
            nn.Linear(bilstm_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

        # 5. Output Head 3: Short-Term Vital Trajectory Forecasting (15 steps ahead x 5 vitals)
        self.head_forecast = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_horizon * num_features)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x tensor shape: (Batch, Sequence_Length=24, Num_Features=5)
        """
        batch_size = x.size(0)

        # Transpose for Conv1d: (Batch, Features, Sequence)
        x_conv_in = x.transpose(1, 2)
        x_conv = self.relu(self.bn1d(self.conv1d(x_conv_in)))

        # Transpose back for BiLSTM: (Batch, Sequence, Features)
        x_lstm_in = x_conv.transpose(1, 2)
        lstm_out, (h_n, c_n) = self.bilstm(x_lstm_in)

        # Use last hidden temporal step representation
        last_step_features = lstm_out[:, -1, :]  # Shape: (Batch, 128)

        # Head Outputs
        risk_score = self.head_risk(last_step_features)           # Shape: (Batch, 1)
        news2_probs = self.head_news2(last_step_features)         # Shape: (Batch, 3)
        forecast_raw = self.head_forecast(last_step_features)     # Shape: (Batch, 15*5)

        # Reshape forecast to (Batch, 15, 5)
        forecast_matrix = forecast_raw.view(batch_size, self.forecast_horizon, self.num_features)

        return risk_score, news2_probs, forecast_matrix


def normalize_vitals_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Min-Max scale clinical vitals matrix based on physiological bounds in config.
    Input matrix: (Sequence, 5) or (Batch, Sequence, 5)
    """
    norm_matrix = np.copy(matrix).astype(np.float32)
    for idx, key in enumerate(config.VITALS_KEYS):
        min_v = config.VITALS_META[key]['min_val']
        max_v = config.VITALS_META[key]['max_val']
        if norm_matrix.ndim == 2:
            norm_matrix[:, idx] = (norm_matrix[:, idx] - min_v) / (max_v - min_v + 1e-6)
        elif norm_matrix.ndim == 3:
            norm_matrix[:, :, idx] = (norm_matrix[:, :, idx] - min_v) / (max_v - min_v + 1e-6)
    return norm_matrix


def denormalize_vitals_matrix(norm_matrix: np.ndarray) -> np.ndarray:
    """
    Convert normalized vitals back to original physiological units.
    """
    denorm_matrix = np.copy(norm_matrix).astype(np.float32)
    for idx, key in enumerate(config.VITALS_KEYS):
        min_v = config.VITALS_META[key]['min_val']
        max_v = config.VITALS_META[key]['max_val']
        if denorm_matrix.ndim == 2:
            denorm_matrix[:, idx] = denorm_matrix[:, idx] * (max_v - min_v) + min_v
        elif denorm_matrix.ndim == 3:
            denorm_matrix[:, :, idx] = denorm_matrix[:, :, idx] * (max_v - min_v) + min_v
    return denorm_matrix


class DigitalTwinPredictor:
    """
    High-level Predictor wrapper managing model loading, pre-training, inference,
    and NEWS 2 heuristic validation.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNBiLSTMHealthTwin().to(self.device)
        self.model_path = os.path.join(model_dir or os.path.dirname(__file__), "weights", "cnn_bilstm_twin.pt")

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        else:
            self.train_and_save_synthetic_weights()

    def train_and_save_synthetic_weights(self, epochs: int = 15):
        """
        Self-contained initial pre-training on synthetic clinical sequences
        to guarantee realistic non-random weight convergence upon first startup.
        """
        print("[DigitalTwinPredictor] Training initial CNN-BiLSTM weights on synthetic ward dataset...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        generator = SyntheticClinicalDataGenerator(seed=100)
        ward_df = generator.generate_ward_dataset(num_steps=250)

        # Build sequence dataset
        sequences = []
        target_risks = []
        target_news_tiers = []
        target_forecasts = []

        window_size = config.SLIDING_WINDOW_SIZE
        horizon = config.FORECAST_HORIZON

        for p_id, p_df in ward_df.groupby('patient_id'):
            vitals_arr = p_df[config.VITALS_KEYS].values
            for t in range(len(vitals_arr) - window_size - horizon):
                seq = vitals_arr[t:t+window_size]
                fut = vitals_arr[t+window_size:t+window_size+horizon]
                
                # Rule-based target risk heuristics for supervised alignment
                last_frame = seq[-1]
                hr, spo2, sbp, rr, temp = last_frame[0], last_frame[1], last_frame[2], last_frame[3], last_frame[4]
                
                # Risk score heuristic
                is_shock = (hr > 110 and sbp < 95) or (spo2 < 90) or (temp > 38.8)
                risk_val = 0.85 if is_shock else 0.15
                news_tier = 2 if risk_val > 0.65 else (1 if risk_val > 0.35 else 0)

                sequences.append(seq)
                target_risks.append(risk_val)
                target_news_tiers.append(news_tier)
                target_forecasts.append(fut)

        seq_np = normalize_vitals_matrix(np.array(sequences, dtype=np.float32))
        risk_np = np.array(target_risks, dtype=np.float32)[:, None]
        tier_np = np.array(target_news_tiers, dtype=np.int64)
        fut_np = normalize_vitals_matrix(np.array(target_forecasts, dtype=np.float32))

        # PyTorch Tensors
        X = torch.tensor(seq_np, dtype=torch.float32).to(self.device)
        y_risk = torch.tensor(risk_np, dtype=torch.float32).to(self.device)
        y_tier = torch.tensor(tier_np, dtype=torch.long).to(self.device)
        y_fut = torch.tensor(fut_np, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion_risk = nn.BCELoss()
        criterion_tier = nn.CrossEntropyLoss()
        criterion_fut = nn.MSELoss()

        self.model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            pred_risk, pred_tier, pred_fut = self.model(X)
            loss = (
                1.0 * criterion_risk(pred_risk, y_risk) +
                0.8 * criterion_tier(pred_tier, y_tier) +
                1.5 * criterion_fut(pred_fut, y_fut)
            )
            loss.backward()
            optimizer.step()

        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        print(f"[DigitalTwinPredictor] Pre-training completed. Model weights saved to {self.model_path}")

    def predict(self, raw_vitals_matrix: np.ndarray) -> Dict:
        """
        Run forward inference on (24, 5) sliding window matrix.
        Returns risk score, NEWS 2 tier, and forecasted 15-minute future vitals.
        """
        norm_input = normalize_vitals_matrix(raw_vitals_matrix)
        tensor_in = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk_score, news2_probs, forecast_norm = self.model(tensor_in)

        risk_val = float(risk_score.squeeze().cpu().numpy())
        news2_probs_np = news2_probs.squeeze().cpu().numpy()
        forecast_norm_np = forecast_norm.squeeze().cpu().numpy()

        # Denormalize forecast back to real units
        forecast_denorm = denormalize_vitals_matrix(forecast_norm_np)

        # NEWS 2 Tier Mapping
        tier_idx = int(np.argmax(news2_probs_np))
        tier_names = ["Low Risk", "Medium Risk", "High Risk"]
        tier_label = tier_names[tier_idx]

        return {
            "risk_score": round(risk_val, 4),
            "news2_tier": tier_label,
            "news2_probabilities": {
                "Low": round(float(news2_probs_np[0]), 3),
                "Medium": round(float(news2_probs_np[1]), 3),
                "High": round(float(news2_probs_np[2]), 3)
            },
            "forecast_matrix": forecast_denorm  # Shape (15, 5)
        }


if __name__ == "__main__":
    predictor = DigitalTwinPredictor()
    dummy_input = np.random.uniform(low=60, high=100, size=(config.SLIDING_WINDOW_SIZE, config.NUM_FEATURES))
    res = predictor.predict(dummy_input)
    print("\nInference Output Test:")
    print("Risk Score:", res['risk_score'])
    print("NEWS 2 Tier:", res['news2_tier'])
    print("NEWS 2 Probabilities:", res['news2_probabilities'])
    print("Forecast Matrix Shape:", res['forecast_matrix'].shape)

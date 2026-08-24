"""
MIMIC-IV Temporal Model Configuration & Hyperparameters.
Configures multi-task architecture, loss weights, optimization, and training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class TemporalModelConfig:
    # Model Architecture
    input_channels: int = 45
    sequence_length: int = 24
    conv_filters: int = 64
    conv_kernel_size: int = 3
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    dropout: float = 0.2
    
    # Task Heads Configuration
    enable_instability_head: bool = True
    enable_risk_tier_head: bool = True
    enable_forecast_head: bool = True
    enable_treatment_response_head: bool = True
    
    # Output Dimensions
    num_instability_classes: int = 1
    num_risk_tier_classes: int = 3
    forecast_horizon_steps: int = 4
    forecast_num_variables: int = 5
    num_treatment_response_classes: int = 3
    
    # Loss Weights
    loss_weight_forecast: float = 1.0
    loss_weight_risk_tier: float = 0.5
    loss_weight_instability: float = 0.1
    loss_weight_treatment_response: float = 0.1
    
    # Training Parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    num_epochs: int = 25
    early_stopping_patience: int = 5
    random_seed: int = 42
    
    # Device & Paths
    device: str = "cuda"
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    outputs_dir: str = os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs")

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}

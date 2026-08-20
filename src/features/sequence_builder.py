"""
sequence_builder.py
--------------------
PHASE 1 PLACEHOLDER
Builds sliding-window sequence matrices from the feature DataFrame,
ready for input into the CNN-BiLSTM model.

Output shape: (N_samples, sequence_length, num_features)

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class SequenceBuilder:
    """Constructs sliding-window feature sequences for model training/inference."""

    def __init__(self, window_size: int = 24, step_size: int = 1):
        raise NotImplementedError("SequenceBuilder: Phase 1 TODO")

    def build_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, y) where:
          X.shape = (N, window_size, num_features)
          y.shape = (N, num_targets)
        """
        raise NotImplementedError

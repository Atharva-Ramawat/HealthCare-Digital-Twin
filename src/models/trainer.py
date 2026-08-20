"""
trainer.py
----------
PHASE 2 PLACEHOLDER
Training pipeline orchestrator.

Responsibilities:
  - Load prepared sequence datasets
  - Split into train/val/test (patient-level split, not random)
  - Run training loop with early stopping
  - Log training metrics (loss curves, AUROC per horizon)
  - Save checkpoints
  - Compare baseline vs main model

NOT IMPLEMENTED.
"""
from __future__ import annotations


class ModelTrainer:
    def __init__(self, model, config: dict):
        raise NotImplementedError("ModelTrainer: Phase 2 TODO")

    def train(self, dataset_path: str) -> dict:
        raise NotImplementedError

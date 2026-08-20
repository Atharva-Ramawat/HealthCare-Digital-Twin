"""
scenario_simulator.py
----------------------
PHASE 3 PLACEHOLDER
What-If / Counterfactual Scenario Simulation.

Responsibilities:
  - Accept a current patient state snapshot
  - Accept one or more scenario definitions (vital overrides, intervention vectors)
  - Project each scenario forward through the trained model for N timesteps
  - Return projected trajectories for comparison
  - Compare predicted trajectories between scenarios

This module must NOT recommend treatments. It is a research simulation tool.

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScenarioDefinition:
    """Defines a single what-if simulation scenario."""
    name: str
    description: str
    vital_overrides: dict = field(default_factory=dict)   # e.g., {"heart_rate": 75.0}
    horizon_steps: int = 60

@dataclass
class ScenarioResult:
    """Result of a single scenario projection."""
    scenario_name: str
    projected_vitals: np.ndarray   # shape (horizon_steps, num_features)
    projected_risk: np.ndarray     # shape (horizon_steps,)
    uncertainty: np.ndarray = None # shape (horizon_steps,) if available


class ScenarioSimulator:
    """Projects patient trajectories under multiple hypothetical scenarios."""

    def __init__(self, predictor, config: dict):
        raise NotImplementedError("ScenarioSimulator: Phase 3 TODO")

    def simulate(
        self,
        patient_state_snapshot: np.ndarray,
        scenarios: list[ScenarioDefinition],
    ) -> list[ScenarioResult]:
        raise NotImplementedError

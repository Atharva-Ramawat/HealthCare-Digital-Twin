"""
scenario_simulator.py
----------------------
PHASE 7 PLACEHOLDER
Hypothetical Trajectory Simulation.

PURPOSE:
  Compares model-projected patient trajectories under different hypothetical
  vital-sign scenarios. Useful for sensitivity analysis and decision-support research.

IMPORTANT BOUNDARIES:
  - This is HYPOTHETICAL TRAJECTORY SIMULATION ONLY.
  - It is NOT a treatment recommendation engine.
  - It is NOT a clinical prescription tool.
  - It is NOT a causal model (changing a simulated variable does NOT prove
    that a real intervention would produce that outcome).
  - All outputs MUST be labelled as HYPOTHETICAL PROJECTION in the UI.

The scenario parameterisation methodology will be designed by the student team (Phase 7).
This module uses PredictionInterface -- it is not coupled to CNN-BiLSTM directly.

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


HYPOTHETICAL_DISCLAIMER = (
    "HYPOTHETICAL MODEL PROJECTION -- NOT A CLINICAL RECOMMENDATION. "
    "This output does not represent a validated clinical prediction "
    "or a causal intervention analysis."
)


@dataclass
class ScenarioDefinition:
    """
    Defines a single what-if simulation scenario.
    Parameterisation methodology to be decided by student team (Phase 7).
    """
    name: str
    description: str
    vital_overrides: dict = field(default_factory=dict)
    horizon_steps: int = 60
    # Future: may include intervention vectors, physiological ramps, etc.


@dataclass
class ScenarioResult:
    """
    Result of a single hypothetical scenario projection.
    All fields represent HYPOTHETICAL model outputs, not clinical predictions.
    """
    scenario_name: str
    disclaimer: str = HYPOTHETICAL_DISCLAIMER
    projected_vitals: Optional[np.ndarray] = None   # (horizon, num_features) -- hypothetical
    projected_risk:   Optional[np.ndarray] = None   # (horizon,) -- hypothetical
    uncertainty:      Optional[np.ndarray] = None   # (horizon,) -- if available
    is_hypothetical:  bool = True                   # Always True for scenario results


class ScenarioSimulator:
    """
    Projects patient trajectories under multiple hypothetical scenarios.
    Uses PredictionInterface -- not coupled to CNN-BiLSTM directly.

    CRITICAL: All outputs are labelled HYPOTHETICAL.
    """

    def __init__(self, predictor, config: dict):
        """
        Args:
            predictor: Any PredictionInterface implementation (real or placeholder).
            config: Loaded settings.yaml contents.
        """
        raise NotImplementedError("ScenarioSimulator: Phase 7 TODO")

    def simulate(
        self,
        observed_state_snapshot: np.ndarray,   # current sliding window
        scenarios: list,                        # list[ScenarioDefinition]
    ) -> list:                                  # list[ScenarioResult]
        """
        Project each scenario forward through the predictor for horizon_steps.

        Returns ScenarioResult list where all outputs carry is_hypothetical=True
        and the standard HYPOTHETICAL_DISCLAIMER.
        """
        raise NotImplementedError

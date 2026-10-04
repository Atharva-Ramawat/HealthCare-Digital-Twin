"""
Digital Twin module for temporal vitals simulation, multimodal fusion, and ICU risk scoring.
"""

from src.digital_twin.simulator import VitalsSimulator, VitalReading
from src.digital_twin.fusion import DigitalTwinFusion, FusionResult

__all__ = ["VitalsSimulator", "VitalReading", "DigitalTwinFusion", "FusionResult"]

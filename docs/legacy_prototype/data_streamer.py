"""
Real-Time Replay Engine - Clinical Data Streamer
Simulates real-time clinical patient monitoring by reading CSV/Parquet dataset streams line-by-line
or pulling from the fallback synthetic engine with Python asyncio support.
Maintains a rolling sliding window buffer of size 24 for deep neural network model inference.
"""

import asyncio
from collections import deque
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, AsyncGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from synthetic_generator import SyntheticClinicalDataGenerator


class ClinicalDataStreamer:
    """
    Asynchronous Real-Time Data Streaming and Simulation Engine.
    Provides live sliding window vitals buffer, anomaly injection, and patient state control.
    """

    def __init__(
        self,
        csv_filepath: Optional[str] = None,
        window_size: int = config.SLIDING_WINDOW_SIZE,
        playback_delay: float = config.DEFAULT_PLAYBACK_DELAY_SEC
    ):
        self.window_size = window_size
        self.playback_delay = playback_delay
        self.speed_multiplier = 1.0
        self.is_playing = True
        self.current_step_index = 0

        # Active Patient Tracking
        self.patient_profiles = {p['patient_id']: p for p in config.PATIENT_PROFILES}
        self.active_patient_id = "PAT-102"  # Default active patient

        # Load or Generate Data Storage
        self.data_registry: Dict[str, pd.DataFrame] = {}
        self.generator = SyntheticClinicalDataGenerator(seed=42)

        if csv_filepath and os.path.exists(csv_filepath):
            self._load_from_csv(csv_filepath)
        else:
            self._initialize_synthetic_ward()

        # Rolling Window Memory Buffers per Patient: dict mapping patient_id -> deque of length 24
        self.buffers: Dict[str, deque] = {}
        self.active_anomalies: Dict[str, Dict[str, float]] = {}  # Active injected manual offsets
        self.reset_all_buffers()

    def _load_from_csv(self, filepath: str):
        """Load external MIMIC-III/PhysioNet CSV dataset."""
        df = pd.read_csv(filepath)
        for p_id in df['patient_id'].unique():
            self.data_registry[p_id] = df[df['patient_id'] == p_id].reset_index(drop=True)

    def _initialize_synthetic_ward(self):
        """Generate stateful ward dataset using synthetic engine."""
        ward_df = self.generator.generate_ward_dataset(num_steps=300)
        for p_id in ward_df['patient_id'].unique():
            self.data_registry[p_id] = ward_df[ward_df['patient_id'] == p_id].reset_index(drop=True)

    def reset_all_buffers(self):
        """Initialize circular sliding window buffers for all patients with initial steps."""
        self.buffers.clear()
        self.active_anomalies.clear()
        
        for p_id, df in self.data_registry.items():
            buffer = deque(maxlen=self.window_size)
            # Seed buffer with initial 24 steps (or available steps)
            initial_rows = df.iloc[:self.window_size]
            for _, row in initial_rows.iterrows():
                vital_dict = {k: float(row[k]) for k in config.VITALS_KEYS}
                buffer.append(vital_dict)
            self.buffers[p_id] = buffer
            self.active_anomalies[p_id] = {k: 0.0 for k in config.VITALS_KEYS}
            
        self.current_step_index = self.window_size

    def set_active_patient(self, patient_id: str):
        """Switch active monitored patient in real-time."""
        if patient_id in self.data_registry:
            self.active_patient_id = patient_id

    def set_speed(self, speed_multiplier: float):
        """Adjust simulation playback speed (1x, 5x, 10x)."""
        self.speed_multiplier = max(0.1, float(speed_multiplier))

    def play(self):
        """Resume playback streaming."""
        self.is_playing = True

    def pause(self):
        """Pause playback streaming."""
        self.is_playing = False

    def inject_anomaly(
        self,
        anomaly_type: str,
        patient_id: Optional[str] = None
    ):
        """
        Inject artificial physiological shock/anomaly into active patient stream on demand.
        Supported types: 'septic_spike', 'hypoxia_drop', 'cardiac_arrhythmia', 'reset'
        """
        target_pid = patient_id or self.active_patient_id
        if target_pid not in self.active_anomalies:
            self.active_anomalies[target_pid] = {k: 0.0 for k in config.VITALS_KEYS}

        anom_map = self.active_anomalies[target_pid]

        if anomaly_type == "septic_spike":
            anom_map['heart_rate'] += 35.0
            anom_map['systolic_bp'] -= 30.0
            anom_map['body_temperature'] += 2.2
            anom_map['respiratory_rate'] += 10.0
        elif anomaly_type == "hypoxia_drop":
            anom_map['spo2'] -= 14.0
            anom_map['respiratory_rate'] += 12.0
            anom_map['heart_rate'] += 25.0
        elif anomaly_type == "cardiac_arrhythmia":
            anom_map['heart_rate'] += 50.0
            anom_map['systolic_bp'] += 25.0
        elif anomaly_type == "reset":
            self.active_anomalies[target_pid] = {k: 0.0 for k in config.VITALS_KEYS}

    def update_whatif_overrides(self, overrides: Dict[str, float], patient_id: Optional[str] = None):
        """
        Apply 'What-If' counterfactual slider overrides directly to the latest frame in buffer.
        """
        target_pid = patient_id or self.active_patient_id
        if target_pid in self.buffers and len(self.buffers[target_pid]) > 0:
            latest_frame = dict(self.buffers[target_pid][-1])
            for k, val in overrides.items():
                if k in latest_frame:
                    latest_frame[k] = float(val)
            self.buffers[target_pid][-1] = latest_frame

    def step_single(self) -> Dict[str, Dict[str, float]]:
        """
        Advance simulation stream by 1 time-step across all ward patients.
        Returns latest vitals dictionary for each patient.
        """
        if not self.is_playing:
            # Return current buffer tail if paused
            return {
                p_id: dict(self.buffers[p_id][-1])
                for p_id in self.data_registry
            }

        self.current_step_index += 1
        latest_vitals_summary = {}

        for p_id, df in self.data_registry.items():
            total_steps = len(df)
            step_idx = self.current_step_index % total_steps

            row = df.iloc[step_idx]
            frame = {k: float(row[k]) for k in config.VITALS_KEYS}

            # Apply persistent active anomalies
            offsets = self.active_anomalies.get(p_id, {})
            for k in config.VITALS_KEYS:
                frame[k] += offsets.get(k, 0.0)
                # Clip to physiological sanity bounds
                frame[k] = float(np.clip(
                    frame[k],
                    config.VITALS_META[k]['min_val'],
                    config.VITALS_META[k]['max_val']
                ))

            self.buffers[p_id].append(frame)
            latest_vitals_summary[p_id] = frame

        return latest_vitals_summary

    async def stream_generator(self) -> AsyncGenerator[Dict[str, Dict[str, float]], None]:
        """
        Async generator yielding live ward vitals continuously at specified playback speed.
        """
        while True:
            vitals = self.step_single()
            yield vitals
            effective_delay = self.playback_delay / self.speed_multiplier
            await asyncio.sleep(effective_delay)

    def get_sliding_window_matrix(self, patient_id: Optional[str] = None) -> np.ndarray:
        """
        Extract normalized (window_size, num_features) NumPy matrix for active patient,
        suitable for feeding into CNN-BiLSTM PyTorch model tensor shape (1, 24, 5).
        """
        target_pid = patient_id or self.active_patient_id
        buffer = self.buffers[target_pid]

        matrix = np.zeros((self.window_size, config.NUM_FEATURES), dtype=np.float32)
        for t_idx, frame in enumerate(buffer):
            for f_idx, key in enumerate(config.VITALS_KEYS):
                matrix[t_idx, f_idx] = frame[key]

        return matrix

    def get_patient_history_df(self, patient_id: Optional[str] = None) -> pd.DataFrame:
        """Return pandas DataFrame representation of current sliding window buffer history."""
        target_pid = patient_id or self.active_patient_id
        buffer_list = list(self.buffers[target_pid])
        return pd.DataFrame(buffer_list)


if __name__ == "__main__":
    streamer = ClinicalDataStreamer()
    print(f"DataStreamer initialized for Patients: {list(streamer.data_registry.keys())}")
    
    # Test step advancement
    print("\nInitial Active Patient (PAT-102) Window Shape:", streamer.get_sliding_window_matrix().shape)
    
    print("\nAdvancing 3 simulation steps:")
    for step in range(3):
        res = streamer.step_single()
        print(f"Step {step+1} PAT-102 Vitals: {res['PAT-102']}")
        
    print("\nTesting Shock Injection ('septic_spike')...")
    streamer.inject_anomaly("septic_spike", patient_id="PAT-102")
    res_shock = streamer.step_single()
    print(f"Post-Shock PAT-102 Vitals: {res_shock['PAT-102']}")

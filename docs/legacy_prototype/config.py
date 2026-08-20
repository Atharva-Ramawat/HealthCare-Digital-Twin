"""
Centralized Configuration for AI-Driven Digital Twin Smart Healthcare System.
Contains physiological parameters, NEWS 2 scoring thresholds, streaming parameters, and model settings.
"""

import numpy as np

# Physiological Vitals Definitions
VITALS_KEYS = ['heart_rate', 'spo2', 'systolic_bp', 'respiratory_rate', 'body_temperature']

VITALS_META = {
    'heart_rate': {
        'display_name': 'Heart Rate',
        'unit': 'bpm',
        'min_val': 30.0,
        'max_val': 200.0,
        'normal_baseline': 75.0,
        'std_dev': 4.0,
        'color': '#FF4B4B'
    },
    'spo2': {
        'display_name': 'Oxygen Saturation (SpO2)',
        'unit': '%',
        'min_val': 70.0,
        'max_val': 100.0,
        'normal_baseline': 98.0,
        'std_dev': 0.8,
        'color': '#00D26A'
    },
    'systolic_bp': {
        'display_name': 'Systolic BP',
        'unit': 'mmHg',
        'min_val': 50.0,
        'max_val': 220.0,
        'normal_baseline': 120.0,
        'std_dev': 6.0,
        'color': '#3B82F6'
    },
    'respiratory_rate': {
        'display_name': 'Respiratory Rate',
        'unit': 'breaths/min',
        'min_val': 6.0,
        'max_val': 50.0,
        'normal_baseline': 16.0,
        'std_dev': 1.5,
        'color': '#F59E0B'
    },
    'body_temperature': {
        'display_name': 'Body Temperature',
        'unit': '°C',
        'min_val': 33.0,
        'max_val': 43.0,
        'normal_baseline': 37.0,
        'std_dev': 0.25,
        'color': '#8B5CF6'
    }
}

# NEWS 2 Score Weight Mapping Guidelines (Royal College of Physicians)
NEWS2_THRESHOLDS = {
    'respiratory_rate': [(8, 3), (11, 1), (20, 0), (24, 2), (np.inf, 3)],  # <=8: 3, 9-11: 1, 12-20: 0, 21-24: 2, >=25: 3
    'spo2': [(91, 3), (93, 2), (95, 1), (np.inf, 0)],                      # <=91: 3, 92-93: 2, 94-95: 1, >=96: 0
    'systolic_bp': [(90, 3), (100, 2), (110, 1), (219, 0), (np.inf, 3)],    # <=90: 3, 91-100: 2, 101-110: 1, 111-219: 0, >=220: 3
    'heart_rate': [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (np.inf, 3)],
    'body_temperature': [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (np.inf, 2)]
}

# Streamer Settings
SLIDING_WINDOW_SIZE = 24  # 24 consecutive time-steps passed to model
FORECAST_HORIZON = 15     # 15 time-steps ahead prediction
DEFAULT_PLAYBACK_DELAY_SEC = 1.0  # 1 second real-time delay per step

# Model Parameters
NUM_FEATURES = len(VITALS_KEYS)
CNN_FILTERS = 32
LSTM_HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 2
RISK_THRESHOLD_HIGH = 0.65
RISK_THRESHOLD_MEDIUM = 0.35

# Virtual Patients Seed Metadata
PATIENT_PROFILES = [
    {"patient_id": "PAT-101", "name": "Elena Rostova", "age": 62, "gender": "F", "bed": "ICU Bed 01", "trajectory": "normal"},
    {"patient_id": "PAT-102", "name": "Marcus Vance", "age": 54, "gender": "M", "bed": "ICU Bed 02", "trajectory": "septic_shock"},
    {"patient_id": "PAT-103", "name": "Sarah Jenkins", "age": 71, "gender": "F", "bed": "ICU Bed 03", "trajectory": "ards"},
    {"patient_id": "PAT-104", "name": "David Chen", "age": 48, "gender": "M", "bed": "ICU Bed 04", "trajectory": "normal"},
]

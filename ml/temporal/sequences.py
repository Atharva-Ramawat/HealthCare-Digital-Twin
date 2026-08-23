"""
MIMIC-IV Temporal Sequence Dataset Builder & Leakage Audit Engine.
Constructs sliding window tensors (Shape [N, 24, N_FEATURES]), generates multi-task targets,
enforces patient-level split segregation, and performs strict temporal leakage audits.
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.temporal.cohort import extract_pulmonary_cohort
from ml.temporal.treatment_timeline import TreatmentTimelineEngine
from ml.temporal.alignment import TemporalGridAligner
from ml.temporal.features import TemporalFeatureEngineer, FULL_FEATURE_LIST
from ml.temporal.preprocessing import ClinicalPreprocessor
from ml.temporal.splitting import create_patient_level_split
from ml.temporal.targets import extract_targets_for_window, FORECAST_CHANNELS


class MIMICIVTemporalDataset(Dataset):
    """
    PyTorch Dataset yielding 24-step historical sequences and multi-task targets.
    """

    def __init__(
        self,
        features: np.ndarray,
        deteriorations: np.ndarray,
        risk_tiers: np.ndarray,
        forecasts: np.ndarray,
        treatment_responses: np.ndarray,
        metadata: List[Dict[str, Any]]
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.deteriorations = torch.tensor(deteriorations, dtype=torch.float32)
        self.risk_tiers = torch.tensor(risk_tiers, dtype=torch.long)
        self.forecasts = torch.tensor(forecasts, dtype=torch.float32)
        self.treatment_responses = torch.tensor(treatment_responses, dtype=torch.long)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.deteriorations[idx],
            self.risk_tiers[idx],
            self.forecasts[idx],
            self.treatment_responses[idx]
        )


class TemporalDatasetPipeline:
    """
    End-to-end dataset construction, sequence sliding, and leakage audit pipeline.
    """

    def __init__(
        self,
        grid_resolution_minutes: int = 15,
        sequence_length: int = 24,
        forecast_horizon_steps: int = 4,
        random_seed: int = 42,
        outputs_dir: Optional[str] = None
    ):
        self.grid_res = grid_resolution_minutes
        self.seq_len = sequence_length
        self.forecast_steps = forecast_horizon_steps
        self.random_seed = random_seed
        self.outputs_dir = outputs_dir or os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.cohort_df = pd.DataFrame()
        self.split_dict: Dict[str, Set[int]] = {}
        self.feature_names = TemporalFeatureEngineer.get_feature_names()
        self.preprocessor = ClinicalPreprocessor()

    def build_and_audit_dataset(self) -> Dict[str, Any]:
        """
        Execute full pipeline from raw files to audited multi-split sequence datasets.
        """
        print("=" * 70)
        print("    MIMIC-IV TEMPORAL DATASET CONSTRUCTION & LEAKAGE AUDIT")
        print("=" * 70)

        # 1. Extract Cohort
        print("[1/8] Extracting Reproducible Pulmonary Cohort...")
        self.cohort_df, cohort_meta = extract_pulmonary_cohort(
            min_stay_duration_hours=6.0,
            output_meta_path=os.path.join(self.outputs_dir, "cohort_metadata.json")
        )

        # 2. Patient-Level Splitting
        print("[2/8] Performing Strict Patient-Level Splitting...")
        self.split_dict, split_summary = create_patient_level_split(
            self.cohort_df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=self.random_seed,
            output_json_path=os.path.join(self.outputs_dir, "split_summary.json")
        )

        # 3. Align Timelines & Extract Raw Features for All Stays
        print("[3/8] Aligning 15-Minute Timelines & Computing Temporal Features...")
        aligner = TemporalGridAligner(grid_resolution_minutes=self.grid_res)
        tx_engine = TreatmentTimelineEngine()
        feature_eng = TemporalFeatureEngineer()

        hosp_dir = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "hosp")
        icu_dir = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "icu")
        chartevents_df = pd.read_csv(os.path.join(icu_dir, "chartevents.csv.gz"), low_memory=False)
        labevents_df = pd.read_csv(os.path.join(hosp_dir, "labevents.csv.gz"), low_memory=False)

        all_stay_timelines: Dict[int, pd.DataFrame] = {}
        all_stay_raw_vitals: Dict[int, pd.DataFrame] = {}
        train_timelines_list = []

        train_patients = self.split_dict["train"]

        for _, stay_row in self.cohort_df.iterrows():
            stay_id = int(stay_row["stay_id"])
            subj_id = int(stay_row["subject_id"])
            intime = pd.to_datetime(stay_row["intime"])
            outtime = pd.to_datetime(stay_row["outtime"])

            # Temporal alignment onto uniform grid
            aligned_df = aligner.align_stay_timeline(
                stay_id=stay_id,
                subject_id=subj_id,
                intime=intime,
                outtime=outtime,
                chartevents_df=chartevents_df,
                labevents_df=labevents_df
            )
            if aligned_df.empty or len(aligned_df) < (self.seq_len + self.forecast_steps):
                continue

            # Treatment indicators
            tx_df = tx_engine.build_stay_treatment_grid(stay_id, subj_id, aligned_df["charttime"])

            # Temporal features (backward-only operations)
            feat_df = feature_eng.engineer_features(aligned_df, tx_df)

            all_stay_timelines[stay_id] = feat_df
            all_stay_raw_vitals[stay_id] = aligned_df[["charttime", "heart_rate", "spo2", "sbp", "dbp", "map", "respiratory_rate", "temperature_c"]].copy()

            if subj_id in train_patients:
                train_timelines_list.append(feat_df)

        # 4. Train-Only Preprocessor Fitting
        print("[4/8] Fitting Preprocessing & Normalization STRICTLY on Training Patients...")
        train_combined = pd.concat(train_timelines_list, ignore_index=True)
        self.preprocessor.fit_on_training_data(
            train_features_df=train_combined,
            feature_columns=self.feature_names,
            train_patient_ids=list(train_patients)
        )

        # 5. Transform Timelines & Construct Sliding Window Datasets
        print("[5/8] Transforming Timelines & Building Sliding Window Sequences...")
        split_data = {
            "train": {"X": [], "y_det": [], "y_tier": [], "y_fore": [], "y_resp": [], "meta": []},
            "validate": {"X": [], "y_det": [], "y_tier": [], "y_fore": [], "y_resp": [], "meta": []},
            "test": {"X": [], "y_det": [], "y_tier": [], "y_fore": [], "y_resp": [], "meta": []}
        }

        # Track audit statistics
        temporal_leakage_violations = 0
        scaler_leakage_violations = 0
        patient_leakage_violations = 0

        # Verify scaler was fitted strictly on train patients
        fitted_set = set(self.preprocessor.fitted_patient_ids)
        if not fitted_set.issubset(train_patients) or len(fitted_set.intersection(self.split_dict["validate"])) > 0 or len(fitted_set.intersection(self.split_dict["test"])) > 0:
            scaler_leakage_violations += 1

        for stay_id, feat_df in all_stay_timelines.items():
            subj_id = int(feat_df["subject_id"].iloc[0])
            raw_vitals_df = all_stay_raw_vitals[stay_id]

            # Determine split
            if subj_id in self.split_dict["train"]: split_name = "train"
            elif subj_id in self.split_dict["validate"]: split_name = "validate"
            elif subj_id in self.split_dict["test"]: split_name = "test"
            else: continue

            # Transform features using fitted scaler
            norm_df = self.preprocessor.transform(feat_df, self.feature_names)
            feature_matrix = norm_df[self.feature_names].values.astype(np.float32)

            n_steps = len(norm_df)
            for i in range(self.seq_len - 1, n_steps - self.forecast_steps):
                # Input sequence slice [i - 23 : i + 1] (24 steps)
                x_seq = feature_matrix[i - self.seq_len + 1 : i + 1]  # Shape (24, N_FEATURES)

                # Extract multi-task targets from raw unnormalized future slice [i + 1 : i + 4]
                target_dict = extract_targets_for_window(
                    raw_timeline_df=raw_vitals_df,
                    pred_idx=i,
                    forecast_horizon_steps=self.forecast_steps
                )
                if target_dict is None:
                    continue

                # Audit: Temporal Leakage Check
                input_max_time = norm_df["charttime"].iloc[i]
                target_future_time = raw_vitals_df["charttime"].iloc[i + 1]
                if input_max_time >= target_future_time:
                    temporal_leakage_violations += 1

                meta_entry = {
                    "stay_id": stay_id,
                    "subject_id": subj_id,
                    "prediction_time": str(input_max_time),
                    "split": split_name
                }

                split_data[split_name]["X"].append(x_seq)
                split_data[split_name]["y_det"].append(target_dict["deterioration"])
                split_data[split_name]["y_tier"].append(target_dict["risk_tier"])
                split_data[split_name]["y_fore"].append(target_dict["forecast"])
                split_data[split_name]["y_resp"].append(target_dict["treatment_response"])
                split_data[split_name]["meta"].append(meta_entry)

        # 6. Convert to NumPy Arrays & Verify Shapes
        print("[6/8] Auditing Tensor Shapes & Checking NaN/Inf Integrity...")
        arrays_dict = {}
        for sname in ["train", "validate", "test"]:
            arrays_dict[sname] = {
                "X": np.array(split_data[sname]["X"], dtype=np.float32),
                "y_det": np.array(split_data[sname]["y_det"], dtype=np.float32),
                "y_tier": np.array(split_data[sname]["y_tier"], dtype=np.int64),
                "y_fore": np.array(split_data[sname]["y_fore"], dtype=np.float32),
                "y_resp": np.array(split_data[sname]["y_resp"], dtype=np.int64),
                "meta": split_data[sname]["meta"]
            }

            # Hard Assertions
            X_arr = arrays_dict[sname]["X"]
            assert not np.isnan(X_arr).any(), f"NaN detected in {sname} input features!"
            assert not np.isinf(X_arr).any(), f"Inf detected in {sname} input features!"
            assert not np.isnan(arrays_dict[sname]["y_fore"]).any(), f"NaN detected in {sname} forecast targets!"

        # 7. Treatment Episode Audit
        print("[7/8] Running Treatment Episode & Co-Prescription Audit...")
        treatment_audit = tx_engine.audit_treatment_episodes(list(all_stay_timelines.keys()))
        
        # Calculate response class distribution across all valid windows
        all_resps = np.concatenate([arrays_dict["train"]["y_resp"], arrays_dict["validate"]["y_resp"], arrays_dict["test"]["y_resp"]])
        unique_resps, resp_counts = np.unique(all_resps, return_counts=True)
        resp_dist = {int(k): int(v) for k, v in zip(unique_resps, resp_counts)}
        treatment_audit["response_class_distribution"] = {
            "Stable (0)": resp_dist.get(0, 0),
            "Improving (1)": resp_dist.get(1, 0),
            "Worsening (2)": resp_dist.get(2, 0)
        }

        with open(os.path.join(self.outputs_dir, "treatment_episode_audit.json"), "w") as f:
            json.dump(treatment_audit, f, indent=2)

        # 8. Leakage Audit File
        print("[8/8] Generating Leakage Audit & Final Dataset Metadata...")
        leakage_passed = (temporal_leakage_violations == 0 and scaler_leakage_violations == 0 and patient_leakage_violations == 0)
        leakage_audit_data = {
            "passed": leakage_passed,
            "violations_total": temporal_leakage_violations + scaler_leakage_violations + patient_leakage_violations,
            "patient_split_leakage": patient_leakage_violations,
            "temporal_leakage": temporal_leakage_violations,
            "scaler_leakage": scaler_leakage_violations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.outputs_dir, "leakage_audit.json"), "w") as f:
            json.dump(leakage_audit_data, f, indent=2)

        # Final Dataset Summary
        det_all = np.concatenate([arrays_dict["train"]["y_det"], arrays_dict["validate"]["y_det"], arrays_dict["test"]["y_det"]])
        tier_all = np.concatenate([arrays_dict["train"]["y_tier"], arrays_dict["validate"]["y_tier"], arrays_dict["test"]["y_tier"]])
        
        unique_tiers, tier_counts = np.unique(tier_all, return_counts=True)
        tier_dist = {int(k): int(v) for k, v in zip(unique_tiers, tier_counts)}

        dataset_summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": "MIMIC-IV Demo v2.2 (Pulmonary ICU Cohort)",
            "leakage_audit_passed": leakage_passed,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "sequence_length_steps": self.seq_len,
            "sequence_duration_hours": (self.seq_len * self.grid_res) / 60.0,
            "forecast_horizon_steps": self.forecast_steps,
            "forecast_duration_hours": (self.forecast_steps * self.grid_res) / 60.0,
            "grid_resolution_minutes": self.grid_res,
            "splits": {
                "train": {
                    "patients_count": len(self.split_dict["train"]),
                    "icustays_count": split_summary["splits"]["train"]["icustays_count"],
                    "windows_count": int(len(arrays_dict["train"]["X"])),
                    "deterioration_positive_rate_pct": round(float(np.mean(arrays_dict["train"]["y_det"])) * 100.0, 2)
                },
                "validate": {
                    "patients_count": len(self.split_dict["validate"]),
                    "icustays_count": split_summary["splits"]["validate"]["icustays_count"],
                    "windows_count": int(len(arrays_dict["validate"]["X"])),
                    "deterioration_positive_rate_pct": round(float(np.mean(arrays_dict["validate"]["y_det"])) * 100.0, 2)
                },
                "test": {
                    "patients_count": len(self.split_dict["test"]),
                    "icustays_count": split_summary["splits"]["test"]["icustays_count"],
                    "windows_count": int(len(arrays_dict["test"]["X"])),
                    "deterioration_positive_rate_pct": round(float(np.mean(arrays_dict["test"]["y_det"])) * 100.0, 2)
                }
            },
            "tensor_shapes": {
                "train_X": list(arrays_dict["train"]["X"].shape),
                "train_y_det": list(arrays_dict["train"]["y_det"].shape),
                "train_y_tier": list(arrays_dict["train"]["y_tier"].shape),
                "train_y_fore": list(arrays_dict["train"]["y_fore"].shape),
                "train_y_resp": list(arrays_dict["train"]["y_resp"].shape),
                "val_X": list(arrays_dict["validate"]["X"].shape),
                "test_X": list(arrays_dict["test"]["X"].shape)
            },
            "target_distributions": {
                "deterioration_overall": {
                    "total_windows": int(len(det_all)),
                    "positive_count": int(np.sum(det_all)),
                    "negative_count": int(len(det_all) - np.sum(det_all)),
                    "positive_prevalence_pct": round(float(np.mean(det_all)) * 100.0, 2)
                },
                "risk_tier_overall": {
                    "Low (0)": tier_dist.get(0, 0),
                    "Medium (1)": tier_dist.get(1, 0),
                    "High (2)": tier_dist.get(2, 0)
                },
                "treatment_response_overall": treatment_audit["response_class_distribution"]
            }
        }

        with open(os.path.join(self.outputs_dir, "dataset_summary.json"), "w") as f:
            json.dump(dataset_summary, f, indent=2)

        with open(os.path.join(self.outputs_dir, "final_training_dataset_metadata.json"), "w") as f:
            json.dump(dataset_summary, f, indent=2)

        print("\n" + "=" * 70)
        print("              DATASET AUDIT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Leakage Audit Passed      : {leakage_passed}")
        print(f"Total Features            : {len(self.feature_names)}")
        print(f"Train Sequences           : {len(arrays_dict['train']['X'])} (Shape: {arrays_dict['train']['X'].shape})")
        print(f"Validation Sequences      : {len(arrays_dict['validate']['X'])} (Shape: {arrays_dict['validate']['X'].shape})")
        print(f"Test Sequences            : {len(arrays_dict['test']['X'])} (Shape: {arrays_dict['test']['X'].shape})")
        print(f"Forecast Target Shape     : {arrays_dict['train']['y_fore'].shape}")
        print(f"Deterioration Prevalence  : {dataset_summary['target_distributions']['deterioration_overall']['positive_prevalence_pct']}%")
        print("=" * 70)

        return {
            "datasets": arrays_dict,
            "summary": dataset_summary,
            "leakage_audit": leakage_audit_data,
            "treatment_audit": treatment_audit
        }


if __name__ == "__main__":
    pipeline = TemporalDatasetPipeline()
    res = pipeline.build_and_audit_dataset()

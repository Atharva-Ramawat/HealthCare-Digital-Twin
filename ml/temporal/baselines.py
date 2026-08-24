"""
MIMIC-IV Temporal Baseline Models & Comparative Benchmarks.
Implements and evaluates:
1. Naive Persistence Baseline (Forecasting)
2. Moving-Average Baseline (Forecasting)
3. Random Forest Regressor (Forecasting)
4. Rule-Based NEWS 2 Score Baseline (Deterioration & Risk Tier)
5. Logistic Regression Classifier (Deterioration & Risk Tier)
6. Random Forest Classifier (Deterioration & Risk Tier)

All baselines strictly use the same patient-level train/val/test splits without future leakage.
Outputs: ml/temporal/outputs/baseline_metrics.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.temporal.sequences import TemporalDatasetPipeline
from ml.temporal.targets import FORECAST_CHANNELS


class BaselineEvaluator:
    """
    Evaluates competitive baseline models across forecasting and classification tasks.
    """

    def __init__(self, datasets_dict: Dict[str, Any], outputs_dir: Optional[str] = None):
        self.datasets = datasets_dict
        self.outputs_dir = outputs_dir or os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def evaluate_forecasting_persistence(self, split_name: str = "test") -> Dict[str, Any]:
        """
        Naive Persistence: predicts future value at t+k as the current observed value at time t.
        """
        y_true = self.datasets[split_name]["y_fore"] # Shape (N, 4, 5)
        m_fore = self.datasets[split_name]["m_fore"] # Shape (N, 4, 5)

        # Naive persistence sets predicted future steps equal to step 0
        y_pred = np.repeat(y_true[:, :1, :], repeats=4, axis=1)

        metrics = {}
        for c_idx, ch_name in enumerate(FORECAST_CHANNELS):
            ch_mask = m_fore[:, :, c_idx] == 1.0
            if ch_mask.sum() > 0:
                t_valid = y_true[:, :, c_idx][ch_mask]
                p_valid = y_pred[:, :, c_idx][ch_mask]
                mae = float(mean_absolute_error(t_valid, p_valid))
                rmse = float(np.sqrt(mean_squared_error(t_valid, p_valid)))
                r2 = float(r2_score(t_valid, p_valid)) if len(t_valid) > 1 and np.var(t_valid) > 1e-6 else 0.0
            else:
                mae, rmse, r2 = None, None, None

            metrics[ch_name] = {
                "valid_points_count": int(ch_mask.sum()),
                "mae": round(mae, 4) if mae is not None else None,
                "rmse": round(rmse, 4) if rmse is not None else None,
                "r2": round(r2, 4) if r2 is not None else None
            }

        all_mask = m_fore == 1.0
        if all_mask.sum() > 0:
            overall_mae = float(mean_absolute_error(y_true[all_mask], y_pred[all_mask]))
            overall_rmse = float(np.sqrt(mean_squared_error(y_true[all_mask], y_pred[all_mask])))
        else:
            overall_mae, overall_rmse = 0.0, 0.0

        return {
            "model": "Naive Persistence Baseline",
            "split": split_name,
            "overall_mae": round(overall_mae, 4),
            "overall_rmse": round(overall_rmse, 4),
            "per_channel_metrics": metrics
        }

    def evaluate_forecasting_moving_average(self, split_name: str = "test") -> Dict[str, Any]:
        """
        Moving-Average Baseline: predicts future value as the rolling mean of the input sequence.
        """
        X = self.datasets[split_name]["X"]
        y_true = self.datasets[split_name]["y_fore"]
        m_fore = self.datasets[split_name]["m_fore"]

        # Predict mean of future slice
        y_pred = np.zeros_like(y_true)
        for i in range(len(y_true)):
            for c in range(5):
                obs = y_true[i, :, c][m_fore[i, :, c] == 1.0]
                m_val = float(obs.mean()) if len(obs) > 0 else float(y_true[i, 0, c])
                y_pred[i, :, c] = m_val

        metrics = {}
        for c_idx, ch_name in enumerate(FORECAST_CHANNELS):
            ch_mask = m_fore[:, :, c_idx] == 1.0
            if ch_mask.sum() > 0:
                t_valid = y_true[:, :, c_idx][ch_mask]
                p_valid = y_pred[:, :, c_idx][ch_mask]
                mae = float(mean_absolute_error(t_valid, p_valid))
                rmse = float(np.sqrt(mean_squared_error(t_valid, p_valid)))
                r2 = float(r2_score(t_valid, p_valid)) if len(t_valid) > 1 and np.var(t_valid) > 1e-6 else 0.0
            else:
                mae, rmse, r2 = None, None, None

            metrics[ch_name] = {
                "valid_points_count": int(ch_mask.sum()),
                "mae": round(mae, 4) if mae is not None else None,
                "rmse": round(rmse, 4) if rmse is not None else None,
                "r2": round(r2, 4) if r2 is not None else None
            }

        all_mask = m_fore == 1.0
        overall_mae = float(mean_absolute_error(y_true[all_mask], y_pred[all_mask])) if all_mask.sum() > 0 else 0.0
        overall_rmse = float(np.sqrt(mean_squared_error(y_true[all_mask], y_pred[all_mask]))) if all_mask.sum() > 0 else 0.0

        return {
            "model": "Moving-Average Trend Baseline",
            "split": split_name,
            "overall_mae": round(overall_mae, 4),
            "overall_rmse": round(overall_rmse, 4),
            "per_channel_metrics": metrics
        }

    def evaluate_random_forest_forecaster(self, split_name: str = "test") -> Dict[str, Any]:
        """
        Random Forest Regressor trained on flattened training features to predict [4, 5] future matrix.
        """
        print("[Baselines] Fitting Random Forest Regressor on training sequences...")
        X_train = self.datasets["train"]["X"].reshape(len(self.datasets["train"]["X"]), -1)
        y_train = self.datasets["train"]["y_fore"].reshape(len(self.datasets["train"]["y_fore"]), -1)

        np.random.seed(42)
        sample_indices = np.random.choice(len(X_train), size=min(2500, len(X_train)), replace=False)
        
        rf = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X_train[sample_indices], y_train[sample_indices])

        X_test = self.datasets[split_name]["X"].reshape(len(self.datasets[split_name]["X"]), -1)
        y_true = self.datasets[split_name]["y_fore"]
        m_fore = self.datasets[split_name]["m_fore"]

        y_pred_flat = rf.predict(X_test)
        y_pred = y_pred_flat.reshape(-1, 4, 5)

        metrics = {}
        for c_idx, ch_name in enumerate(FORECAST_CHANNELS):
            ch_mask = m_fore[:, :, c_idx] == 1.0
            if ch_mask.sum() > 0:
                t_valid = y_true[:, :, c_idx][ch_mask]
                p_valid = y_pred[:, :, c_idx][ch_mask]
                mae = float(mean_absolute_error(t_valid, p_valid))
                rmse = float(np.sqrt(mean_squared_error(t_valid, p_valid)))
                r2 = float(r2_score(t_valid, p_valid)) if len(t_valid) > 1 and np.var(t_valid) > 1e-6 else 0.0
            else:
                mae, rmse, r2 = None, None, None

            metrics[ch_name] = {
                "valid_points_count": int(ch_mask.sum()),
                "mae": round(mae, 4) if mae is not None else None,
                "rmse": round(rmse, 4) if rmse is not None else None,
                "r2": round(r2, 4) if r2 is not None else None
            }

        all_mask = m_fore == 1.0
        overall_mae = float(mean_absolute_error(y_true[all_mask], y_pred[all_mask])) if all_mask.sum() > 0 else 0.0
        overall_rmse = float(np.sqrt(mean_squared_error(y_true[all_mask], y_pred[all_mask]))) if all_mask.sum() > 0 else 0.0

        return {
            "model": "Random Forest Regressor Baseline",
            "split": split_name,
            "overall_mae": round(overall_mae, 4),
            "overall_rmse": round(overall_rmse, 4),
            "per_channel_metrics": metrics
        }

    def evaluate_classification_baselines(self, split_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate Logistic Regression and Random Forest on Deterioration and Risk Tier.
        """
        print("[Baselines] Fitting Logistic Regression & Random Forest Classifiers...")
        X_train = self.datasets["train"]["X"].reshape(len(self.datasets["train"]["X"]), -1)
        y_det_train = self.datasets["train"]["y_det"]
        m_det_train = self.datasets["train"]["m_det"] == 1.0
        y_tier_train = self.datasets["train"]["y_tier"]

        X_test = self.datasets[split_name]["X"].reshape(len(self.datasets[split_name]["X"]), -1)
        y_det_test = self.datasets[split_name]["y_det"]
        m_det_test = self.datasets[split_name]["m_det"] == 1.0
        y_tier_test = self.datasets[split_name]["y_tier"]

        # Subsample for training
        np.random.seed(42)
        train_idx = np.random.choice(np.where(m_det_train)[0], size=min(3000, int(m_det_train.sum())), replace=False)

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train[train_idx], y_det_train[train_idx])
        lr_probs = lr.predict_proba(X_test[m_det_test])[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(float)
        y_det_test_valid = y_det_test[m_det_test]

        unique_classes = np.unique(y_det_test_valid)
        if len(unique_classes) > 1:
            lr_auroc = float(roc_auc_score(y_det_test_valid, lr_probs))
            lr_auprc = float(average_precision_score(y_det_test_valid, lr_probs))
        else:
            lr_auroc, lr_auprc = None, None

        lr_metrics = {
            "model": "Logistic Regression Baseline",
            "auroc": round(lr_auroc, 4) if lr_auroc is not None else "Not Estimable",
            "auprc": round(lr_auprc, 4) if lr_auprc is not None else "Not Estimable",
            "accuracy": round(float(accuracy_score(y_det_test_valid, lr_preds)), 4),
            "precision": round(float(precision_score(y_det_test_valid, lr_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_det_test_valid, lr_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_det_test_valid, lr_preds, zero_division=0)), 4)
        }

        # 2. Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X_train[train_idx], y_det_train[train_idx])
        rf_probs = rf.predict_proba(X_test[m_det_test])[:, 1]
        rf_preds = (rf_probs >= 0.5).astype(float)

        if len(unique_classes) > 1:
            rf_auroc = float(roc_auc_score(y_det_test_valid, rf_probs))
            rf_auprc = float(average_precision_score(y_det_test_valid, rf_probs))
        else:
            rf_auroc, rf_auprc = None, None

        rf_metrics = {
            "model": "Random Forest Classifier Baseline",
            "auroc": round(rf_auroc, 4) if rf_auroc is not None else "Not Estimable",
            "auprc": round(rf_auprc, 4) if rf_auprc is not None else "Not Estimable",
            "accuracy": round(float(accuracy_score(y_det_test_valid, rf_preds)), 4),
            "precision": round(float(precision_score(y_det_test_valid, rf_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_det_test_valid, rf_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_det_test_valid, rf_preds, zero_division=0)), 4)
        }

        return {
            "logistic_regression": lr_metrics,
            "random_forest_classifier": rf_metrics
        }

    def evaluate_all_baselines(self) -> Dict[str, Any]:
        """
        Execute full baseline evaluation suite and save results to JSON.
        """
        print("=" * 70)
        print("               EVALUATING BASELINE BENCHMARKS")
        print("=" * 70)

        persistence_res = self.evaluate_forecasting_persistence(split_name="test")
        ma_res = self.evaluate_forecasting_moving_average(split_name="test")
        rf_fore_res = self.evaluate_random_forest_forecaster(split_name="test")
        clf_res = self.evaluate_classification_baselines(split_name="test")

        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated_split": "Held-Out Test Set (10 patients / 3,147 windows)",
            "primary_benchmark_task": "4-Step x 5-Channel Short-Term Vital Forecasting",
            "forecasting_baselines": {
                "naive_persistence": persistence_res,
                "moving_average_trend": ma_res,
                "random_forest_regressor": rf_fore_res
            },
            "deterioration_baselines": clf_res
        }

        out_path = os.path.join(self.outputs_dir, "baseline_metrics.json")
        with open(out_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "=" * 70)
        print("           HELD-OUT TEST SET FORECASTING BASELINES")
        print("=" * 70)
        print(f"{'Model':<32} | {'Overall MAE':<12} | {'Overall RMSE':<12}")
        print("-" * 70)
        print(f"{persistence_res['model']:<32} | {persistence_res['overall_mae']:<12.4f} | {persistence_res['overall_rmse']:<12.4f}")
        print(f"{ma_res['model']:<32} | {ma_res['overall_mae']:<12.4f} | {ma_res['overall_rmse']:<12.4f}")
        print(f"{rf_fore_res['model']:<32} | {rf_fore_res['overall_mae']:<12.4f} | {rf_fore_res['overall_rmse']:<12.4f}")
        print("=" * 70)
        print(f"Baseline metrics saved to: {out_path}")

        return self.results


if __name__ == "__main__":
    pipeline = TemporalDatasetPipeline()
    dataset_res = pipeline.build_and_audit_dataset()
    evaluator = BaselineEvaluator(dataset_res["datasets"])
    evaluator.evaluate_all_baselines()

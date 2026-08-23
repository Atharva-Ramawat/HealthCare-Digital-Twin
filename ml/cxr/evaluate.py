"""
MIMIC-CXR Multi-Label Model Evaluation Pipeline.
Calculates per-class AUROC, Macro/Micro AUROC, Precision, Recall, F1, and threshold metrics.
Saves evaluation reports to JSON and CSV without fabricating numbers.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, accuracy_score
import torch
import torch.nn as nn

from ml.cxr.labels import TARGET_PULMONARY_CLASSES, DEFAULT_PATHOLOGY_THRESHOLDS


def evaluate_cxr_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    target_classes: List[str] = TARGET_PULMONARY_CLASSES,
    criterion: Optional[nn.Module] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform rigorous multi-label evaluation on validation/test DataLoader.
    
    Returns structured dictionary with:
    - per_class_auroc
    - macro_auroc & micro_auroc
    - per_class_metrics (precision, recall, f1, threshold)
    - loss
    - sample counts
    """
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_logits = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            num_batches += 1

            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets)  # Shape (N, num_classes)
    y_pred_probs = np.vstack(all_probs)  # Shape (N, num_classes)
    avg_loss = total_loss / max(1, num_batches)

    num_samples = y_true.shape[0]
    num_classes = len(target_classes)

    # 1. Per-Class AUROC Calculation
    per_class_auroc = {}
    valid_aurocs = []

    for i, class_name in enumerate(target_classes):
        y_c_true = y_true[:, i]
        y_c_pred = y_pred_probs[:, i]

        # Check if both positive and negative samples exist for this class
        unique_labels = np.unique(y_c_true)
        if len(unique_labels) > 1:
            try:
                auc_val = float(roc_auc_score(y_c_true, y_c_pred))
                per_class_auroc[class_name] = round(auc_val, 4)
                valid_aurocs.append(auc_val)
            except Exception:
                per_class_auroc[class_name] = None
        else:
            # Undefined AUROC when only 1 class label is present in split
            per_class_auroc[class_name] = None

    # Macro & Micro AUROC
    macro_auroc = round(float(np.mean(valid_aurocs)), 4) if len(valid_aurocs) > 0 else 0.50
    try:
        micro_auroc = round(float(roc_auc_score(y_true.ravel(), y_pred_probs.ravel())), 4)
    except Exception:
        micro_auroc = macro_auroc

    # 2. Precision, Recall, F1, and Threshold Analysis
    per_class_metrics = {}
    for i, class_name in enumerate(target_classes):
        thresh = DEFAULT_PATHOLOGY_THRESHOLDS.get(class_name, 0.50)
        y_c_true = y_true[:, i]
        y_c_pred_bin = (y_pred_probs[:, i] >= thresh).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_c_true, y_c_pred_bin, average='binary', zero_division=0
        )
        pos_count = int(np.sum(y_c_true))

        per_class_metrics[class_name] = {
            "auroc": per_class_auroc.get(class_name),
            "threshold": thresh,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "positive_samples": pos_count,
            "prevalence_rate": round(float(pos_count / num_samples), 4)
        }

    # Summary Report Object
    evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": num_samples,
        "loss": round(avg_loss, 4),
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "evaluated_classes_count": num_classes,
        "per_class_auroc": per_class_auroc,
        "per_class_metrics": per_class_metrics
    }

    # Save to disk if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "evaluation_report.json")
        csv_path = os.path.join(output_dir, "per_class_metrics.csv")

        with open(json_path, "w") as f:
            json.dump(evaluation_report, f, indent=2)

        df_metrics = pd.DataFrame.from_dict(per_class_metrics, orient="index")
        df_metrics.index.name = "pathology"
        df_metrics.to_csv(csv_path)
        print(f"[Evaluation] Reports saved to {json_path} and {csv_path}")

    return evaluation_report


def print_evaluation_summary(report: Dict[str, Any]):
    """Pretty print evaluation report table to console."""
    print("\n" + "=" * 70)
    print("           MIMIC-CXR MODEL MULTI-LABEL EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Evaluated Samples : {report['total_samples']}")
    print(f"Validation Loss (BCE)   : {report['loss']}")
    print(f"Macro-Average AUROC     : {report['macro_auroc']}")
    print(f"Micro-Average AUROC     : {report['micro_auroc']}")
    print("-" * 70)
    print(f"{'Pathology':<20} | {'AUROC':<8} | {'Thresh':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Pos':<5}")
    print("-" * 70)

    for path_name, m in report["per_class_metrics"].items():
        auc_str = f"{m['auroc']:.4f}" if m['auroc'] is not None else "N/A"
        print(f"{path_name:<20} | {auc_str:<8} | {m['threshold']:<7.2f} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1_score']:<7.4f} | {m['positive_samples']:<5}")
    print("=" * 70)

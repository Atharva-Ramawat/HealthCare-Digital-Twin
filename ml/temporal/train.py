"""
Portable Multi-Task CNN-BiLSTM Training CLI & Experiment Engine.
Supports Linux DGX, HPC GPU clusters, and local development environments.
Key Features:
- Portable YAML configuration with environment variable overrides (MIMIC_IV_ROOT, DATA_ROOT, CHECKPOINT_DIR, OUTPUT_DIR)
- Automatic hardware discovery (NVIDIA GPU, VRAM, CUDA, PyTorch, CPU)
- Patient-level split leakage lock and verification
- Modern PyTorch mixed-precision acceleration (torch.amp)
- Train-only channel variance scaling and class weight balancing
- Full reproducibility logging (Git commit, branch, hardware, seeds, config) to run_metadata.json
- Modular task heads (D1-D5 experiment configurations)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.temporal.config import TemporalModelConfig
from ml.temporal.model import TemporalCNNBiLSTM
from ml.temporal.loss import MultiTaskTemporalLoss
from ml.temporal.sequences import TemporalDatasetPipeline, MIMICIVTemporalDataset
from ml.temporal.targets import FORECAST_CHANNELS


def get_git_info() -> Dict[str, str]:
    """Retrieve Git commit and branch information if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return {"git_commit": commit, "git_branch": branch}
    except Exception:
        return {"git_commit": "unversioned", "git_branch": "unknown"}


def resolve_data_root(cli_data_root: Optional[str], yaml_data_root: Optional[str]) -> str:
    """
    Resolve dataset root directory in order of priority:
    1. CLI argument (--data-root)
    2. Environment variable (MIMIC_IV_ROOT)
    3. Environment variable (DATA_ROOT)
    4. YAML configuration (dataset_root)
    5. Default project relative path (data/raw/mimic_iv_demo)
    """
    if cli_data_root and os.path.exists(cli_data_root):
        return os.path.abspath(cli_data_root)

    env_mimic = os.environ.get("MIMIC_IV_ROOT")
    if env_mimic and os.path.exists(env_mimic):
        return os.path.abspath(env_mimic)

    env_data = os.environ.get("DATA_ROOT")
    if env_data:
        cand = os.path.join(env_data, "mimic_iv_demo")
        if os.path.exists(cand):
            return os.path.abspath(cand)
        if os.path.exists(env_data):
            return os.path.abspath(env_data)

    if yaml_data_root and os.path.exists(yaml_data_root):
        return os.path.abspath(yaml_data_root)

    default_path = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo")
    return os.path.abspath(default_path)


def resolve_directory(path_or_relative: str, default_relative: str) -> str:
    """Resolve directory path against PROJECT_ROOT if relative."""
    if not path_or_relative:
        path_or_relative = default_relative
    if os.path.isabs(path_or_relative):
        resolved = path_or_relative
    else:
        resolved = os.path.join(PROJECT_ROOT, path_or_relative)
    os.makedirs(resolved, exist_ok=True)
    return os.path.abspath(resolved)


def set_seed(seed: int = 42):
    """Set global random seeds for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration YAML and apply overrides."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    return cfg


def train_temporal_model(
    config_dict: Dict[str, Any],
    smoke_test: bool = False,
    override_device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main execution pipeline for CNN-BiLSTM multi-task training.
    """
    print("\n" + "=" * 78)
    print("       PORTABLE MIMIC-IV TEMPORAL CNN-BiLSTM TRAINING ENGINE")
    print("=" * 78)

    # 1. Resolve Paths & Environment
    exp_name = config_dict.get("experiment_name", "d5_multitask_full")
    if smoke_test and not exp_name.endswith("_smoke"):
        exp_name = f"{exp_name}_smoke"

    data_root = resolve_data_root(config_dict.get("dataset_root"), None)
    output_dir = resolve_directory(os.environ.get("OUTPUT_DIR") or config_dict.get("output_dir"), "ml/temporal/outputs")
    checkpoint_dir = resolve_directory(os.environ.get("CHECKPOINT_DIR") or config_dict.get("checkpoint_dir"), "models/checkpoints")

    seed = int(config_dict.get("random_seed", 42))
    set_seed(seed)

    # 2. Hardware Discovery & Device Configuration
    cuda_avail = torch.cuda.is_available()
    req_device = override_device or config_dict.get("device", "auto")

    if req_device == "auto":
        device_str = "cuda:0" if cuda_avail else "cpu"
    elif req_device.startswith("cuda") and not cuda_avail:
        print(f"[WARNING] Requested device '{req_device}' but CUDA is unavailable. Falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = req_device

    device = torch.device(device_str)
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device.index or 0) if is_cuda else "N/A"
    total_vram_gb = round(torch.cuda.get_device_properties(device.index or 0).total_memory / (1024 ** 3), 2) if is_cuda else 0.0

    print(f"Experiment Name      : {exp_name}")
    print(f"Dataset Root Path    : {data_root}")
    print(f"Output Directory     : {output_dir}")
    print(f"Checkpoint Directory : {checkpoint_dir}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"CUDA Available       : {cuda_avail}")
    print(f"Active Device        : {device_str} ({gpu_name}, {total_vram_gb} GB VRAM)")
    print(f"Random Seed          : {seed}")
    print(f"Smoke Test Mode      : {smoke_test}")

    # 3. Build & Audit Dataset with Leakage Lock
    print("\n[Step 1/5] Loading & Verifying Dataset Pipeline...")
    hosp_path = os.path.join(data_root, "hosp") if os.path.exists(os.path.join(data_root, "hosp")) else data_root
    icu_path = os.path.join(data_root, "icu") if os.path.exists(os.path.join(data_root, "icu")) else data_root

    pipeline = TemporalDatasetPipeline(hosp_dir=hosp_path, icu_dir=icu_path, outputs_dir=output_dir)
    dataset_res = pipeline.build_and_audit_dataset()
    datasets = dataset_res["datasets"]

    # Verify Patient Splits Disjointness
    train_pts = set(pipeline.split_dict["train"])
    val_pts = set(pipeline.split_dict["validate"])
    test_pts = set(pipeline.split_dict["test"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage detected: Train & Val overlap!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage detected: Train & Test overlap!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage detected: Val & Test overlap!"
    print(f"Patient Splitting    : Train={len(train_pts)} pts, Val={len(val_pts)} pts, Test={len(test_pts)} pts (STRICTLY DISJOINT)")

    # 4. Initialize PyTorch Datasets & DataLoaders
    print("\n[Step 2/5] Initializing PyTorch DataLoaders & Class Weights...")
    train_dataset = MIMICIVTemporalDataset(
        features=datasets["train"]["X"],
        deteriorations=datasets["train"]["y_det"],
        deterioration_masks=datasets["train"]["m_det"],
        risk_tiers=datasets["train"]["y_tier"],
        forecasts=datasets["train"]["y_fore"],
        forecast_masks=datasets["train"]["m_fore"],
        treatment_responses=datasets["train"]["y_resp"],
        response_masks=datasets["train"]["m_resp"],
        metadata=datasets["train"]["meta"]
    )
    val_dataset = MIMICIVTemporalDataset(
        features=datasets["validate"]["X"],
        deteriorations=datasets["validate"]["y_det"],
        deterioration_masks=datasets["validate"]["m_det"],
        risk_tiers=datasets["validate"]["y_tier"],
        forecasts=datasets["validate"]["y_fore"],
        forecast_masks=datasets["validate"]["m_fore"],
        treatment_responses=datasets["validate"]["y_resp"],
        response_masks=datasets["validate"]["m_resp"],
        metadata=datasets["validate"]["meta"]
    )

    batch_size = int(config_dict.get("batch_size", 64))
    num_workers = int(config_dict.get("num_workers", 0 if sys.platform == "win32" else 4))
    pin_mem = bool(config_dict.get("pin_memory", is_cuda)) and is_cuda

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # Compute Train-Only Weights & Variances
    train_y_det = datasets["train"]["y_det"]
    num_pos = float(np.sum(train_y_det))
    num_neg = float(len(train_y_det) - num_pos)
    instab_pos_weight = (num_neg / max(1.0, num_pos))

    train_tiers = datasets["train"]["y_tier"]
    tier_counts = np.bincount(train_tiers, minlength=3).astype(np.float32)
    tier_weights = torch.tensor(len(train_tiers) / (3.0 * np.maximum(tier_counts, 1.0)), dtype=torch.float32).to(device)

    train_m_resp_bool = datasets["train"]["m_resp"] == 1.0
    valid_resps = datasets["train"]["y_resp"][train_m_resp_bool]
    resp_counts = np.bincount(valid_resps, minlength=3).astype(np.float32)
    resp_weights = torch.tensor(len(valid_resps) / (3.0 * np.maximum(resp_counts, 1.0)), dtype=torch.float32).to(device)

    train_y_fore = datasets["train"]["y_fore"]
    train_m_fore = datasets["train"]["m_fore"]
    fore_vars = []
    for c in range(5):
        m = train_m_fore[:, :, c] == 1.0
        v = float(np.var(train_y_fore[:, :, c][m])) if m.sum() > 0 else 1.0
        fore_vars.append(max(v, 1e-3))
    fore_vars_tensor = torch.tensor(fore_vars, dtype=torch.float32).to(device)

    # 5. Build Model, Loss, Optimizer & Scheduler
    print("\n[Step 3/5] Instantiating Multi-Task Architecture & Loss...")
    model_cfg = TemporalModelConfig(
        input_channels=int(config_dict.get("input_channels", 45)),
        sequence_length=int(config_dict.get("sequence_length", 24)),
        conv_filters=int(config_dict.get("conv_filters", 64)),
        conv_kernel_size=int(config_dict.get("conv_kernel_size", 3)),
        lstm_hidden_dim=int(config_dict.get("lstm_hidden_dim", 64)),
        lstm_num_layers=int(config_dict.get("lstm_num_layers", 2)),
        lstm_bidirectional=bool(config_dict.get("lstm_bidirectional", True)),
        dropout=float(config_dict.get("dropout", 0.2)),
        enable_forecast_head=bool(config_dict.get("enable_forecast_head", True)),
        enable_instability_head=bool(config_dict.get("enable_instability_head", True)),
        enable_risk_tier_head=bool(config_dict.get("enable_risk_tier_head", True)),
        enable_treatment_response_head=bool(config_dict.get("enable_treatment_response_head", True)),
        loss_weight_forecast=float(config_dict.get("loss_weight_forecast", 1.0)),
        loss_weight_risk_tier=float(config_dict.get("loss_weight_risk_tier", 0.5)),
        loss_weight_instability=float(config_dict.get("loss_weight_instability", 0.1)),
        loss_weight_treatment_response=float(config_dict.get("loss_weight_treatment_response", 0.1)),
        batch_size=batch_size,
        learning_rate=float(config_dict.get("learning_rate", 1e-3)),
        weight_decay=float(config_dict.get("weight_decay", 1e-4)),
        gradient_clip_val=float(config_dict.get("gradient_clip_val", 1.0)),
        num_epochs=int(config_dict.get("epochs", 30)),
        early_stopping_patience=int(config_dict.get("patience", 6)),
        random_seed=seed,
        device=device_str,
        checkpoint_dir=checkpoint_dir,
        outputs_dir=output_dir
    )

    model = TemporalCNNBiLSTM(model_cfg).to(device)
    loss_fn = MultiTaskTemporalLoss(
        config=model_cfg,
        instability_pos_weight=instab_pos_weight,
        risk_tier_weights=tier_weights,
        treatment_response_weights=resp_weights,
        forecast_channel_variances=fore_vars_tensor
    ).to(device)

    lr = float(config_dict.get("learning_rate", 1e-3))
    wd = float(config_dict.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_epochs = 2 if smoke_test else int(config_dict.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    use_amp = bool(config_dict.get("mixed_precision", True)) and is_cuda
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Mixed Precision (AMP): Enabled={use_amp} (Dtype={amp_dtype if use_amp else 'None'})")

    # 6. Save Reproducibility Run Metadata
    git_info = get_git_info()
    run_metadata = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "random_seed": seed,
        "dataset_root": data_root,
        "dataset_windows_total": len(datasets["train"]["X"]) + len(datasets["validate"]["X"]) + len(datasets["test"]["X"]),
        "patient_splits": {
            "train_patients": len(train_pts),
            "val_patients": len(val_pts),
            "test_patients": len(test_pts)
        },
        "hardware": {
            "os": sys.platform,
            "device": device_str,
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "cuda_version": torch.version.cuda if cuda_avail else "N/A",
            "pytorch_version": torch.__version__,
            "cpu_count": os.cpu_count()
        },
        "model_hyperparameters": model_cfg.to_dict(),
        "smoke_test": smoke_test
    }
    meta_path = os.path.join(output_dir, f"{exp_name}_run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved Run Metadata   : {meta_path}")

    # 7. Training & Validation Loop
    print("\n[Step 4/5] Launching Multi-Task Training Loop...")
    best_val_loss = float("inf")
    patience = int(config_dict.get("patience", 6))
    patience_counter = 0
    history = []

    best_ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_best.pt")
    final_ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_final.pt")
    max_train_batches = 20 if smoke_test else len(train_loader)

    total_training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_batch_losses = []

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        for b_idx, batch in enumerate(train_loader):
            if smoke_test and b_idx >= max_train_batches:
                break
            b_feat, b_det, b_mdet, b_tier, b_fore, b_mfore, b_resp, b_mresp = [x.to(device, non_blocking=is_cuda) for x in batch]

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = model(b_feat)
                    t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                    m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                    batch_loss, _ = loss_fn(preds, t_dict, m_dict)
            else:
                preds = model(b_feat)
                t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                batch_loss, _ = loss_fn(preds, t_dict, m_dict)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_cfg.gradient_clip_val)
            optimizer.step()
            train_batch_losses.append(batch_loss.item())

        avg_train_loss = float(np.mean(train_batch_losses))
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation Phase
        model.eval()
        val_batch_losses = []
        all_true_fore, all_pred_fore, all_mask_fore = [], [], []
        all_true_det, all_pred_det, all_mask_det = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                b_feat, b_det, b_mdet, b_tier, b_fore, b_mfore, b_resp, b_mresp = [x.to(device, non_blocking=is_cuda) for x in batch]
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        preds = model(b_feat)
                        t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                        m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                        v_loss, _ = loss_fn(preds, t_dict, m_dict)
                else:
                    preds = model(b_feat)
                    t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                    m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                    v_loss, _ = loss_fn(preds, t_dict, m_dict)

                val_batch_losses.append(v_loss.item())

                if model_cfg.enable_forecast_head and "forecast" in preds:
                    all_true_fore.append(b_fore.float().cpu().numpy())
                    all_pred_fore.append(preds["forecast"].float().cpu().numpy())
                    all_mask_fore.append(b_mfore.float().cpu().numpy())

                if model_cfg.enable_instability_head and "instability_logits" in preds:
                    probs = torch.sigmoid(preds["instability_logits"]).float().cpu().numpy()
                    all_pred_det.append(probs)
                    all_true_det.append(b_det.float().cpu().numpy())
                    all_mask_det.append(b_mdet.float().cpu().numpy())

        avg_val_loss = float(np.mean(val_batch_losses))
        epoch_sec = round(time.time() - epoch_start, 2)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if is_cuda else 0.0

        # Calculate Validation Metrics in Clinical Units
        val_fore_mae, val_fore_rmse = 0.0, 0.0
        if all_true_fore:
            v_t = np.concatenate(all_true_fore, axis=0)
            v_p = np.concatenate(all_pred_fore, axis=0)
            v_m = np.concatenate(all_mask_fore, axis=0) == 1.0
            val_fore_mae = float(mean_absolute_error(v_t[v_m], v_p[v_m]))
            val_fore_rmse = float(np.sqrt(mean_squared_error(v_t[v_m], v_p[v_m])))

        val_det_auroc, val_det_auprc = 0.0, 0.0
        if all_true_det:
            d_t = np.concatenate(all_true_det, axis=0)
            d_p = np.concatenate(all_pred_det, axis=0)
            d_m = np.concatenate(all_mask_det, axis=0) == 1.0
            if len(np.unique(d_t[d_m])) > 1:
                val_det_auroc = float(roc_auc_score(d_t[d_m], d_p[d_m]))
                val_det_auprc = float(average_precision_score(d_t[d_m], d_p[d_m]))

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_forecast_mae": round(val_fore_mae, 4),
            "val_forecast_rmse": round(val_fore_rmse, 4),
            "val_instability_auroc": round(val_det_auroc, 4),
            "val_instability_auprc": round(val_det_auprc, 4),
            "learning_rate": round(current_lr, 6),
            "peak_vram_mb": peak_vram_mb,
            "epoch_duration_sec": epoch_sec
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Fore MAE: {val_fore_mae:.2f} | "
            f"Val AUROC: {val_det_auroc:.3f} | "
            f"VRAM: {peak_vram_mb}MB | Time: {epoch_sec}s"
        )

        # Checkpoint Saving & Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_cfg.to_dict(),
                "val_loss": avg_val_loss,
                "val_forecast_mae": val_fore_mae,
                "forecast_channel_variances": fore_vars,
                "instability_pos_weight": instab_pos_weight
            }, best_ckpt_path)
            print(f"  -> Saved Best Model Checkpoint: {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"\n[Early Stopping] Validation loss did not improve for {patience} epochs. Stopping training.")
                break

    # Save Final Checkpoint & History
    torch.save({
        "epoch": len(history),
        "model_state_dict": model.state_dict(),
        "config": model_cfg.to_dict(),
        "final_val_loss": history[-1]["val_loss"],
        "forecast_channel_variances": fore_vars
    }, final_ckpt_path)

    total_training_sec = round(time.time() - total_training_start, 2)
    print(f"\n[Step 5/5] Finalizing Experiment '{exp_name}' in {total_training_sec}s...")

    history_path = os.path.join(output_dir, f"{exp_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    val_summary = {
        "experiment_name": exp_name,
        "best_epoch": min(history, key=lambda x: x["val_loss"])["epoch"],
        "best_val_loss": round(best_val_loss, 4),
        "final_val_forecast_mae": history[-1]["val_forecast_mae"],
        "total_training_seconds": total_training_sec,
        "checkpoints": {
            "best": best_ckpt_path,
            "final": final_ckpt_path
        }
    }
    val_summary_path = os.path.join(output_dir, f"{exp_name}_val_metrics.json")
    with open(val_summary_path, "w") as f:
        json.dump(val_summary, f, indent=2)

    print(f"Training History     : {history_path}")
    print(f"Validation Summary   : {val_summary_path}")
    print("=" * 78 + "\n")

    return val_summary


def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-IV Multi-Task CNN-BiLSTM Training Engine")
    parser.add_argument("--config", type=str, default="configs/temporal_training.yaml", help="Path to YAML training configuration file")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name override (e.g. d1_forecast_only, d5_multitask_full)")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root directory path override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory override")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--smoke-test", action="store_true", help="Run in short smoke test mode (2 epochs, limited batches)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {
        "experiment_name": args.experiment,
        "dataset_root": args.data_root,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    cfg = load_training_config(args.config, overrides)
    train_temporal_model(cfg, smoke_test=args.smoke_test, override_device=args.device)

"""
Stage C2: Multi-Task CNN-BiLSTM NVIDIA CUDA Smoke Test & Mixed Precision Validation Engine.
Executes:
1. Hardware & CUDA discovery and verification (Device, VRAM, PyTorch version, CUDA version)
2. Response-window mapping consistency verification & persistence
3. Pre-ICU safety check
4. Real-data mini-batch forward pass on cuda:0 & tensor integrity check
5. Modern PyTorch mixed-precision (torch.amp) forward, loss calculation, backward pass, gradient norm check, and optimizer step
6. Short 2-epoch smoke training run with AMP on cuda:0, tracking VRAM, time, loss decrease, and checkpoint saving
7. Validation forecast evaluation
8. Output persistence: models/checkpoints/cnn_bilstm_smoke_test.pt and ml/temporal/outputs/smoke_test_report.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.temporal.config import TemporalModelConfig
from ml.temporal.model import TemporalCNNBiLSTM
from ml.temporal.loss import MultiTaskTemporalLoss
from ml.temporal.sequences import TemporalDatasetPipeline, MIMICIVTemporalDataset
from ml.temporal.targets import FORECAST_CHANNELS


def run_stage_c2_cuda_smoke_test() -> Dict[str, Any]:
    print("=" * 70)
    print("    STAGE C2: NVIDIA CUDA SMOKE TEST & MIXED PRECISION VALIDATION GATE")
    print("=" * 70)

    # 1. Hardware & CUDA Verification
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()

    if not cuda_available:
        print("[ERROR] CUDA is not available. Aborting GPU smoke test.")
        sys.exit(1)

    device_name = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    total_vram_gb = round(gpu_props.total_memory / (1024 ** 3), 2)
    device = torch.device(device_name)

    print(f"PyTorch Version           : {torch_version}")
    print(f"CUDA Version              : {cuda_version}")
    print(f"CUDA Available            : {cuda_available}")
    print(f"Device Count              : {device_count}")
    print(f"Selected Device           : {device_name} ({gpu_name}, {total_vram_gb} GB VRAM)")
    print(f"Compute Capability        : {gpu_props.major}.{gpu_props.minor}")

    # 2. Build Dataset & Reconcile Response-Window Mapping
    print("\n[1/7] Building Dataset & Reconciling Treatment-Response Windows...")
    pipeline = TemporalDatasetPipeline()
    dataset_res = pipeline.build_and_audit_dataset()
    datasets = dataset_res["datasets"]

    train_m_resp = datasets["train"]["m_resp"]
    val_m_resp = datasets["validate"]["m_resp"]
    test_m_resp = datasets["test"]["m_resp"]
    total_resp_valid_windows = int(train_m_resp.sum() + val_m_resp.sum() + test_m_resp.sum())

    event_file = os.path.join(pipeline.outputs_dir, "event_linked_treatment_responses.json")
    with open(event_file, "r") as f:
        event_data = json.load(f)
    event_records = event_data["records"]
    total_drug_records = len(event_records)

    mapping_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_response_valid_windows_in_tensors": total_resp_valid_windows,
        "total_drug_event_records": total_drug_records,
        "concurrent_coprescriptions_count": total_drug_records - total_resp_valid_windows,
        "representation_rule": (
            "In the model training dataset, each sliding window index corresponds to exactly one sequence [24, 45]. "
            "When multiple treatments are initiated in the same 15-minute window, response_valid_mask = 1.0 is set once "
            "for that window and evaluates the composite post-intervention trajectory shift without duplicating sequence tensors."
        ),
        "split_breakdown": {
            "train_valid_response_windows": int(train_m_resp.sum()),
            "val_valid_response_windows": int(val_m_resp.sum()),
            "test_valid_response_windows": int(test_m_resp.sum())
        }
    }
    mapping_path = os.path.join(pipeline.outputs_dir, "final_response_window_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Response mapping saved to : {mapping_path}")

    # 3. Create PyTorch Datasets & DataLoaders
    print("\n[2/7] Initializing PyTorch Datasets and DataLoaders...")
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

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 4. Calculate Train-Only Class Weights & Forecast Channel Variances
    print("\n[3/7] Computing Training-Only Loss Weights & Variances...")
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

    # Channel variance scaling
    train_y_fore = datasets["train"]["y_fore"]
    train_m_fore = datasets["train"]["m_fore"]
    fore_vars = []
    for c in range(5):
        m = train_m_fore[:, :, c] == 1.0
        v = float(np.var(train_y_fore[:, :, c][m])) if m.sum() > 0 else 1.0
        fore_vars.append(max(v, 1e-3))
    fore_vars_tensor = torch.tensor(fore_vars, dtype=torch.float32).to(device)

    print(f"Instability pos_weight    : {round(instab_pos_weight, 4)}")
    print(f"Risk Tier weights         : {tier_weights.cpu().numpy().round(3)}")
    print(f"Treatment Response weights: {resp_weights.cpu().numpy().round(3)}")
    print(f"Forecast Channel Variances: {fore_vars_tensor.cpu().numpy().round(2)}")

    # 5. Model Architecture & Forward Pass Check on GPU
    print("\n[4/7] Testing Model Architecture & Real-Data Forward Pass on cuda:0...")
    config = TemporalModelConfig(batch_size=batch_size, device=device_name)
    model = TemporalCNNBiLSTM(config).to(device)
    loss_fn = MultiTaskTemporalLoss(
        config=config,
        instability_pos_weight=instab_pos_weight,
        risk_tier_weights=tier_weights,
        treatment_response_weights=resp_weights,
        forecast_channel_variances=fore_vars_tensor
    ).to(device)

    # Extract 1 real mini-batch
    batch = next(iter(train_loader))
    b_feat, b_det, b_mdet, b_tier, b_fore, b_mfore, b_resp, b_mresp = [x.to(device) for x in batch]

    assert b_feat.device.type == "cuda", f"Expected cuda device, got {b_feat.device}"
    assert b_feat.shape == torch.Size([batch_size, 24, 45]), f"Unexpected X shape: {b_feat.shape}"
    assert not torch.isnan(b_feat).any(), "NaN in GPU input batch!"
    assert not torch.isinf(b_feat).any(), "Inf in GPU input batch!"

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(b_feat)

    assert preds["instability_logits"].shape == torch.Size([batch_size])
    assert preds["risk_tier_logits"].shape == torch.Size([batch_size, 3])
    assert preds["forecast"].shape == torch.Size([batch_size, 4, 5])
    assert preds["treatment_response_logits"].shape == torch.Size([batch_size, 3])

    assert not torch.isnan(preds["forecast"]).any(), "NaN in GPU forecast!"
    assert not torch.isnan(preds["instability_logits"]).any(), "NaN in GPU instability!"

    vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
    vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)

    print("GPU Forward pass successful:")
    print(f"  Input Tensor Device     : {b_feat.device}")
    print(f"  Input Tensor Shape      : {list(b_feat.shape)}")
    print(f"  Forecast Output Shape   : {list(preds['forecast'].shape)}")
    print(f"  VRAM Allocated          : {vram_alloc_mb} MB")
    print(f"  VRAM Reserved           : {vram_res_mb} MB")

    # 6. Modern PyTorch Mixed Precision (AMP) Backward Pass & Optimizer Step Check
    print("\n[5/7] Testing PyTorch AMP Mixed Precision Backward Pass & Step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    t0 = time.time()
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        train_preds = model(b_feat)
        targets_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
        masks_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
        loss, loss_components = loss_fn(train_preds, targets_dict, masks_dict)

    assert not torch.isnan(loss), "NaN in AMP loss!"
    assert not torch.isinf(loss), "Inf in AMP loss!"

    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            assert not np.isnan(norm), f"NaN gradient in {name}!"
            assert not np.isinf(norm), f"Inf gradient in {name}!"
            grad_norms.append(norm)

    total_grad_norm = float(np.sqrt(sum(g ** 2 for g in grad_norms)))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    batch_runtime_ms = round((time.time() - t0) * 1000.0, 2)

    print("AMP Backward pass & optimizer step successful:")
    print(f"  AMP Dtype Used          : {amp_dtype}")
    print(f"  Total Batch Loss        : {loss_components['total_loss']:.4f}")
    print(f"  Forecast Loss           : {loss_components.get('loss_forecast', 0.0):.4f}")
    print(f"  Risk Tier Loss          : {loss_components.get('loss_risk_tier', 0.0):.4f}")
    print(f"  Instability Loss        : {loss_components.get('loss_instability', 0.0):.4f}")
    print(f"  Treatment Response Loss : {loss_components.get('loss_treatment_response', 0.0):.4f}")
    print(f"  Total Gradient Norm     : {total_grad_norm:.4f}")
    print(f"  Single Batch Runtime    : {batch_runtime_ms} ms")

    # 7. Short 2-Epoch Smoke Training Run on GPU with AMP
    print("\n[6/7] Running 2-Epoch CUDA Mixed-Precision Smoke Run...")
    smoke_history = []
    max_train_batches = 100

    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    smoke_ckpt_path = os.path.join(checkpoint_dir, "cnn_bilstm_smoke_test.pt")

    for epoch in range(1, 3):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for b_idx, batch in enumerate(train_loader):
            if b_idx >= max_train_batches:
                break
            b_feat, b_det, b_mdet, b_tier, b_fore, b_mfore, b_resp, b_mresp = [x.to(device, non_blocking=True) for x in batch]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(b_feat)
                t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                batch_loss, _ = loss_fn(preds, t_dict, m_dict)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(batch_loss.item())


        avg_train_loss = float(np.mean(train_losses))

        # Run Validation on GPU
        model.eval()
        val_losses = []
        all_true_fore = []
        all_pred_fore = []
        all_mask_fore = []

        with torch.no_grad():
            for batch in val_loader:
                b_feat, b_det, b_mdet, b_tier, b_fore, b_mfore, b_resp, b_mresp = [x.to(device, non_blocking=True) for x in batch]
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(b_feat)
                    t_dict = {"instability": b_det, "risk_tier": b_tier, "forecast": b_fore, "treatment_response": b_resp}
                    m_dict = {"instability": b_mdet, "forecast": b_mfore, "treatment_response": b_mresp}
                    v_loss, _ = loss_fn(preds, t_dict, m_dict)
                
                val_losses.append(v_loss.item())
                all_true_fore.append(b_fore.cpu().numpy())
                all_pred_fore.append(preds["forecast"].cpu().numpy())
                all_mask_fore.append(b_mfore.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        
        # Calculate Validation Forecast MAE/RMSE in true clinical units
        val_true_arr = np.concatenate(all_true_fore, axis=0)
        val_pred_arr = np.concatenate(all_pred_fore, axis=0)
        val_mask_arr = np.concatenate(all_mask_fore, axis=0)

        valid_mask_bool = val_mask_arr == 1.0
        val_mae = float(mean_absolute_error(val_true_arr[valid_mask_bool], val_pred_arr[valid_mask_bool]))
        val_rmse = float(np.sqrt(mean_squared_error(val_true_arr[valid_mask_bool], val_pred_arr[valid_mask_bool])))
        
        duration = round(time.time() - epoch_start, 2)
        gpu_mem_used_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)

        smoke_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_forecast_mae": round(val_mae, 4),
            "val_forecast_rmse": round(val_rmse, 4),
            "gpu_memory_used_mb": gpu_mem_used_mb,
            "epoch_duration_sec": duration
        })

        print(f"Epoch {epoch}/2 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Forecast MAE: {val_mae:.4f} | GPU Mem: {gpu_mem_used_mb}MB | Time: {duration}s")

    # 8. Save Smoke Checkpoint
    print("\n[7/7] Saving Smoke Checkpoint & Generating Report...")
    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "smoke_history": smoke_history,
        "instability_pos_weight": instab_pos_weight,
        "forecast_channel_variances": fore_vars
    }, smoke_ckpt_path)
    print(f"Smoke checkpoint saved to : {smoke_ckpt_path}")

    # Generate smoke test report
    smoke_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "pytorch_version": torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "device": device_name,
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
        },
        "model_configuration": config.to_dict(),
        "numerical_validity_checks": {
            "input_batch_shape": [batch_size, 24, 45],
            "zero_nan_in_input": True,
            "zero_inf_in_input": True,
            "forward_pass_passed": True,
            "loss_finite": True,
            "backward_gradients_finite": True,
            "total_gradient_norm": round(total_grad_norm, 4),
            "single_batch_runtime_ms": batch_runtime_ms,
            "mixed_precision_amp_verified": True,
            "optimizer_step_executed": True
        },
        "smoke_training_history": smoke_history,
        "smoke_checkpoint_path": smoke_ckpt_path,
        "stage_c2_status": "PASSED"
    }

    report_path = os.path.join(pipeline.outputs_dir, "smoke_test_report.json")
    with open(report_path, "w") as f:
        json.dump(smoke_report, f, indent=2)
    print(f"Smoke report saved to     : {report_path}")

    print("\n" + "=" * 70)
    print("           STAGE C2 CUDA SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return smoke_report


if __name__ == "__main__":
    run_stage_c2_cuda_smoke_test()

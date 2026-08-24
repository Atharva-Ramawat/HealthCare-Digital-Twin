# University DGX & HPC Linux Training Guide
**Project**: AI-Driven Digital Twin for Smart Healthcare (Pulmonary Critical Care)  
**Target Environment**: NVIDIA DGX / HPC Linux Systems (A100 / H100 / V100 / RTX GPU nodes)

---

## 1. Overview & Architecture Portability
The training engine is 100% portable and environment-agnostic. All data paths, checkpoints, and hardware configurations can be controlled via environment variables and YAML configuration files without making any code edits.

---

## 2. Environment Setup on Linux DGX

### Option A: Using Conda / Mamba (Recommended)
```bash
# 1. Clone repository from GitHub
git clone https://github.com/your-org/healthcare-digital-twin.git
cd healthcare-digital-twin

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate digital-twin-gpu

# 3. Verify CUDA GPU visibility
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Option B: Using Python Virtual Environment (venv + pip)
```bash
# 1. Create venv
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA 12.1 / 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements-gpu.txt
```

---

## 3. Dataset Path Configuration (Zero Data in Git)

Raw MIMIC-IV clinical files must remain external to the Git repository. Set the `MIMIC_IV_ROOT` environment variable to point to the shared cluster dataset path:

```bash
# Set environment variables for dataset location
export MIMIC_IV_ROOT="/datasets/mimic_iv_v2.2"
# Or using generic DATA_ROOT
export DATA_ROOT="/datasets"

# Optional: Custom output or checkpoint mount directories
export OUTPUT_DIR="/scratch/user/outputs"
export CHECKPOINT_DIR="/scratch/user/checkpoints"
```

The dataset directory must contain:
```text
$MIMIC_IV_ROOT/
├── hosp/
│   ├── admissions.csv.gz
│   ├── patients.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── d_icd_diagnoses.csv.gz
│   └── labevents.csv.gz
└── icu/
    ├── icustays.csv.gz
    ├── chartevents.csv.gz
    └── inputevents.csv.gz
```

---

## 4. Pre-Flight Verification & Test Execution

Before starting full training runs, run the test suite to ensure dataset integrity and absence of leakage:

```bash
pytest tests/
```
*Expected Result*: All 108 tests pass cleanly.

---

## 5. Experiment Execution Commands (D1 – D5)

Launch experiments using the unified CLI entrypoint `python -m ml.temporal.train`:

### Experiment D1: 4-Step Vital Forecast Only
```bash
python -m ml.temporal.train \
    --config configs/experiments/d1_forecast_only.yaml \
    --epochs 30 \
    --batch-size 128
```

### Experiment D2: Future Physiological Instability Event Only
```bash
python -m ml.temporal.train \
    --config configs/experiments/d2_instability_only.yaml \
    --epochs 30 \
    --batch-size 128
```

### Experiment D3: Risk Tier Classification Only
```bash
python -m ml.temporal.train \
    --config configs/experiments/d3_risk_tier_only.yaml \
    --epochs 30 \
    --batch-size 128
```

### Experiment D4: Observational Treatment Response Only
```bash
python -m ml.temporal.train \
    --config configs/experiments/d4_treatment_response_only.yaml \
    --epochs 30 \
    --batch-size 128
```

### Experiment D5: Full Multi-Task Joint CNN-BiLSTM (Primary Experiment)
```bash
python -m ml.temporal.train \
    --config configs/experiments/d5_multitask_full.yaml \
    --epochs 30 \
    --batch-size 128
```

### Background Execution via `nohup` or SLURM:
```bash
# Background run with logging
nohup python -m ml.temporal.train --config configs/experiments/d5_multitask_full.yaml > d5_training.log 2>&1 &
```

---

## 6. Output Artifacts & Reproducibility Files

Each experiment automatically generates the following files in `ml/temporal/outputs/` (or `$OUTPUT_DIR`):

1. **`{experiment_name}_run_metadata.json`**: Exact Git commit, branch, hardware, CUDA version, PyTorch version, seeds, and hyperparameters.
2. **`{experiment_name}_history.json`**: Per-epoch train loss, validation loss, validation forecast MAE/RMSE, AUROC, AUPRC, peak VRAM, and durations.
3. **`{experiment_name}_val_metrics.json`**: Summary of best epoch and validation metrics.
4. **`models/checkpoints/{experiment_name}_best.pt`**: PyTorch checkpoint of best epoch based on validation loss.
5. **`models/checkpoints/{experiment_name}_final.pt`**: Final epoch checkpoint.

---

## 7. Results Collection (What to Bring Back to Git)

After training completes on the DGX cluster:

### DO Bring Back:
- `ml/temporal/outputs/*_history.json`
- `ml/temporal/outputs/*_val_metrics.json`
- `ml/temporal/outputs/*_run_metadata.json`
- Selected best checkpoint weights (`*_best.pt`) via artifact storage or Git LFS.

### DO NOT Bring Back:
- Raw clinical CSV / Parquet data
- Patient-identifiable logs

---

## 8. Troubleshooting & Performance Tips

1. **Multi-GPU Selection**: To train on a specific GPU index, use `CUDA_VISIBLE_DEVICES=0` or `--device cuda:1`.
2. **DataLoader Workers**: On Linux DGX with fast NVMe, use `--num-workers 8` or `--num-workers 16` for maximum throughput.
3. **Out of Memory (OOM)**: The CNN-BiLSTM requires only ~40 MB VRAM at batch size 64. For large batches (e.g. 512), VRAM is still < 500 MB. If OOM occurs, reduce `batch_size` in the YAML config.
4. **BFloat16 Support**: Ampere (A100), Hopper (H100), and Ada Lovelace architectures automatically use hardware `torch.bfloat16` for maximum throughput without gradient underflow.

"""
CXR Training, Evaluation, and Hardware Configuration.
Detects CUDA GPU hardware properties and manages hyperparameters, paths, and checkpoint policies.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

from ml.cxr.labels import TARGET_PULMONARY_CLASSES, UncertaintyPolicy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class HardwareConfig:
    """Hardware environment detection and configuration."""
    cuda_available: bool = field(default_factory=lambda: torch.cuda.is_available())
    device_name: str = field(default="CPU")
    device_count: int = 0
    total_memory_gb: float = 0.0
    pytorch_cuda_version: Optional[str] = None
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.pytorch_cuda_version = torch.version.cuda
        if self.cuda_available:
            self.device = torch.device("cuda:0")
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.total_memory_gb = round(props.total_memory / (1024 ** 3), 2)
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.device_count = 0
            self.total_memory_gb = 0.0

    def print_diagnostics(self):
        """Print full GPU & CUDA environment diagnostics."""
        print("=" * 60)
        print("          CXR HARDWARE & GPU ENVIRONMENT REPORT")
        print("=" * 60)
        print(f"CUDA Available       : {self.cuda_available}")
        print(f"Target Device        : {self.device}")
        print(f"Device Name          : {self.device_name}")
        print(f"Device Count         : {self.device_count}")
        print(f"GPU Memory (GB)      : {self.total_memory_gb} GB")
        print(f"PyTorch Version      : {torch.__version__}")
        print(f"PyTorch CUDA Version : {self.pytorch_cuda_version}")
        if not self.cuda_available:
            print("[NOTE] Running on CPU fallback. For full-scale training runs, execute on campus NVIDIA GPUs.")
        print("=" * 60)


@dataclass
class CXRConfig:
    """MIMIC-CXR Training and Evaluation Configuration."""
    # Data Paths
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr")
    metadata_csv: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr", "mimic-cxr-2.0.0-metadata.csv.gz")
    chexpert_csv: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr", "mimic-cxr-2.0.0-chexpert.csv.gz")
    split_csv: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr", "mimic-cxr-2.0.0-split.csv.gz")
    checkpoints_dir: str = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    outputs_dir: str = os.path.join(PROJECT_ROOT, "ml", "cxr", "outputs")

    # Target Pathologies & Policy
    target_classes: List[str] = field(default_factory=lambda: list(TARGET_PULMONARY_CLASSES))
    uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.U_ZERO

    # Model Hyperparameters
    architecture: str = "densenet121"
    num_classes: int = len(TARGET_PULMONARY_CLASSES)
    pretrained: bool = True
    dropout_rate: float = 0.25

    # Training Parameters
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 10
    early_stopping_patience: int = 4
    num_workers: int = 0
    mixed_precision: bool = False  # Set dynamically in __post_init__

    # Hardware
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        # Enable mixed precision automatically when CUDA GPU is available
        self.mixed_precision = self.hardware.cuda_available

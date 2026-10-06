"""
System & MLOps Telemetry Router.
Provides real-time hardware telemetry (GPU VRAM, CUDA device attributes, host RAM, CPU usage)
with graceful fallback to psutil CPU metrics when CUDA is unavailable.
"""

import os
import sys
from datetime import datetime, timezone
from fastapi import APIRouter, Request, status
import torch
import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.schemas.system_schema import SystemTelemetryResponse, MemoryBreakdown

router = APIRouter(prefix="/api/system", tags=["System & MLOps Telemetry"])


def _bytes_to_gb(b: int) -> float:
    """Convert bytes to gigabytes rounded to 1 decimal place."""
    return round(float(b) / (1024.0 ** 3), 1)


@router.get(
    "/telemetry",
    response_model=SystemTelemetryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get real-time GPU/CPU hardware and memory telemetry",
    description=(
        "Returns live dynamic compute metrics including torch.cuda device name, total VRAM, "
        "allocated VRAM, reserved VRAM, and psutil host memory metrics. "
        "Gracefully falls back to CPU & host RAM when CUDA is unavailable."
    ),
)
async def get_system_telemetry(request: Request) -> SystemTelemetryResponse:
    """Fetch live hardware telemetry from torch.cuda and psutil."""
    # 1. System RAM Metrics via psutil
    vm = psutil.virtual_memory()
    sys_total = vm.total
    sys_allocated = vm.used
    sys_reserved = vm.used
    sys_free = vm.available
    sys_percent = round(float(vm.percent), 1)

    system_memory = MemoryBreakdown(
        total_bytes=sys_total,
        allocated_bytes=sys_allocated,
        reserved_bytes=sys_reserved,
        free_bytes=sys_free,
        usage_percent=sys_percent,
        total_gb=_bytes_to_gb(sys_total),
        allocated_gb=_bytes_to_gb(sys_allocated),
        reserved_gb=_bytes_to_gb(sys_reserved),
        free_gb=_bytes_to_gb(sys_free),
    )

    # 2. Hardware / VRAM Telemetry (CUDA vs CPU fallback)
    cuda_is_avail = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_is_avail else None

    if cuda_is_avail and torch.cuda.device_count() > 0:
        device_name = torch.cuda.get_device_name(0)
        device_type = "cuda"
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory
        allocated_vram = torch.cuda.memory_allocated(0)
        reserved_vram = torch.cuda.memory_reserved(0)
        free_vram = max(0, total_vram - reserved_vram)
        vram_percent = round((float(reserved_vram) / float(total_vram) * 100.0), 1) if total_vram > 0 else 0.0

        gpu_memory = MemoryBreakdown(
            total_bytes=total_vram,
            allocated_bytes=allocated_vram,
            reserved_bytes=reserved_vram,
            free_bytes=free_vram,
            usage_percent=vram_percent,
            total_gb=_bytes_to_gb(total_vram),
            allocated_gb=_bytes_to_gb(allocated_vram),
            reserved_gb=_bytes_to_gb(reserved_vram),
            free_gb=_bytes_to_gb(free_vram),
        )
    else:
        # Graceful fallback to CPU and system RAM metrics
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1
        device_name = f"Host CPU ({physical_cores} Cores / {logical_cores} Threads)"
        device_type = "cpu"

        gpu_memory = MemoryBreakdown(
            total_bytes=sys_total,
            allocated_bytes=sys_allocated,
            reserved_bytes=sys_reserved,
            free_bytes=sys_free,
            usage_percent=sys_percent,
            total_gb=_bytes_to_gb(sys_total),
            allocated_gb=_bytes_to_gb(sys_allocated),
            reserved_gb=_bytes_to_gb(sys_reserved),
            free_gb=_bytes_to_gb(sys_free),
        )

    # 3. Model Residency and Host State
    app_model = getattr(request.app.state, "model", None)
    model_loaded = app_model is not None
    active_model = "densenet121_mimic_v2.pt" if model_loaded else "densenet121_mimic_v2.pt (Standby)"

    cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
    cpu_logical = psutil.cpu_count(logical=True) or 1
    cpu_physical = psutil.cpu_count(logical=False) or 1

    return SystemTelemetryResponse(
        device_name=device_name,
        device_type=device_type,
        cuda_available=cuda_is_avail,
        cuda_version=cuda_version,
        pytorch_version=torch.__version__,
        gpu_memory=gpu_memory,
        system_memory=system_memory,
        cpu_percent=cpu_percent,
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        active_model=active_model,
        model_loaded=model_loaded,
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

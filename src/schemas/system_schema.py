"""
Pydantic v2 schemas for real-time MLOps and System Telemetry endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class MemoryBreakdown(BaseModel):
    total_bytes: int = Field(..., description="Total memory in bytes")
    allocated_bytes: int = Field(..., description="Currently allocated memory in bytes")
    reserved_bytes: int = Field(..., description="Currently reserved/cached memory in bytes")
    free_bytes: int = Field(..., description="Free/available memory in bytes")
    usage_percent: float = Field(..., description="Memory utilization percentage (0-100%)")
    total_gb: float = Field(..., description="Total memory in Gigabytes rounded to 1 decimal place")
    allocated_gb: float = Field(..., description="Allocated memory in Gigabytes rounded to 1 decimal place")
    reserved_gb: float = Field(..., description="Reserved memory in Gigabytes rounded to 1 decimal place")
    free_gb: float = Field(..., description="Free memory in Gigabytes rounded to 1 decimal place")


class SystemTelemetryResponse(BaseModel):
    device_name: str = Field(..., description="Active compute device name, e.g. NVIDIA GeForce RTX 3050 6GB Laptop GPU")
    device_type: str = Field(..., description="Compute backend type ('cuda' or 'cpu')")
    cuda_available: bool = Field(..., description="Whether CUDA GPU acceleration is detected and active")
    cuda_version: Optional[str] = Field(None, description="CUDA driver/toolkit version")
    pytorch_version: str = Field(..., description="Installed PyTorch runtime version")
    gpu_memory: MemoryBreakdown = Field(..., description="GPU VRAM allocation and reservation metrics")
    system_memory: MemoryBreakdown = Field(..., description="Host system RAM metrics via psutil")
    cpu_percent: float = Field(..., description="Host CPU utilization percentage")
    cpu_count_logical: int = Field(..., description="Total logical CPU cores / threads")
    cpu_count_physical: int = Field(..., description="Physical CPU cores")
    active_model: str = Field("densenet121_mimic_v2.pt", description="Current loaded model checkpoint filename")
    model_loaded: bool = Field(..., description="Whether the vision model is actively resident in memory")
    status: str = Field("healthy", description="Operational health status of the ML inference subsystem")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of telemetry capture")

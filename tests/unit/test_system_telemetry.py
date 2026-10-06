"""
Unit tests for the System & MLOps Hardware Telemetry endpoint.
Tests real hardware retrieval (CUDA) and CPU fallback mode.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_get_system_telemetry_live():
    """Verify GET /api/system/telemetry returns 200 and matches the expected schema."""
    response = client.get("/api/system/telemetry")
    assert response.status_code == 200
    data = response.json()

    # Core identification fields
    assert "device_name" in data
    assert "device_type" in data
    assert data["device_type"] in ["cuda", "cpu"]
    assert "cuda_available" in data
    assert isinstance(data["cuda_available"], bool)
    assert "pytorch_version" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data

    # Memory fields
    assert "gpu_memory" in data
    gpu_mem = data["gpu_memory"]
    assert "total_bytes" in gpu_mem
    assert "allocated_bytes" in gpu_mem
    assert "reserved_bytes" in gpu_mem
    assert "free_bytes" in gpu_mem
    assert "usage_percent" in gpu_mem
    assert "total_gb" in gpu_mem
    assert "allocated_gb" in gpu_mem
    assert "reserved_gb" in gpu_mem
    assert "free_gb" in gpu_mem

    assert "system_memory" in data
    sys_mem = data["system_memory"]
    assert sys_mem["total_gb"] > 0
    assert sys_mem["total_bytes"] > 0

    # CPU metrics
    assert "cpu_percent" in data
    assert "cpu_count_logical" in data
    assert data["cpu_count_logical"] >= 1
    assert "cpu_count_physical" in data

    # Model status
    assert "active_model" in data
    assert "densenet121" in data["active_model"].lower()
    assert "model_loaded" in data


def test_get_system_telemetry_cpu_fallback():
    """Verify fallback to CPU and psutil when CUDA is unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        response = client.get("/api/system/telemetry")
        assert response.status_code == 200
        data = response.json()

        assert data["cuda_available"] is False
        assert data["device_type"] == "cpu"
        assert "Host CPU" in data["device_name"]
        assert data["cuda_version"] is None
        assert data["gpu_memory"]["total_gb"] > 0

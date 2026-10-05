"""
Unit tests for Chest Radiograph Modality & OOD Gatekeeper (src/ml/cxr/validator.py).
Verifies:
1. Genuine and synthetic CXRs are accepted.
2. Natural color photos are rejected due to chromatic saturation.
3. Documents and white scans are rejected due to background dominance.
4. Flat / blank images are rejected due to low contrast standard deviation.
5. High-frequency noise is rejected due to abnormal edge density.
6. Extreme aspect ratios are rejected.
"""

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.ml.cxr.validator import is_chest_xray, validate_chest_xray


def test_validator_accepts_synthetic_cxr():
    """Verify synthetic CXR phantom with thoracic contrast is accepted."""
    x = np.linspace(-1, 1, 224)
    y = np.linspace(-1, 1, 224)
    xx, yy = np.meshgrid(x, y)
    val = 80 + 110 * np.exp(-4 * xx**2) + 25 * np.cos(yy * 6)
    val = np.clip(val, 0, 255).astype(np.uint8)
    cxr = Image.fromarray(np.stack([val, val, val], axis=-1))

    is_valid, reason = validate_chest_xray(cxr)
    assert is_valid is True
    assert "Valid" in reason
    assert is_chest_xray(cxr) is True


def test_validator_rejects_color_photograph():
    """Verify natural color photographs are rejected for high chromatic saturation."""
    # Natural color photo (RGB with high saturation)
    photo = Image.new("RGB", (224, 224), color=(240, 110, 45))
    is_valid, reason = validate_chest_xray(photo)
    assert is_valid is False
    assert "saturation" in reason.lower()
    assert is_chest_xray(photo) is False


def test_validator_rejects_flat_uniform_image():
    """Verify solid or flat images are rejected for lack of dynamic range."""
    flat = Image.new("RGB", (224, 224), color=(128, 128, 128))
    is_valid, reason = validate_chest_xray(flat)
    assert is_valid is False
    assert "dynamic range" in reason.lower()
    assert is_chest_xray(flat) is False


def test_validator_rejects_document_scan():
    """Verify text documents or white sheet scans are rejected."""
    doc = Image.new("RGB", (224, 224), color=(255, 255, 255))
    d = ImageDraw.Draw(doc)
    d.text((20, 20), "Patient Medical Record Notes", fill=(0, 0, 0))

    is_valid, reason = validate_chest_xray(doc)
    assert is_valid is False
    assert is_chest_xray(doc) is False


def test_validator_rejects_extreme_aspect_ratio():
    """Verify non-thoracic aspect ratios are rejected."""
    banner = Image.new("L", (1000, 100), color=120)
    is_valid, reason = validate_chest_xray(banner)
    assert is_valid is False
    assert "aspect ratio" in reason.lower()


def test_validator_rejects_random_noise():
    """Verify high-frequency random noise is rejected."""
    noise_arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)

    is_valid, reason = validate_chest_xray(noise_img)
    assert is_valid is False
    assert is_chest_xray(noise_img) is False

"""
Out-of-Distribution (OOD) / Image Modality Gatekeeper for Chest Radiographs.

Provides lightweight, fast structural and statistical validation to determine
whether an uploaded image is a Chest X-Ray (CXR) prior to running DenseNet-121
inference. This prevents hallucinated pathology predictions and risk scores
on non-radiograph inputs (e.g., natural color photos, documents, noise, cartoons).
"""

import io
from typing import Tuple, Union
import numpy as np
from PIL import Image


def validate_chest_xray(image: Union[Image.Image, bytes, str]) -> Tuple[bool, str]:
    """
    Validates whether an input image conforms to the structural and statistical
    characteristics of a human chest radiograph (CXR).

    Args:
        image: A PIL Image, raw image bytes, or a filesystem path string.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    # 1. Resolve to PIL Image
    pil_img: Image.Image
    try:
        if isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            return False, "Unsupported image input type."
    except Exception as e:
        return False, f"Failed to decode image: {e}"

    # 2. Minimum resolution & aspect ratio
    w, h = pil_img.size
    if w < 32 or h < 32:
        return False, f"Image dimensions ({w}x{h}) are too small for clinical radiograph evaluation."

    aspect_ratio = w / float(h)
    if aspect_ratio < 0.45 or aspect_ratio > 2.2:
        return False, f"Aspect ratio ({aspect_ratio:.2f}) is outside plausible clinical thoracic bounds."

    # Convert to float32 numpy array
    arr = np.array(pil_img).astype(np.float32)

    # 3. Chromaticity / Color Saturation Check
    # Chest radiographs are fundamentally monochromatic (R ≈ G ≈ B).
    # Natural color photographs (faces, food, landscapes, charts) exhibit high inter-channel divergence.
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    chroma = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
    if chroma > 15.0:
        return False, f"Chromatic color saturation detected (chroma={chroma:.1f} > 15.0). Chest radiographs must be monochromatic."

    # 4. Dynamic Range & Contrast Check
    gray = np.mean(arr, axis=2)
    std_dev = float(np.std(gray))
    if std_dev < 15.0:
        return False, f"Image lacks sufficient dynamic range (std={std_dev:.1f} < 15.0). Flat or blank image detected."

    # 5. Document / Background Dominance Check
    white_fraction = float(np.mean(gray > 245.0))
    if white_fraction > 0.70:
        return False, f"Excessive white background ({white_fraction * 100:.1f}%). Document or text scan detected."

    black_fraction = float(np.mean(gray < 10.0))
    if black_fraction > 0.85:
        return False, f"Excessive black background ({black_fraction * 100:.1f}%). Blank or void image detected."

    # 6. Edge Gradient & Structural Texture Check
    # Natural radiographs display smooth anatomical tissue attenuation and defined skeletal borders.
    # High-frequency random noise or checkerboards exhibit extreme edge magnitudes.
    dy, dx = np.gradient(gray)
    edge_magnitude = float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))
    if edge_magnitude < 0.5:
        return False, f"Edge density too low (magnitude={edge_magnitude:.2f} < 0.5). Lacks anatomical structure."
    if edge_magnitude > 65.0:
        return False, f"Edge density abnormally high (magnitude={edge_magnitude:.2f} > 65.0). Non-medical noise or high-frequency pattern detected."

    return True, "Valid chest radiograph."


def is_chest_xray(image: Union[Image.Image, bytes, str]) -> bool:
    """
    Convenience predicate returning True if the image is a valid Chest Radiograph,
    False otherwise.
    """
    valid, _ = validate_chest_xray(image)
    return valid

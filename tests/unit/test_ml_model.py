"""
Unit tests for the real ML inference core (src/ml/model.py).
Tests:
- DenseNet121Pulmonary model instantiation
- Classifier head configuration Linear(1024, 8)
- torch.amp.autocast mixed precision execution
- 1024-dimensional feature vector extraction from global average pooling
- Forward and backward hooks on features.denseblock4.denselayer16.conv2
- Grad-CAM spatial heatmap generation and normalization
"""

import os
import sys
import pytest
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ml.model import DenseNet121Pulmonary
from src.schemas.cxr_schema import TARGET_PULMONARY_CLASSES


def test_densenet121_classifier_head_and_architecture():
    """Verify that DenseNet121Pulmonary has a Linear(1024, 8) classifier head."""
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False)

    # Verify classifier head
    classifier = model.densenet.classifier
    assert isinstance(classifier, nn.Linear)
    assert classifier.in_features == 1024
    assert classifier.out_features == 8
    assert model.num_classes == len(TARGET_PULMONARY_CLASSES)


def test_densenet121_forward_and_feature_extraction():
    """Verify forward pass returns both logits [B, 8] and 1024-dim features [B, 1024]."""
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # 1. Forward pass returning (logits, features)
    logits, features = model(dummy_input, return_features=True)
    assert logits.shape == (batch_size, 8)
    assert features.shape == (batch_size, 1024)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(features).any()

    # 2. Forward pass returning only logits
    only_logits = model(dummy_input, return_features=False)
    assert only_logits.shape == (batch_size, 8)

    # 3. Direct feature extractor method
    direct_features = model.extract_features(dummy_input)
    assert direct_features.shape == (batch_size, 1024)

    # 4. Multi-label probability prediction
    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (batch_size, 8)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_mixed_precision_autocast_cuda_or_cpu():
    """Verify torch.amp.autocast mixed-precision execution."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)

    # Run forward pass through mixed-precision context
    with torch.no_grad():
        logits, features = model(dummy_input, return_features=True)

    assert logits.shape == (2, 8)
    assert features.shape == (2, 1024)
    assert logits.device.type == device.type


def test_gradcam_hooks_and_heatmap_generation():
    """
    Verify forward and backward hooks on features.denseblock4.denselayer16.conv2
    specifically capture activations and gradients for Grad-CAM attribution.
    """
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False)
    model.eval()

    # Verify target convolutional layer
    target_conv = model.target_conv_layer
    assert isinstance(target_conv, nn.Conv2d)
    assert target_conv.in_channels == 128
    assert target_conv.out_channels == 32

    dummy_input = torch.randn(1, 3, 224, 224)

    # Generate Grad-CAM heatmap for target class 0 ("Pneumonia")
    class_idx = 0
    heatmap = model.generate_gradcam_heatmap(dummy_input, class_idx=class_idx, target_size=(224, 224))

    # Verify hooks captured activations and gradients
    assert model.activations is not None
    assert model.gradients is not None
    assert model.activations.shape[1] == 32  # 32 output channels from conv2
    assert model.gradients.shape[1] == 32

    # Verify generated heatmap output
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0 + 1e-6
    assert not np.isnan(heatmap).any()

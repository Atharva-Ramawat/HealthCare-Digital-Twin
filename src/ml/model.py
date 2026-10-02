"""
Machine Learning Core for MIMIC-CXR Pulmonary Image Analysis.
Provides DenseNet121Pulmonary wrapper with ImageNet pretraining, Linear(1024, 8) classifier head,
torch.amp.autocast mixed-precision execution, 1024-dimensional feature vector extraction,
and forward/backward hooks on features.denseblock4.denselayer16.conv2 for Grad-CAM explainability.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.schemas.cxr_schema import TARGET_PULMONARY_CLASSES


class DenseNet121Pulmonary(nn.Module):
    """
    PyTorch DenseNet-121 model customized for 8-class pulmonary multi-label classification.
    Features:
    - Pretrained ImageNet backbone initialization.
    - Modified classifier head: Linear(1024, 8).
    - torch.amp.autocast mixed-precision inference.
    - Global average pooling 1024-dim visual embedding extraction.
    - Grad-CAM hooks attached to features.denseblock4.denselayer16.conv2.
    """

    def __init__(
        self,
        num_classes: int = len(TARGET_PULMONARY_CLASSES),
        pretrained: bool = True,
        dropout_rate: float = 0.0
    ):
        super(DenseNet121Pulmonary, self).__init__()
        self.num_classes = num_classes
        self.target_classes = list(TARGET_PULMONARY_CLASSES)

        # 1. Initialize DenseNet-121 backbone with ImageNet weights
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.densenet = models.densenet121(weights=weights)
        except Exception:
            # Fallback for offline / disconnected sandbox environments
            self.densenet = models.densenet121(weights=None)

        # 2. Modify Classifier Head to Linear(1024, 8)
        num_in_features = self.densenet.classifier.in_features  # 1024
        if dropout_rate > 0.0:
            self.densenet.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_in_features, num_classes)
            )
        else:
            self.densenet.classifier = nn.Linear(num_in_features, num_classes)

        # 3. Hook placeholders for Grad-CAM feature attribution
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []

        # 4. Target Convolutional Layer for Grad-CAM
        self.target_conv_layer: nn.Module = self.densenet.features.denseblock4.denselayer16.conv2
        self.register_gradcam_hooks()

    def _forward_hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        """Save feature activations from target layer during forward pass."""
        self.activations = output.detach()

    def _backward_hook_fn(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
        """Save gradients from target layer during backward pass."""
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def register_gradcam_hooks(self):
        """Register forward and backward hooks on features.denseblock4.denselayer16.conv2."""
        self.remove_gradcam_hooks()
        h_fwd = self.target_conv_layer.register_forward_hook(self._forward_hook_fn)
        h_bwd = self.target_conv_layer.register_full_backward_hook(self._backward_hook_fn)
        self._hook_handles.extend([h_fwd, h_bwd])

    def remove_gradcam_hooks(self):
        """Remove registered Grad-CAM hooks to free resources."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 1024-dimensional feature vector from the global average pooling layer.
        Input x: [B, 3, H, W]
        Returns: [B, 1024]
        """
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=x.is_cuda):
            features = self.densenet.features(x)
            out = F.relu(features, inplace=False)
            pooled = F.adaptive_avg_pool2d(out, (1, 1))
            embedding = torch.flatten(pooled, 1)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Mixed-precision forward pass using torch.amp.autocast.
        Extracts and returns the raw logits [B, 8] and the 1024-dim latent feature vector [B, 1024].
        
        Args:
            x: Input image tensor of shape [B, 3, 224, 224]
            return_features: If True, returns (logits, embedding); else returns logits.
        
        Returns:
            (logits, embedding) if return_features is True, else logits
        """
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=x.is_cuda):
            features = self.densenet.features(x)
            out = F.relu(features, inplace=False)
            pooled = F.adaptive_avg_pool2d(out, (1, 1))
            embedding = torch.flatten(pooled, 1)  # [B, 1024]
            logits = self.densenet.classifier(embedding)  # [B, 8]

        if return_features:
            return logits, embedding
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-label Sigmoid probabilities across all 8 pulmonary target findings.
        Returns: [B, 8] in range [0.0, 1.0]
        """
        logits, _ = self.forward(x, return_features=True)
        return torch.sigmoid(logits)

    def generate_gradcam_heatmap(
        self,
        x: torch.Tensor,
        class_idx: int,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate a 2D Grad-CAM spatial activation heatmap for the specified target pathology class.
        
        Args:
            x: Single input image tensor [1, 3, H, W] or batch
            class_idx: Integer target class index (0 to 7)
            target_size: Output 2D (H, W) spatial resolution (defaults to input size)
            
        Returns:
            Normalized 2D numpy array heatmap in range [0.0, 1.0]
        """
        self.eval()
        x = x.clone().requires_grad_(True)
        h, w = (x.shape[2], x.shape[3]) if target_size is None else target_size

        # Forward pass (triggers forward hook on conv2)
        logits, _ = self.forward(x, return_features=True)
        score = logits[:, class_idx].sum()

        # Zero gradients and backward pass (triggers backward hook on conv2)
        self.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling on gradients: alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Linear combination of feature maps weighted by gradients
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [B, 1, H', W']
        cam = F.relu(cam)  # ReLU to highlight positive contributions

        # Bilinear interpolation upsampling to target spatial resolution
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalize to [0.0, 1.0]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

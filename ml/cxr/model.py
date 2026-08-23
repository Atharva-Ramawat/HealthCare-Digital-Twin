"""
DenseNet-121 Multi-Label Pulmonary Vision Architecture & Grad-CAM Visual Explainability.
Provides feature embedding extraction for multimodal Digital Twin fusion and localized pathology attention.
"""

import os
import io
import base64
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ml.cxr.labels import TARGET_PULMONARY_CLASSES


class DenseNet121Pulmonary(nn.Module):
    """
    DenseNet-121 Architecture customized for multi-label thoracic pulmonary disease classification.
    """

    def __init__(
        self,
        num_classes: int = len(TARGET_PULMONARY_CLASSES),
        pretrained: bool = True,
        dropout_rate: float = 0.25
    ):
        super(DenseNet121Pulmonary, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load DenseNet-121 backbone
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.densenet = models.densenet121(weights=weights)
        except Exception:
            self.densenet = models.densenet121(pretrained=False)

        num_features = self.densenet.classifier.in_features  # 1024

        # Customized Multi-Label Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Replace default classifier
        self.densenet.classifier = nn.Identity()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 1024-dimensional visual feature embedding vector for multimodal Digital Twin fusion.
        x: (Batch, 3, 224, 224)
        Returns: (Batch, 1024)
        """
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        embedding = torch.flatten(out, 1)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unnormalized logits for BCEWithLogitsLoss.
        x: (Batch, 3, 224, 224)
        Returns: logits (Batch, num_classes)
        """
        embeddings = self.extract_features(x)
        logits = self.classifier_head(embeddings)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return multi-label Sigmoid probabilities [0.0 - 1.0].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block in DenseNet-121 for Grad-CAM."""
        return self.densenet.features.denseblock4


class GradCAMExplainer:
    """
    Grad-CAM: Visual Explanations from Deep Networks for Chest X-Ray Radiographs.
    Calculates spatial activation gradients to highlight lesion regions.
    """

    def __init__(self, model: DenseNet121Pulmonary, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_cam_target_layer()
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int
    ) -> np.ndarray:
        """
        Generate 2D spatial Grad-CAM attention heatmap normalized to [0.0, 1.0].
        input_tensor: (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        logits = self.model(input_tensor)

        # Target class score
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach().cpu().numpy()[0]  # Shape (C, h, w)
        activations = self.activations.detach().cpu().numpy()[0]  # Shape (C, h, w)

        # Global Average Pooling of gradients -> feature weights alpha_k
        weights = np.mean(gradients, axis=(1, 2))  # Shape (C,)

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: isolate positive contributions
        cam = np.maximum(cam, 0)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        original_pil_image: Image.Image,
        cam_heatmap: np.ndarray,
        colormap_name: str = "jet",
        alpha: float = 0.40
    ) -> str:
        """
        Overlay Grad-CAM heatmap onto the original image and return Base64 encoded PNG string.
        """
        w, h = original_pil_image.size
        cam_pil = Image.fromarray(np.uint8(255 * cam_heatmap)).resize((w, h), Image.BILINEAR)
        cam_resized = np.array(cam_pil) / 255.0

        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap_name]
        except Exception:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap_name)

        colored_cam = cmap(cam_resized)[:, :, :3]
        colored_cam = np.uint8(255 * colored_cam)

        orig_rgb = original_pil_image.convert("RGB")
        orig_np = np.array(orig_rgb)

        blended = np.uint8(orig_np * (1.0 - alpha) + colored_cam * alpha)
        blended_pil = Image.fromarray(blended)

        buffered = io.BytesIO()
        blended_pil.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{b64}"

    overlay_heatmap_on_image = overlay_heatmap

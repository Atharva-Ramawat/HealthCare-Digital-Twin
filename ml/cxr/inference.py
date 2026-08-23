"""
MIMIC-CXR Inference Engine & Multimodal Digital Twin Feature Embedding Provider.
Executes trained DenseNet-121 inference, generates Grad-CAM attention visualizations,
and extracts 1024-dim visual embeddings for future multimodal Digital Twin fusion.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.cxr.labels import TARGET_PULMONARY_CLASSES, DEFAULT_PATHOLOGY_THRESHOLDS
from ml.cxr.preprocessing import load_and_preprocess_image, get_cxr_transforms
from ml.cxr.model import DenseNet121Pulmonary, GradCAMExplainer


class CXRInferenceEngine:
    """
    Production-ready CXR Inference and Explainability Engine.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        target_classes: List[str] = TARGET_PULMONARY_CLASSES,
        device: Optional[torch.device] = None
    ):
        self.target_classes = target_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or os.path.join(PROJECT_ROOT, "models", "checkpoints", "densenet121_mimic_cxr.pt")
        
        self.model: Optional[DenseNet121Pulmonary] = None
        self.explainer: Optional[GradCAMExplainer] = None
        self.is_model_available = False
        
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load trained weights from checkpoint path or mark as unavailable."""
        try:
            self.model = DenseNet121Pulmonary(num_classes=len(self.target_classes), pretrained=False)
            
            if os.path.exists(self.checkpoint_path):
                state = torch.load(self.checkpoint_path, map_location=self.device)
                # Check if it is a full checkpoint dict or state_dict
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                else:
                    self.model.load_state_dict(state)
                
                self.model.to(self.device)
                self.model.eval()
                self.explainer = GradCAMExplainer(self.model)
                self.is_model_available = True
                print(f"[CXRInference] Loaded trained model weights from {self.checkpoint_path}")
            else:
                self.is_model_available = False
                print(f"[CXRInference] Checkpoint not found at {self.checkpoint_path}. Model is unavailable.")
        except Exception as e:
            self.is_model_available = False
            print(f"[CXRInference] Warning: Failed to load checkpoint: {e}")

    def predict(
        self,
        image_source: Union[str, Image.Image, bytes],
        target_pathology_for_heatmap: str = "Pneumonia",
        colormap: str = "jet"
    ) -> Dict[str, Any]:
        """
        Execute full inference pass.
        Returns pathology probabilities, multi-label classifications, 1024-dim feature embedding, and Grad-CAM overlay.
        """
        if not self.is_model_available or self.model is None:
            return {
                "is_model_available": False,
                "model_version": "DenseNet121-MIMICCXR-v1.0",
                "model_status_message": "Model weights unavailable. Checkpoint missing.",
                "findings": [],
                "top_finding": "Unavailable",
                "top_probability": 0.0,
                "feature_embedding": None,
                "heatmap_overlay_base64": ""
            }

        # 1. Preprocess Image
        img_tensor, pil_img = load_and_preprocess_image(image_source)
        img_tensor = img_tensor.to(self.device)

        # 2. Forward Pass & Embedding Extraction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            embedding = self.model.extract_features(img_tensor).squeeze().cpu().numpy()

        # 3. Format Multi-Label Findings
        findings = []
        for i, class_name in enumerate(self.target_classes):
            prob = float(probs[i])
            thresh = DEFAULT_PATHOLOGY_THRESHOLDS.get(class_name, 0.50)
            is_pos = prob >= thresh

            if not is_pos:
                sev = "none"
            elif prob < 0.65:
                sev = "mild"
            elif prob < 0.85:
                sev = "moderate"
            else:
                sev = "severe"

            findings.append({
                "name": class_name,
                "probability": round(prob, 4),
                "threshold": thresh,
                "is_positive": is_pos,
                "severity": sev
            })

        sorted_findings = sorted(findings, key=lambda x: x["probability"], reverse=True)
        top_finding = sorted_findings[0]

        # 4. Generate Grad-CAM Attention Heatmap
        try:
            target_idx = self.target_classes.index(target_pathology_for_heatmap)
        except ValueError:
            target_idx = self.target_classes.index("Pneumonia") if "Pneumonia" in self.target_classes else 0

        cam_2d = self.explainer.generate_heatmap(img_tensor, target_idx)
        overlay_b64 = self.explainer.overlay_heatmap(pil_img, cam_2d, colormap_name=colormap)

        return {
            "is_model_available": True,
            "model_version": "DenseNet121-MIMICCXR-v1.0",
            "model_status_message": "Inference executed successfully on PyTorch DenseNet-121",
            "findings": findings,
            "top_finding": top_finding["name"],
            "top_probability": top_finding["probability"],
            "feature_embedding": embedding.tolist(),  # 1024-dimensional visual embedding
            "heatmap_overlay_base64": overlay_b64,
            "target_pathology_explained": self.target_classes[target_idx]
        }

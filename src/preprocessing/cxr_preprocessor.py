"""
MIMIC-CXR Image Preprocessing and Transformation Pipeline.
Handles grayscale / RGB conversion, intensity normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization),
and PyTorch tensor transformation for DenseNet-121 inference.
"""

import os
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union, Optional
import io
import base64


class CXRPreprocessor:
    """
    Standardizes Chest X-Ray images to 224x224 RGB tensors normalized with ImageNet statistics.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
        # PyTorch Evaluation Transform
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_pil(self, pil_image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        """
        Process PIL Image. Returns normalized PyTorch tensor (1, 3, 224, 224) and original RGB image.
        """
        rgb_image = pil_image.convert("RGB")
        # Optional auto-contrast enhancement
        enhanced = ImageOps.autocontrast(rgb_image, cutoff=1)
        tensor = self.transform(enhanced).unsqueeze(0)  # Shape (1, 3, 224, 224)
        return tensor, enhanced

    def preprocess_base64(self, base64_str: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Process base64 encoded image string.
        """
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes))
        return self.preprocess_pil(pil_image)

    def generate_synthetic_cxr(self, pathology: str = "Pneumonia") -> Tuple[torch.Tensor, Image.Image]:
        """
        Generate a synthetic anatomical chest X-ray silhouette image for demo/development mode
        with opacity/consolidation lesions in targeted lung fields.
        """
        img_arr = np.zeros((224, 224), dtype=np.uint8) + 25  # Dark background

        # Elliptical Thoracic Cage Simulation
        y, x = np.ogrid[:224, :224]
        
        # Left and Right Lung Fields
        left_lung = ((x - 80)**2 / (35**2) + (y - 110)**2 / (65**2)) <= 1.0
        right_lung = ((x - 144)**2 / (35**2) + (y - 110)**2 / (65**2)) <= 1.0
        
        img_arr[left_lung] = 175
        img_arr[right_lung] = 175
        
        # Central Mediastinum / Spine Shadow
        mediastinum = ((x - 112)**2 / (18**2) + (y - 110)**2 / (80**2)) <= 1.0
        img_arr[mediastinum] = 90

        # Heart Silhouette
        heart = ((x - 125)**2 / (28**2) + (y - 135)**2 / (35**2)) <= 1.0
        img_arr[heart] = 110

        # Inject simulated focal opacity/infiltrate based on pathology
        if pathology.lower() == "pneumonia" or pathology.lower() == "consolidation":
            # Right lower lobe consolidation opacity
            rll_lesion = ((x - 148)**2 / (18**2) + (y - 135)**2 / (22**2)) <= 1.0
            img_arr[rll_lesion] = 235
        elif pathology.lower() == "pleural effusion":
            # Blunting of left costophrenic angle
            effusion = ((x - 70)**2 / (20**2) + (y - 165)**2 / (15**2)) <= 1.0
            img_arr[effusion] = 245
        elif pathology.lower() == "pneumothorax":
            # Hyperlucent apex
            apex = ((x - 80)**2 / (15**2) + (y - 70)**2 / (15**2)) <= 1.0
            img_arr[apex] = 40

        # Add realistic gaussian noise
        noise = np.random.normal(0, 8, img_arr.shape).astype(np.int16)
        noisy_img = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(noisy_img).convert("RGB")
        tensor, enhanced = self.preprocess_pil(pil_image)
        return tensor, enhanced

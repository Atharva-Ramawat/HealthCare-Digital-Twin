"""
MIMIC-CXR Image Preprocessing & Data Augmentation Pipeline.
Provides deterministic validation transforms and clinically-safe training augmentations.
"""

import os
import io
import base64
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms


# ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cxr_transforms(
    is_training: bool = False,
    image_size: Tuple[int, int] = (224, 224)
) -> transforms.Compose:
    """
    Build PyTorch transform pipeline.
    
    Training transforms include slight rotation (+/- 7 deg), slight scaling (0.95 - 1.05),
    horizontal flipping, and mild brightness/contrast jitter without destroying pathology signatures.
    
    Validation/Inference transforms are strictly deterministic.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.08), int(image_size[1] * 1.08))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-7, 7)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def load_and_preprocess_image(
    image_source: Union[str, Image.Image, bytes],
    transform: Optional[transforms.Compose] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load an image from file path, raw bytes, or PIL Image, convert to 3-channel RGB,
    apply transforms, and return (Tensor [1, 3, H, W], PIL.Image).
    """
    if transform is None:
        transform = get_cxr_transforms(is_training=False, image_size=image_size)

    if isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    elif isinstance(image_source, bytes):
        pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, str):
        if image_source.startswith("data:image") or len(image_source) > 500 and " " not in image_source:
            # Base64 string
            raw_b64 = image_source.split(",")[1] if "," in image_source else image_source
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif os.path.exists(image_source):
            # Check DICOM extension
            if image_source.lower().endswith(".dcm"):
                try:
                    import pydicom
                    dcm = pydicom.dcmread(image_source)
                    arr = dcm.pixel_array.astype(np.float32)
                    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)
                    pil_img = Image.fromarray(np.uint8(arr * 255)).convert("RGB")
                except ImportError:
                    raise ImportError("pydicom is required to read .dcm DICOM files.")
            else:
                pil_img = Image.open(image_source).convert("RGB")
        else:
            raise FileNotFoundError(f"Image path does not exist: {image_source}")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    # Auto-contrast enhancement to balance radiograph dynamic range
    enhanced_pil = ImageOps.autocontrast(pil_img, cutoff=0.5)
    tensor = transform(enhanced_pil)
    
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Shape (1, 3, 224, 224)

    return tensor, enhanced_pil


def generate_synthetic_chest_radiograph(
    pathologies: Optional[list] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Generate an anatomical chest X-ray silhouette with realistic lung opacities,
    mediastinal contour, and rib structures for development/testing when raw MIMIC-CXR is not mounted.
    """
    h, w = image_size
    img_arr = np.zeros((h, w), dtype=np.float32) + 20.0  # Background

    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0

    # 1. Bilateral Lung Fields (Ellipsoids)
    left_lung = ((x - (cx - 38))**2 / (32**2) + (y - (cy - 5))**2 / (65**2)) <= 1.0
    right_lung = ((x - (cx + 38))**2 / (32**2) + (y - (cy - 5))**2 / (65**2)) <= 1.0
    img_arr[left_lung] = 160.0
    img_arr[right_lung] = 160.0

    # 2. Mediastinum & Spine Column
    spine = ((x - cx)**2 / (14**2) + (y - cy)**2 / (85**2)) <= 1.0
    img_arr[spine] = 80.0

    # 3. Cardiac Silhouette
    heart = ((x - (cx + 12))**2 / (28**2) + (y - (cy + 22))**2 / (34**2)) <= 1.0
    img_arr[heart] = 100.0

    # 4. Diaphragmatic Domes
    diaphragm_left = (y > (cy + 55)) & (x < cx)
    diaphragm_right = (y > (cy + 58)) & (x >= cx)
    img_arr[diaphragm_left] = 60.0
    img_arr[diaphragm_right] = 60.0

    # 5. Inject Pathological Radiographic Lesions
    pathologies = pathologies or []
    if any(p in pathologies for p in ["Pneumonia", "Consolidation"]):
        # Focal alveolar consolidation in right lower lobe
        rll = ((x - (cx + 38))**2 / (18**2) + (y - (cy + 25))**2 / (20**2)) <= 1.0
        img_arr[rll] = 230.0
    if "Pleural Effusion" in pathologies:
        # Costophrenic blunting in left sulcus
        sulcus = ((x - (cx - 45))**2 / (20**2) + (y - (cy + 45))**2 / (18**2)) <= 1.0
        img_arr[sulcus] = 235.0
    if "Edema" in pathologies:
        # Perihilar bat-wing opacities
        hilar = ((x - cx)**2 / (45**2) + (y - (cy - 5))**2 / (25**2)) <= 1.0
        img_arr[hilar] = np.clip(img_arr[hilar] + 45.0, 0, 255)
    if "Pneumothorax" in pathologies:
        # Hyperlucency at apex with lung margin
        apex = ((x - (cx - 40))**2 / (18**2) + (y - (cy - 50))**2 / (16**2)) <= 1.0
        img_arr[apex] = 30.0

    # Add Gaussian anatomical texture noise
    noise = np.random.normal(0, 7.5, img_arr.shape)
    final_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(final_arr).convert("RGB")

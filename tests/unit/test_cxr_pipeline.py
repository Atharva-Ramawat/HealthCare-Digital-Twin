"""
Unit Tests for MIMIC-CXR Machine Learning Pipeline (ml/cxr).
Tests label mapping, split verification, DenseNet-121 forward pass, feature extraction, evaluation, and Grad-CAM.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ml.cxr.labels import (
    TARGET_PULMONARY_CLASSES,
    UncertaintyPolicy,
    map_chexpert_labels,
    verify_patient_level_split,
    calculate_positive_class_weights
)
from ml.cxr.preprocessing import (
    get_cxr_transforms,
    load_and_preprocess_image,
    generate_synthetic_chest_radiograph
)
from ml.cxr.model import DenseNet121Pulmonary, GradCAMExplainer
from ml.cxr.inference import CXRInferenceEngine


def test_label_mapping_and_uncertainty():
    raw_df = pd.DataFrame({
        "subject_id": ["P1", "P2", "P3"],
        "study_id": ["S1", "S2", "S3"],
        "Pneumonia": [1.0, -1.0, 0.0],
        "Pleural Effusion": [0.0, 1.0, -1.0],
        "Atelectasis": [np.nan, 0.0, 1.0],
        "No Finding": [np.nan, 0.0, 0.0]
    })
    
    # U-Zero mapping
    df_u0 = map_chexpert_labels(raw_df, target_classes=TARGET_PULMONARY_CLASSES, policy=UncertaintyPolicy.U_ZERO)
    assert df_u0.loc[df_u0["subject_id"] == "P1", "Pneumonia"].values[0] == 1.0
    assert df_u0.loc[df_u0["subject_id"] == "P2", "Pneumonia"].values[0] == 0.0  # -1 -> 0

    # U-Ones mapping
    df_u1 = map_chexpert_labels(raw_df, target_classes=TARGET_PULMONARY_CLASSES, policy=UncertaintyPolicy.U_ONES)
    assert df_u1.loc[df_u1["subject_id"] == "P2", "Pneumonia"].values[0] == 1.0  # -1 -> 1


def test_patient_split_verification():
    # Disjoint split
    df_good = pd.DataFrame({
        "subject_id": ["P1", "P1", "P2", "P3", "P4"],
        "split": ["train", "train", "train", "validate", "test"]
    })
    res_good = verify_patient_level_split(df_good)
    assert res_good["is_valid"] is True
    assert len(res_good["leakage_overlaps"]) == 0

    # Contaminated split (data leakage)
    df_leaky = pd.DataFrame({
        "subject_id": ["P1", "P1", "P2", "P1", "P4"],
        "split": ["train", "train", "train", "validate", "test"]
    })
    res_leaky = verify_patient_level_split(df_leaky)
    assert res_leaky["is_valid"] is False
    assert "train_vs_validate" in res_leaky["leakage_overlaps"]


def test_class_weights_calculation():
    labels = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    weights = calculate_positive_class_weights(labels)
    assert weights.shape == (3,)
    assert weights[0] == pytest.approx(1.0, rel=1e-4)
    assert weights[1] == pytest.approx(1.0, rel=1e-4)
    assert weights[2] == pytest.approx(3.0, rel=1e-4)


def test_densenet121_model_forward_and_embeddings():
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)

    # 1. Forward Pass
    logits = model(dummy_input)
    assert logits.shape == (2, 8)

    # 2. Probability prediction
    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (2, 8)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    # 3. Multimodal Latent Embedding Extraction (1024-dim)
    embeddings = model.extract_features(dummy_input)
    assert embeddings.shape == (2, 1024)


def test_gradcam_explainer():
    model = DenseNet121Pulmonary(num_classes=8, pretrained=False)
    explainer = GradCAMExplainer(model)
    
    img = generate_synthetic_chest_radiograph(["Pneumonia"])
    img_tensor, _ = load_and_preprocess_image(img)

    cam_2d = explainer.generate_heatmap(img_tensor, class_idx=0)
    assert cam_2d.shape == (7, 7) or len(cam_2d.shape) == 2
    assert np.min(cam_2d) >= 0.0 and np.max(cam_2d) <= 1.0

    overlay_b64 = explainer.overlay_heatmap(img, cam_2d, colormap_name="jet")
    assert overlay_b64.startswith("data:image/png;base64,")
    assert len(overlay_b64) > 1000

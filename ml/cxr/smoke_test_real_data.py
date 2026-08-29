"""
MIMIC-CXR Real Image Smoke Test & Visual Verification Montage Generator.
Loads actual disk JPGs, executes DenseNet-121 forward pass with zero NaNs/Infs,
and creates a visual verification montage of real radiographs.
Outputs:
- ml/cxr/outputs/real_cxr_smoke_test_results.json
- ml/cxr/outputs/real_cxr_verification_montage.png
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.cxr.config import CXRConfig
from ml.cxr.dataset import MIMICCXRDataset, create_cxr_dataloaders
from ml.cxr.model import DenseNet121Pulmonary
from ml.cxr.labels import TARGET_PULMONARY_CLASSES


def create_verification_montage(
    image_records: List[Dict[str, Any]],
    output_path: str,
    grid_size: Tuple[int, int] = (2, 4),
    thumb_size: Tuple[int, int] = (250, 250)
):
    """
    Generate a 2x4 montage image of real MIMIC-CXR radiographs with clinical metadata overlays.
    """
    n_rows, n_cols = grid_size
    padding = 10
    header_h = 45
    card_w = thumb_size[0] + padding * 2
    card_h = thumb_size[1] + header_h + padding * 2

    montage_w = card_w * n_cols
    montage_h = card_h * n_rows + 50

    montage = Image.new("RGB", (montage_w, montage_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(montage)

    # Title Banner
    draw.text((20, 15), "MIMIC-CXR Real Radiograph Integration Verification (Actual Disk JPGs)", fill=(255, 255, 255))

    for idx, rec in enumerate(image_records[:n_rows * n_cols]):
        r = idx // n_cols
        c = idx % n_cols

        x_offset = c * card_w + padding
        y_offset = r * card_h + 50 + padding

        # Load and resize real JPG
        try:
            raw_img = Image.open(rec["resolved_path"]).convert("RGB")
            thumb = raw_img.resize(thumb_size)
            montage.paste(thumb, (x_offset, y_offset + header_h))
        except Exception as e:
            draw.rectangle([x_offset, y_offset + header_h, x_offset + thumb_size[0], y_offset + header_h + thumb_size[1]], fill=(50, 50, 50))
            draw.text((x_offset + 10, y_offset + header_h + 10), f"Error: {e}", fill=(255, 100, 100))

        # Metadata Header Box
        draw.rectangle([x_offset, y_offset, x_offset + thumb_size[0], y_offset + header_h], fill=(35, 42, 58))
        pat_info = f"Subj: {rec['subject_id']} | View: {rec['view_position']}"
        pos_findings = [cls for cls, val in zip(TARGET_PULMONARY_CLASSES, rec["labels"]) if val == 1.0]
        finding_str = ", ".join(pos_findings[:2]) if pos_findings else "No Finding"

        draw.text((x_offset + 5, y_offset + 4), pat_info, fill=(100, 200, 255))
        draw.text((x_offset + 5, y_offset + 22), f"Path: {finding_str}", fill=(120, 255, 120) if "No Finding" in finding_str else (255, 200, 100))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    montage.save(output_path)
    print(f"[Visual Verification] Saved real image montage to: {output_path}")


def run_real_image_smoke_test() -> Dict[str, Any]:
    """
    Execute real MIMIC-CXR smoke test loading actual disk JPGs and running DenseNet-121 forward pass.
    """
    print("\n" + "=" * 70)
    print("       MIMIC-CXR REAL-IMAGE SMOKE TEST & FORWARD PASS AUDIT")
    print("=" * 70)

    config = CXRConfig(allow_synthetic_fallback=False)
    device = config.hardware.device
    print(f"Target Device        : {device} ({config.hardware.device_name})")
    print(f"CXR Image Root       : {config.image_root}")
    print(f"Fail-Fast Policy     : Synthetic Fallback Strictly Disabled (allow_synthetic_fallback=False)")

    # 1. Instantiate Real Validation Dataset
    print("\n[Step 1/3] Loading Real Validation Image Batch...")
    val_dataset = MIMICCXRDataset(
        df=config.val_metadata_path,
        images_dir=config.image_root,
        split="validate",
        target_classes=TARGET_PULMONARY_CLASSES,
        allow_synthetic_fallback=False,  # Enforce fail-fast
        filter_missing_images=True,
        max_rows=50
    )
    print(f"Loaded {len(val_dataset)} verified real validation images on disk.")

    # 2. Instantiate Real Training Dataset (sample 50 rows)
    train_dataset = MIMICCXRDataset(
        df=config.train_metadata_path,
        images_dir=config.image_root,
        split="train",
        target_classes=TARGET_PULMONARY_CLASSES,
        allow_synthetic_fallback=False,
        filter_missing_images=True,
        max_rows=50
    )
    print(f"Loaded {len(train_dataset)} verified real training images on disk.")

    # 3. Inspect Sample Real Images
    sample_records = []
    print("\n--- Detailed Sample Real Image Audits ---")
    for i in range(min(8, len(val_dataset))):
        img_tensor, label_tensor, meta = val_dataset[i]
        assert os.path.exists(meta["resolved_path"]), f"Physical file missing: {meta['resolved_path']}"
        assert not torch.isnan(img_tensor).any(), "NaN found in preprocessed image tensor!"
        assert not torch.isinf(img_tensor).any(), "Inf found in preprocessed image tensor!"

        raw_img = Image.open(meta["resolved_path"])
        rec = {
            "index": i,
            "subject_id": meta["subject_id"],
            "study_id": meta["study_id"],
            "dicom_id": meta["dicom_id"],
            "view_position": meta["view_position"],
            "split": meta["split"],
            "resolved_path": meta["resolved_path"],
            "raw_dimensions": list(raw_img.size),
            "tensor_shape": list(img_tensor.shape),
            "tensor_dtype": str(img_tensor.dtype),
            "labels": label_tensor.numpy().tolist(),
            "positive_findings": [cls for cls, v in zip(TARGET_PULMONARY_CLASSES, label_tensor.numpy()) if v == 1.0]
        }
        sample_records.append(rec)
        print(
            f"[{i+1}/8] Subj: {rec['subject_id']} | Study: {rec['study_id']} | "
            f"Raw: {rec['raw_dimensions']} | Tensor: {rec['tensor_shape']} | "
            f"Findings: {rec['positive_findings']} | Path: {os.path.basename(rec['resolved_path'])}"
        )

    # 4. Run DenseNet-121 Forward Pass on Real Batch
    print("\n[Step 2/3] Executing DenseNet-121 Multi-Label Forward Pass...")
    model = DenseNet121Pulmonary(num_classes=len(TARGET_PULMONARY_CLASSES), pretrained=False).to(device)
    model.eval()

    batch_tensors = torch.stack([val_dataset[i][0] for i in range(min(8, len(val_dataset)))]).to(device)
    with torch.no_grad():
        t0 = time.time()
        logits = model(batch_tensors)
        probabilities = torch.sigmoid(logits)
        forward_time = round(time.time() - t0, 4)

    assert logits.shape == (batch_tensors.shape[0], len(TARGET_PULMONARY_CLASSES)), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN found in DenseNet output logits!"
    assert not torch.isinf(logits).any(), "Inf found in DenseNet output logits!"
    print(f"Forward Pass Completed in {forward_time}s | Input Shape: {list(batch_tensors.shape)} | Output Shape: {list(logits.shape)}")
    print(f"Logits Min/Max       : [{float(logits.min()):.4f}, {float(logits.max()):.4f}] (0 NaNs, 0 Infs)")

    # 5. Create Real Image Verification Montage
    print("\n[Step 3/3] Generating Visual Verification Montage of Real Radiographs...")
    montage_path = os.path.join(config.outputs_dir, "real_cxr_verification_montage.png")
    create_verification_montage(sample_records, montage_path)

    # 6. Save Smoke Results
    smoke_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASSED",
        "device": str(device),
        "forward_time_sec": forward_time,
        "input_batch_shape": list(batch_tensors.shape),
        "output_logits_shape": list(logits.shape),
        "sample_images_audited": sample_records,
        "verification_montage_path": montage_path
    }
    out_json = os.path.join(config.outputs_dir, "real_cxr_smoke_test_results.json")
    with open(out_json, "w") as f:
        json.dump(smoke_results, f, indent=2)
    print(f"Saved Smoke Test JSON to: {out_json}")
    print("=" * 70 + "\n")

    return smoke_results


if __name__ == "__main__":
    run_real_image_smoke_test()

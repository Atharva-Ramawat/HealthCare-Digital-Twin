"""
Batch Ingestion Script for MIMIC-CXR Augmented Metadata and Disk Radiographs.
Reads mimic_cxr_aug_train.csv and mimic_cxr_aug_validate.csv, validates physical JPG files
against CXR_IMAGE_ROOT, enforces a strict fail-fast policy (zero synthetic fallback),
logs audit violations for missing images, and performs bulk database insertion into
PostgreSQL / SQLite via SQLAlchemy.
"""

import os
import sys
import ast
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
from sqlalchemy import select, insert
from sqlalchemy.orm import Session

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.models import Base, Patient, CXRStudy
from src.database.connection import get_engine, init_db
from ml.cxr.labels import TARGET_PULMONARY_CLASSES, extract_labels_from_radiology_report

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("mimic_cxr_ingest")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the batch ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest MIMIC-CXR patient & study metadata into database.")
    parser.add_argument(
        "--train-csv",
        type=str,
        default=os.environ.get("CXR_TRAIN_METADATA", os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr_aug_train.csv")),
        help="Path to training augmented metadata CSV."
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=os.environ.get("CXR_VAL_METADATA", os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr_aug_validate.csv")),
        help="Path to validation augmented metadata CSV."
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=os.environ.get("CXR_IMAGE_ROOT", os.path.join(PROJECT_ROOT, "data", "raw", "mimic_cxr_aug_validate")),
        help="Root directory containing the physical CXR image tree (files/pXX/...)."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database connection URL (PostgreSQL / SQLite). Defaults to DATABASE_URL env var."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for bulk database insertions."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to ingest per split (useful for testing and partial ingests)."
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise FileNotFoundError immediately upon encountering any missing physical image."
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip ingesting the training set."
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip ingesting the validation set."
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Perform physical path verification and audit without writing to the database."
    )
    return parser.parse_args()


def process_csv_split(
    csv_path: str,
    image_root: str,
    split_name: str,
    fail_fast: bool = False,
    max_rows: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse CSV rows into validated Patient and CXRStudy records while auditing physical file existence.
    Returns: (patients_list, valid_studies_list, audit_violations_list)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    logger.info(f"Loading {split_name} metadata from: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)
    logger.info(f"Read {len(df)} patient records for split: {split_name}")

    patients_dict: Dict[str, Dict[str, Any]] = {}
    valid_studies: List[Dict[str, Any]] = []
    audit_violations: List[Dict[str, Any]] = []

    image_root_abs = os.path.abspath(image_root)

    for idx, row in df.iterrows():
        subj_id = str(row.get("subject_id", "")).strip()
        if not subj_id:
            continue

        # Patient record mapping
        if subj_id not in patients_dict:
            patients_dict[subj_id] = {
                "subject_id": subj_id,
                "name": f"Patient {subj_id}",
                "gender": None,
                "age": None,
                "split": split_name,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

        # 1. Parse Image List
        img_col = row.get("image", "[]")
        if isinstance(img_col, str) and img_col.strip().startswith("["):
            try:
                img_list = ast.literal_eval(img_col)
            except Exception:
                img_list = [img_col]
        elif isinstance(img_col, list):
            img_list = img_col
        elif pd.notna(img_col) and str(img_col).strip():
            img_list = [str(img_col)]
        else:
            img_list = []

        # 2. Parse Text Reports List
        text_col = row.get("text", "[]")
        if isinstance(text_col, str) and text_col.strip().startswith("["):
            try:
                text_list = ast.literal_eval(text_col)
            except Exception:
                text_list = [text_col]
        elif isinstance(text_col, list):
            text_list = text_col
        elif pd.notna(text_col) and str(text_col).strip():
            text_list = [str(text_col)]
        else:
            text_list = []

        # 3. Parse View Positions List
        view_col = row.get("view", "[]")
        if isinstance(view_col, str) and view_col.strip().startswith("["):
            try:
                view_list = ast.literal_eval(view_col)
            except Exception:
                view_list = [view_col]
        elif isinstance(view_col, list):
            view_list = view_col
        else:
            view_list = [str(view_col)] if pd.notna(view_col) else []

        # Map study_ids to distinct text reports
        study_order = list(dict.fromkeys([p.split("/")[3] for p in img_list if len(p.split("/")) > 3]))
        study_to_text = {}
        for s_idx, s_id in enumerate(study_order):
            study_to_text[s_id] = text_list[s_idx] if s_idx < len(text_list) else ""

        for img_idx, img_rel in enumerate(img_list):
            img_rel_clean = str(img_rel).strip().replace("\\", "/")
            parts = img_rel_clean.split("/")
            study_id = parts[3] if len(parts) > 3 else f"STD-{idx:04d}"
            dicom_id = parts[4].replace(".jpg", "").replace(".png", "") if len(parts) > 4 else f"DCM-{idx:04d}"
            view_pos = view_list[img_idx] if img_idx < len(view_list) else "PA"

            # Physical Path Resolution
            full_path = os.path.normpath(os.path.join(image_root_abs, img_rel_clean))
            file_exists = os.path.exists(full_path)

            if not file_exists:
                # AUDIT VIOLATION: Image path does not exist on disk
                violation = {
                    "split": split_name,
                    "subject_id": subj_id,
                    "study_id": study_id,
                    "dicom_id": dicom_id,
                    "image_path": img_rel_clean,
                    "attempted_physical_path": full_path,
                    "reason": "FILE_NOT_FOUND_ON_DISK",
                    "timestamp": datetime.utcnow().isoformat()
                }
                audit_violations.append(violation)

                if fail_fast:
                    raise FileNotFoundError(
                        f"CRITICAL AUDIT VIOLATION [Fail-Fast]: Physical image not found at '{full_path}' "
                        f"for subject_id={subj_id}, study_id={study_id}. "
                        f"Synthetic fallbacks are strictly prohibited."
                    )
                continue

            # Extract 8 pulmonary pathology labels from clinical report
            report = study_to_text.get(study_id, text_list[0] if text_list else "")
            labels = extract_labels_from_radiology_report(report, target_classes=TARGET_PULMONARY_CLASSES)

            study_record = {
                "study_id": study_id,
                "dicom_id": dicom_id,
                "subject_id": subj_id,
                "view_position": view_pos,
                "image_path": img_rel_clean,
                "resolved_path": full_path,
                "split": split_name,
                "report_text": report,
                "pneumonia": labels.get("Pneumonia", 0.0),
                "pleural_effusion": labels.get("Pleural Effusion", 0.0),
                "atelectasis": labels.get("Atelectasis", 0.0),
                "consolidation": labels.get("Consolidation", 0.0),
                "edema": labels.get("Edema", 0.0),
                "pneumothorax": labels.get("Pneumothorax", 0.0),
                "cardiomegaly": labels.get("Cardiomegaly", 0.0),
                "no_finding": labels.get("No Finding", 0.0),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            valid_studies.append(study_record)

    patients_list = list(patients_dict.values())
    logger.info(
        f"[{split_name.upper()} Summary] Patients: {len(patients_list)} | "
        f"Verified Disk Images: {len(valid_studies)} | "
        f"Audit Violations (Missing Files): {len(audit_violations)}"
    )

    return patients_list, valid_studies, audit_violations


def bulk_insert_records(
    session: Session,
    patients: List[Dict[str, Any]],
    studies: List[Dict[str, Any]],
    batch_size: int = 1000
) -> Tuple[int, int]:
    """
    Execute efficient bulk chunked insertions into Patient and CXRStudy tables.
    Returns: (inserted_patients_count, inserted_studies_count)
    """
    # 1. Fetch already existing subject_ids to prevent primary key / unique constraint collisions
    existing_subjects: Set[str] = set(
        session.scalars(select(Patient.subject_id)).all()
    )

    new_patients = [p for p in patients if p["subject_id"] not in existing_subjects]
    logger.info(f"Inserting {len(new_patients)} new patients ({len(patients) - len(new_patients)} already exist in DB)...")

    # Bulk insert patients
    for i in range(0, len(new_patients), batch_size):
        chunk = new_patients[i:i + batch_size]
        session.bulk_insert_mappings(Patient, chunk)
        session.commit()

    # 2. Fetch existing study-dicom pairs to avoid duplicate study insertions
    existing_studies_set: Set[Tuple[str, Optional[str]]] = set(
        session.execute(select(CXRStudy.study_id, CXRStudy.dicom_id)).all()
    )

    new_studies = [
        s for s in studies
        if (s["study_id"], s["dicom_id"]) not in existing_studies_set
    ]
    logger.info(f"Inserting {len(new_studies)} verified CXR studies in chunks of {batch_size}...")

    # Bulk insert studies in chunks
    for i in range(0, len(new_studies), batch_size):
        chunk = new_studies[i:i + batch_size]
        session.bulk_insert_mappings(CXRStudy, chunk)
        session.commit()
        if (i + batch_size) % (batch_size * 5) == 0 or (i + len(chunk)) == len(new_studies):
            logger.info(f"  Committed {min(i + len(chunk), len(new_studies))}/{len(new_studies)} studies to database.")

    return len(new_patients), len(new_studies)


def run_ingestion_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the end-to-end MIMIC-CXR ingestion and audit process."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("       MIMIC-CXR BATCH INGESTION & PHYSICAL DISK AUDIT")
    logger.info("=" * 70)
    logger.info(f"Train CSV        : {args.train_csv}")
    logger.info(f"Validation CSV   : {args.val_csv}")
    logger.info(f"CXR Image Root   : {args.image_root}")
    logger.info(f"Fail-Fast Mode   : {args.fail_fast} (No synthetic fallbacks allowed)")
    logger.info(f"Audit Only Mode  : {args.audit_only}")

    if not os.path.exists(args.image_root):
        raise FileNotFoundError(f"CXR_IMAGE_ROOT directory does not exist: {args.image_root}")

    all_patients: List[Dict[str, Any]] = []
    all_studies: List[Dict[str, Any]] = []
    all_violations: List[Dict[str, Any]] = []

    # 1. Process Validation Split
    if not args.skip_val:
        val_patients, val_studies, val_violations = process_csv_split(
            csv_path=args.val_csv,
            image_root=args.image_root,
            split_name="validate",
            fail_fast=args.fail_fast,
            max_rows=args.max_rows
        )
        all_patients.extend(val_patients)
        all_studies.extend(val_studies)
        all_violations.extend(val_violations)

    # 2. Process Training Split
    if not args.skip_train:
        train_patients, train_studies, train_violations = process_csv_split(
            csv_path=args.train_csv,
            image_root=args.image_root,
            split_name="train",
            fail_fast=args.fail_fast,
            max_rows=args.max_rows
        )
        all_patients.extend(train_patients)
        all_studies.extend(train_studies)
        all_violations.extend(train_violations)

    # 3. Database Insertion (unless audit-only)
    patients_inserted = 0
    studies_inserted = 0

    if not args.audit_only:
        engine = get_engine(db_url=args.db_url)
        init_db(engine)
        logger.info(f"Connected to database: {engine.url}")

        with Session(engine) as session:
            patients_inserted, studies_inserted = bulk_insert_records(
                session=session,
                patients=all_patients,
                studies=all_studies,
                batch_size=args.batch_size
            )
    else:
        logger.info("[AUDIT ONLY] Skipping database insertions as requested.")

    elapsed_sec = round(time.time() - start_time, 2)

    # 4. Generate Ingestion Audit Report
    audit_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_seconds": elapsed_sec,
        "parameters": {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "image_root": args.image_root,
            "fail_fast": args.fail_fast,
            "audit_only": args.audit_only
        },
        "metrics": {
            "total_patients_parsed": len(all_patients),
            "total_patients_inserted": patients_inserted,
            "total_valid_studies_verified": len(all_studies),
            "total_studies_inserted": studies_inserted,
            "total_missing_image_violations": len(all_violations),
            "disk_validity_percentage": round(
                (len(all_studies) / max(1, len(all_studies) + len(all_violations))) * 100, 2
            )
        },
        "sample_audit_violations": all_violations[:25],
        "status": "COMPLETED"
    }

    # Save Audit JSON and Violation Log
    output_dir = os.path.join(PROJECT_ROOT, "ml", "cxr", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    audit_json_path = os.path.join(output_dir, "ingestion_audit.json")
    with open(audit_json_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    violation_log_path = os.path.join(output_dir, "ingestion_audit_violations.log")
    with open(violation_log_path, "w") as f:
        for v in all_violations:
            f.write(
                f"[{v['timestamp']}] VIOLATION | Split: {v['split']} | "
                f"Subject: {v['subject_id']} | Study: {v['study_id']} | "
                f"Attempted Path: {v['attempted_physical_path']} | Reason: {v['reason']}\n"
            )

    logger.info("=" * 70)
    logger.info(f"Ingestion Pipeline Finished in {elapsed_sec}s")
    logger.info(f" - Patients Parsed       : {len(all_patients)}")
    logger.info(f" - Patients Inserted     : {patients_inserted}")
    logger.info(f" - Verified Disk Images  : {len(all_studies)}")
    logger.info(f" - Studies Inserted      : {studies_inserted}")
    logger.info(f" - Missing Image Violations: {len(all_violations)}")
    logger.info(f" - Saved Audit JSON      : {audit_json_path}")
    logger.info(f" - Saved Violations Log  : {violation_log_path}")
    logger.info("=" * 70)

    return audit_report


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_ingestion_pipeline(cli_args)

"""
MIMIC-IV Data Inventory & Dataset Profiler.
Scans and profiles all available hospital and ICU clinical tables in the configured MIMIC-IV environment.
Outputs complete structural metadata, identifier availability, record counts, and missingness rates.
"""

import os
import json
import gzip
import time
from typing import Dict, List, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generate_data_inventory(
    hosp_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "hosp"),
    icu_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "icu"),
    output_json_path: str = os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs", "data_inventory.json")
) -> Dict[str, Any]:
    """
    Profile all tables in hosp and icu subdirectories and save data_inventory.json.
    """
    inventory = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_source": "MIMIC-IV Clinical Database Demo v2.2",
        "hosp_directory": hosp_dir,
        "icu_directory": icu_dir,
        "tables": {}
    }

    directories = [("hosp", hosp_dir), ("icu", icu_dir)]
    total_files_profiled = 0
    total_rows_across_dataset = 0

    for section_name, dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"[Warning] Directory {dir_path} does not exist.")
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not (fname.endswith(".csv.gz") or fname.endswith(".csv")):
                continue

            table_key = f"{section_name}/{fname}"
            full_path = os.path.join(dir_path, fname)
            file_size_bytes = os.path.getsize(full_path)

            print(f"[Inventory] Profiling {table_key} ({file_size_bytes / 1024:.1f} KB)...")

            # Load full table into pandas for accurate profiling
            df = pd.read_csv(full_path, low_memory=False)
            num_rows, num_cols = df.shape
            total_files_profiled += 1
            total_rows_across_dataset += num_rows

            # Identify key columns
            has_subject = "subject_id" in df.columns
            has_hadm = "hadm_id" in df.columns
            has_stay = "stay_id" in df.columns

            n_unique_subjects = int(df["subject_id"].nunique()) if has_subject else 0
            n_unique_hadm = int(df["hadm_id"].nunique()) if has_hadm else 0
            n_unique_stay = int(df["stay_id"].nunique()) if has_stay else 0

            # Detect timestamp columns
            time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date"])]
            time_ranges = {}
            for tc in time_cols:
                valid_ts = df[tc].dropna()
                if len(valid_ts) > 0:
                    time_ranges[tc] = {
                        "min": str(valid_ts.min()),
                        "max": str(valid_ts.max()),
                        "non_null_count": int(len(valid_ts)),
                        "null_count": int(df[tc].isnull().sum())
                    }

            # Column-level missingness and dtypes
            columns_meta = {}
            for col in df.columns:
                null_cnt = int(df[col].isnull().sum())
                null_pct = round(float(null_cnt / max(1, num_rows)) * 100.0, 2)
                columns_meta[col] = {
                    "dtype": str(df[col].dtype),
                    "null_count": null_cnt,
                    "missing_percentage": null_pct,
                    "sample_values": [str(x) for x in df[col].dropna().head(3).tolist()]
                }

            inventory["tables"][table_key] = {
                "section": section_name,
                "file_name": fname,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_bytes / 1024, 2),
                "row_count": num_rows,
                "column_count": num_cols,
                "identifiers": {
                    "has_subject_id": has_subject,
                    "unique_subjects": n_unique_subjects,
                    "has_hadm_id": has_hadm,
                    "unique_hadm_ids": n_unique_hadm,
                    "has_stay_id": has_stay,
                    "unique_stay_ids": n_unique_stay
                },
                "timestamp_columns": time_ranges,
                "columns": columns_meta
            }

    inventory["summary"] = {
        "total_tables_profiled": total_files_profiled,
        "total_rows_dataset": total_rows_across_dataset,
        "patient_registry_count": inventory["tables"].get("hosp/patients.csv.gz", {}).get("row_count", 0),
        "icu_stays_count": inventory["tables"].get("icu/icustays.csv.gz", {}).get("row_count", 0),
        "admissions_count": inventory["tables"].get("hosp/admissions.csv.gz", {}).get("row_count", 0)
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(inventory, f, indent=2)

    print(f"\n[Inventory] Complete. Profiled {total_files_profiled} tables ({total_rows_across_dataset} rows).")
    print(f"[Inventory] Saved structured report to {output_json_path}")
    return inventory


if __name__ == "__main__":
    generate_data_inventory()

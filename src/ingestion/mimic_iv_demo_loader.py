"""
mimic_iv_demo_loader.py
-----------------------
Phase 1 — Thin subclass of MIMICIVLoader pointing at MIMIC-IV Demo.

MIMIC-IV Clinical Database Demo v2.2:
  - 100-patient subset of MIMIC-IV
  - Publicly available (free PhysioNet account, no credentialing required)
  - Download: https://physionet.org/content/mimic-iv-demo/2.2/
  - Local path: data/raw/mimic_iv_demo/

STATUS: LOCALLY AVAILABLE (28/28 tables downloaded in Phase 1)

PRIMARY DEVELOPMENT DATASET for Phase 1 EDA and ingestion testing.
Full MIMIC-IV is the intended research dataset (requires CITI credentialing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.mimic_iv_loader import MIMICIVLoader

DEMO_DEFAULT_ROOT = Path("data") / "raw" / "mimic_iv_demo"


class MIMICIVDemoLoader(MIMICIVLoader):
    """
    Loader for MIMIC-IV Demo (100-patient subset).

    Identical API to MIMICIVLoader. Only the default data root differs.
    Use this for Phase 1 EDA and ingestion infrastructure testing.

    Parameters
    ----------
    data_root : str, optional
        Path to the mimic_iv_demo root directory.
        Default: data/raw/mimic_iv_demo/ (relative to project root).
    chunksize : int, optional
        Chunk size for large table loading. None = load all at once.
        The Demo tables are small enough that chunking is not needed.
    """

    DATASET_LABEL = "MIMIC-IV Clinical Database Demo v2.2 (100 patients)"
    PATIENT_COUNT_DOCUMENTED = 100   # from PhysioNet dataset page
    DATASET_VERSION = "2.2"
    PHYSIONET_URL = "https://physionet.org/content/mimic-iv-demo/2.2/"

    def __init__(
        self,
        data_root: Optional[str] = None,
        chunksize: Optional[int] = None,
    ):
        resolved = data_root if data_root is not None else str(DEMO_DEFAULT_ROOT)
        super().__init__(data_root=resolved, chunksize=chunksize)

    def __repr__(self) -> str:
        return (
            f"MIMICIVDemoLoader(root='{self.data_root}', "
            f"version='{self.DATASET_VERSION}')"
        )

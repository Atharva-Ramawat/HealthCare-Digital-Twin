"""
data_availability.py
--------------------
Phase 1 — Dataset availability checker.

Scans configured data directories and reports which datasets and tables
are locally available. Used by all loaders to give actionable error messages.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TableAvailability:
    name: str
    expected_path: str
    exists: bool
    size_bytes: int = 0
    size_human: str = ""

    def __post_init__(self):
        if self.exists and self.size_bytes > 0:
            self.size_human = _human_size(self.size_bytes)


@dataclass
class DatasetAvailability:
    dataset_name: str
    root_path: str
    root_exists: bool
    tables: list[TableAvailability] = field(default_factory=list)
    notes: str = ""

    @property
    def is_available(self) -> bool:
        return self.root_exists and any(t.exists for t in self.tables)

    @property
    def available_tables(self) -> list[TableAvailability]:
        return [t for t in self.tables if t.exists]

    @property
    def missing_tables(self) -> list[TableAvailability]:
        return [t for t in self.tables if not t.exists]


@dataclass
class AvailabilityReport:
    datasets: list[DatasetAvailability] = field(default_factory=list)

    def get(self, dataset_name: str) -> Optional[DatasetAvailability]:
        for ds in self.datasets:
            if ds.dataset_name == dataset_name:
                return ds
        return None

    def summary(self) -> str:
        lines = ["Dataset Availability Report", "=" * 40]
        for ds in self.datasets:
            status = "AVAILABLE" if ds.is_available else "NOT AVAILABLE"
            lines.append(f"\n{ds.dataset_name}: {status}")
            lines.append(f"  Root: {ds.root_path}")
            if ds.is_available:
                for t in ds.available_tables:
                    lines.append(f"  [OK]  {t.name} ({t.size_human})")
            if ds.missing_tables:
                for t in ds.missing_tables:
                    lines.append(f"  [--]  {t.name} (not found)")
            if ds.notes:
                lines.append(f"  Note: {ds.notes}")
        return "\n".join(lines)


class DataNotAvailableError(FileNotFoundError):
    """
    Raised when a required dataset or table is not found locally.
    The error message includes instructions on how to obtain the data.
    """
    pass


# Required tables per dataset (filenames to look for)
MIMIC_IV_REQUIRED_TABLES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/d_labitems.csv.gz",
]

MIMIC_IV_OPTIONAL_TABLES = [
    "icu/inputevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
    "icu/datetimeevents.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/prescriptions.csv.gz",
]


class DataAvailabilityChecker:
    """
    Scans project data directories and reports availability of all
    configured datasets.

    Usage::

        checker = DataAvailabilityChecker(project_root=".")
        report = checker.check_all()
        print(report.summary())
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()

    def _raw(self, *parts: str) -> Path:
        return self.project_root / "data" / "raw" / Path(*parts)

    def _check_table(self, root: Path, rel_path: str) -> TableAvailability:
        full = root / rel_path
        # Also check without .gz extension (uncompressed variant)
        alt = root / rel_path.replace(".gz", "")
        for candidate in [full, alt]:
            if candidate.exists():
                size = candidate.stat().st_size
                return TableAvailability(
                    name=rel_path,
                    expected_path=str(candidate),
                    exists=True,
                    size_bytes=size,
                )
        return TableAvailability(
            name=rel_path,
            expected_path=str(full),
            exists=False,
        )

    def check_mimic_iv(self, subfolder: str = "mimic_iv") -> DatasetAvailability:
        root = self._raw(subfolder)
        tables = [
            self._check_table(root, t)
            for t in MIMIC_IV_REQUIRED_TABLES + MIMIC_IV_OPTIONAL_TABLES
        ]
        return DatasetAvailability(
            dataset_name=f"MIMIC-IV ({subfolder})",
            root_path=str(root),
            root_exists=root.exists(),
            tables=tables,
            notes=(
                "Access: PhysioNet credentialing required for full MIMIC-IV. "
                "Demo (100 patients): https://physionet.org/content/mimic-iv-demo/2.2/"
            ) if not root.exists() or not any(t.exists for t in tables) else "",
        )

    def check_mimic_iv_demo(self) -> DatasetAvailability:
        return self.check_mimic_iv(subfolder="mimic_iv_demo")

    def check_vitaldb(self) -> DatasetAvailability:
        root = self._raw("vitaldb")
        tables = [
            self._check_table(root, "cases.csv"),
            self._check_table(root, "trks.csv"),
        ]
        return DatasetAvailability(
            dataset_name="VitalDB",
            root_path=str(root),
            root_exists=root.exists(),
            tables=tables,
            notes=(
                "VitalDB API: pip install vitaldb. "
                "Surgical ICU cases — potential high-frequency replay source. "
                "Phase 1 does not download VitalDB data automatically."
            ),
        )

    def check_eicu(self) -> DatasetAvailability:
        root = self._raw("eicu")
        tables = [
            self._check_table(root, "patient.csv.gz"),
            self._check_table(root, "vitalPeriodic.csv.gz"),
        ]
        return DatasetAvailability(
            dataset_name="eICU-CRD",
            root_path=str(root),
            root_exists=root.exists(),
            tables=tables,
            notes="eICU requires PhysioNet credentialing. Potential external validation dataset.",
        )

    def check_all(self) -> AvailabilityReport:
        report = AvailabilityReport()
        report.datasets.append(self.check_mimic_iv_demo())
        report.datasets.append(self.check_mimic_iv())
        report.datasets.append(self.check_vitaldb())
        report.datasets.append(self.check_eicu())
        return report


def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"

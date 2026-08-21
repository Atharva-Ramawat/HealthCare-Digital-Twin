"""
tests/unit/test_data_availability.py
Phase 1 — Tests for DataAvailabilityChecker.
"""
import pytest
from pathlib import Path
from src.ingestion.data_availability import (
    DataAvailabilityChecker,
    DataNotAvailableError,
    AvailabilityReport,
    DatasetAvailability,
    TableAvailability,
)


class TestTableAvailability:
    def test_size_human_populated_when_exists(self):
        t = TableAvailability("t", "/p", exists=True, size_bytes=2048)
        assert t.size_human != ""

    def test_size_human_empty_when_not_exists(self):
        t = TableAvailability("t", "/p", exists=False, size_bytes=0)
        assert t.size_human == ""

    def test_kb_range(self):
        t = TableAvailability("t", "/p", exists=True, size_bytes=1500)
        assert "KB" in t.size_human

    def test_mb_range(self):
        t = TableAvailability("t", "/p", exists=True, size_bytes=2_000_000)
        assert "MB" in t.size_human


class TestDatasetAvailability:
    def test_is_available_true_when_at_least_one_table(self):
        ds = DatasetAvailability(
            "ds", "/root", True,
            tables=[
                TableAvailability("a", "/a", True, 100),
                TableAvailability("b", "/b", False),
            ],
        )
        assert ds.is_available

    def test_is_available_false_when_no_tables(self):
        ds = DatasetAvailability("ds", "/root", False, tables=[])
        assert not ds.is_available

    def test_is_available_false_when_all_missing(self):
        ds = DatasetAvailability(
            "ds", "/root", True,
            tables=[TableAvailability("a", "/a", False)],
        )
        assert not ds.is_available

    def test_available_tables_count(self):
        ds = DatasetAvailability(
            "ds", "/root", True,
            tables=[
                TableAvailability("a", "/a", True, 100),
                TableAvailability("b", "/b", False),
                TableAvailability("c", "/c", True, 200),
            ],
        )
        assert len(ds.available_tables) == 2
        assert len(ds.missing_tables) == 1


class TestDataAvailabilityChecker:
    def test_check_all_returns_report(self, tmp_path):
        checker = DataAvailabilityChecker(str(tmp_path))
        report = checker.check_all()
        assert isinstance(report, AvailabilityReport)
        assert len(report.datasets) > 0

    def test_empty_root_all_unavailable(self, tmp_path):
        checker = DataAvailabilityChecker(str(tmp_path))
        report = checker.check_all()
        for ds in report.datasets:
            assert not ds.is_available

    def test_demo_detected_when_files_present(self, tmp_path):
        root = tmp_path / "data" / "raw" / "mimic_iv_demo"
        (root / "hosp").mkdir(parents=True)
        (root / "icu").mkdir(parents=True)
        (root / "hosp" / "patients.csv.gz").write_bytes(b"fake")
        (root / "hosp" / "admissions.csv.gz").write_bytes(b"fake")
        (root / "icu" / "icustays.csv.gz").write_bytes(b"fake")
        checker = DataAvailabilityChecker(str(tmp_path))
        ds = checker.check_mimic_iv_demo()
        assert ds.is_available

    def test_get_returns_correct_dataset(self, tmp_path):
        checker = DataAvailabilityChecker(str(tmp_path))
        report = checker.check_all()
        ds = report.get("MIMIC-IV (mimic_iv_demo)")
        assert ds is not None

    def test_get_returns_none_for_unknown(self, tmp_path):
        checker = DataAvailabilityChecker(str(tmp_path))
        report = checker.check_all()
        assert report.get("NoSuchDataset") is None

    def test_summary_is_non_empty_string(self, tmp_path):
        checker = DataAvailabilityChecker(str(tmp_path))
        report = checker.check_all()
        summary = report.summary()
        assert isinstance(summary, str) and len(summary) > 10
        assert "MIMIC-IV" in summary

    def test_uncompressed_csv_detected(self, tmp_path):
        root = tmp_path / "data" / "raw" / "mimic_iv_demo"
        (root / "hosp").mkdir(parents=True)
        (root / "hosp" / "patients.csv").write_text("subject_id\n1\n")
        checker = DataAvailabilityChecker(str(tmp_path))
        ds = checker.check_mimic_iv_demo()
        t = next((x for x in ds.tables if "patients" in x.name), None)
        assert t is not None and t.exists

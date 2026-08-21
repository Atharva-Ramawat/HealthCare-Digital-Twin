"""
tests/unit/test_mimic_iv_demo_loader.py
Phase 1 — Tests for MIMICIVDemoLoader.

Unit tests (no real data needed) + integration tests against local demo.
Integration tests auto-skip if demo data is absent.
"""
import pytest
from pathlib import Path
from src.ingestion.data_availability import DataNotAvailableError
from src.ingestion.mimic_iv_loader import MIMICIVLoader
from src.ingestion.mimic_iv_demo_loader import MIMICIVDemoLoader

_DEMO_PATH = Path("data/raw/mimic_iv_demo")
_DEMO_AVAILABLE = (_DEMO_PATH / "hosp" / "patients.csv.gz").exists()


class TestMIMICIVDemoLoaderMeta:
    def test_is_subclass_of_loader(self):
        assert issubclass(MIMICIVDemoLoader, MIMICIVLoader)

    def test_dataset_label_contains_demo(self):
        assert "demo" in MIMICIVDemoLoader.DATASET_LABEL.lower()

    def test_patient_count_documented_is_100(self):
        assert MIMICIVDemoLoader.PATIENT_COUNT_DOCUMENTED == 100

    def test_dataset_version_is_22(self):
        assert MIMICIVDemoLoader.DATASET_VERSION == "2.2"

    def test_physionet_url_present(self):
        assert "physionet.org" in MIMICIVDemoLoader.PHYSIONET_URL

    def test_repr_contains_version(self):
        if _DEMO_AVAILABLE:
            loader = MIMICIVDemoLoader()
            assert "2.2" in repr(loader)


class TestMIMICIVDemoLoaderErrors:
    def test_raises_when_demo_path_missing(self, tmp_path):
        with pytest.raises(DataNotAvailableError):
            MIMICIVDemoLoader(data_root=str(tmp_path / "no_demo"))

    def test_custom_root_accepted(self, tmp_path):
        # Creating the directory makes the loader not raise on init
        (tmp_path / "demo_root").mkdir()
        loader = MIMICIVDemoLoader(data_root=str(tmp_path / "demo_root"))
        assert loader.data_root.exists()


@pytest.mark.skipif(not _DEMO_AVAILABLE, reason="MIMIC-IV Demo not downloaded")
class TestMIMICIVDemoLoaderRealData:
    """Integration tests against the actual MIMIC-IV Demo download."""

    def test_loads_patients_returns_100(self):
        loader = MIMICIVDemoLoader()
        df = loader.load_patients()
        assert len(df) == 100
        assert "subject_id" in df.columns
        assert "gender" in df.columns

    def test_loads_icustays(self):
        loader = MIMICIVDemoLoader()
        df = loader.load_icustays()
        assert len(df) > 0
        assert "stay_id" in df.columns
        assert "intime" in df.columns
        assert "outtime" in df.columns

    def test_charttime_preserved_as_string(self):
        """CRITICAL: charttime must NOT be parsed as datetime — Phase 2 concern.
        Pandas 3.x may use StringDtype; key check is NOT datetime64."""
        loader = MIMICIVDemoLoader()
        df = loader.load_chartevents(item_ids=[220045])
        import pandas.api.types as pat
        assert not pat.is_datetime64_any_dtype(df["charttime"]), (
            f"charttime must NOT be datetime64, got {df['charttime'].dtype}"
        )
        assert isinstance(df["charttime"].iloc[0], str)

    def test_heart_rate_present_in_chartevents(self):
        """Item 220045 (Heart Rate) must exist in Demo."""
        loader = MIMICIVDemoLoader()
        df = loader.load_chartevents(item_ids=[220045])
        assert len(df) > 0, "Heart Rate (item 220045) not found in Demo chartevents"
        assert (df["itemid"] == 220045).all()

    def test_subject_id_preserved_in_chartevents(self):
        loader = MIMICIVDemoLoader()
        df = loader.load_chartevents(item_ids=[220045])
        assert "subject_id" in df.columns
        assert df["subject_id"].notna().all()

    def test_stay_id_preserved_in_chartevents(self):
        loader = MIMICIVDemoLoader()
        df = loader.load_chartevents(item_ids=[220045])
        assert "stay_id" in df.columns

    def test_d_items_contains_heart_rate(self):
        loader = MIMICIVDemoLoader()
        d_items = loader.load_d_items()
        match = d_items[d_items["itemid"] == 220045]
        assert len(match) == 1
        assert "Heart Rate" in match.iloc[0]["label"]

    def test_labevents_glucose_available(self):
        loader = MIMICIVDemoLoader()
        df = loader.load_labevents(item_ids=[50931])
        assert len(df) > 0, "Glucose (item 50931) not found in Demo labevents"

    def test_check_availability_all_required_present(self):
        loader = MIMICIVDemoLoader()
        av = loader.check_availability()
        missing = [k for k, v in av.items() if not v]
        assert not missing, f"Required tables missing in Demo: {missing}"

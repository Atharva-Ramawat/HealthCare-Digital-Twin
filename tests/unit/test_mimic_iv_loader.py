"""
tests/unit/test_mimic_iv_loader.py
Phase 1 — Tests for MIMICIVLoader using minimal fake CSV files.
No real MIMIC-IV data required for these unit tests.
"""
import gzip
import pytest
import pandas as pd
from pathlib import Path

from src.ingestion.data_availability import DataNotAvailableError
from src.ingestion.mimic_iv_loader import (
    MIMICIVLoader,
    CHARTEVENTS_VITAL_ITEM_IDS,
    LABEVENTS_ITEM_IDS,
)


def _gz(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "mimic"
    _gz(root / "hosp" / "patients.csv.gz",
        "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod\n"
        "10001,M,65,2109,2008-2010,\n"
        "10002,F,52,2110,2010-2012,2115-03-01\n")
    _gz(root / "hosp" / "admissions.csv.gz",
        "subject_id,hadm_id,admittime,dischtime,deathtime,"
        "admission_type,insurance,ethnicity\n"
        "10001,20001,2109-05-01 08:00:00,2109-05-10 12:00:00,,EMERGENCY,Medicare,WHITE\n"
        "10002,20002,2110-02-15 14:00:00,2110-02-25 09:00:00,2110-02-24 10:00:00,ELECTIVE,Other,BLACK\n")
    _gz(root / "icu" / "icustays.csv.gz",
        "subject_id,hadm_id,stay_id,first_careunit,last_careunit,"
        "intime,outtime,los\n"
        "10001,20001,30001,MICU,MICU,2109-05-02 06:00:00,2109-05-08 10:00:00,6.17\n"
        "10002,20002,30002,SICU,SICU,2110-02-16 09:00:00,2110-02-22 15:00:00,6.25\n")
    _gz(root / "icu" / "chartevents.csv.gz",
        "subject_id,hadm_id,stay_id,itemid,charttime,storetime,"
        "value,valuenum,valueuom,warning\n"
        "10001,20001,30001,220045,2109-05-03 08:00:00,2109-05-03 08:05:00,75,75.0,bpm,0\n"
        "10001,20001,30001,220045,2109-05-03 09:00:00,2109-05-03 09:05:00,80,80.0,bpm,0\n"
        "10001,20001,30001,99999,2109-05-03 08:30:00,2109-05-03 08:35:00,1,1.0,x,0\n"
        "10002,20002,30002,220045,2110-02-17 10:00:00,2110-02-17 10:05:00,90,90.0,bpm,0\n")
    _gz(root / "icu" / "d_items.csv.gz",
        "itemid,label,abbreviation,category,unitname\n"
        "220045,Heart Rate,HR,Routine Vital Signs,bpm\n"
        "99999,Other Item,OI,Other,x\n")
    _gz(root / "hosp" / "labevents.csv.gz",
        "subject_id,hadm_id,itemid,charttime,storetime,"
        "value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag\n"
        "10001,20001,50931,2109-05-03 12:00:00,2109-05-03 12:30:00,120,120.0,mg/dL,,,\n"
        "10001,20001,88888,2109-05-03 12:00:00,2109-05-03 12:30:00,5,5.0,g/dL,,,\n")
    _gz(root / "hosp" / "d_labitems.csv.gz",
        "itemid,label,fluid,category,loinc_code\n"
        "50931,Glucose,Blood,Chemistry,2345-7\n")
    return root


class TestMIMICIVLoaderErrors:
    def test_raises_when_root_missing(self, tmp_path):
        with pytest.raises(DataNotAvailableError):
            MIMICIVLoader(str(tmp_path / "nonexistent"))

    def test_error_contains_physionet_url(self, tmp_path):
        try:
            MIMICIVLoader(str(tmp_path / "nonexistent"))
        except DataNotAvailableError as e:
            assert "physionet.org" in str(e).lower()

    def test_raises_when_table_missing(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        loader = MIMICIVLoader(str(root))
        with pytest.raises(DataNotAvailableError):
            loader.load_patients()

    def test_check_availability_all_false_empty(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        loader = MIMICIVLoader(str(root))
        av = loader.check_availability()
        assert isinstance(av, dict)
        assert not any(av.values())


class TestMIMICIVLoaderFakeData:
    def test_load_patients_returns_dataframe(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_patients()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_patients_preserves_subject_id(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_patients()
        assert "subject_id" in df.columns
        assert set(df["subject_id"]) == {10001, 10002}

    def test_load_admissions_has_hadm_id(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_admissions()
        assert "hadm_id" in df.columns
        assert len(df) == 2

    def test_load_icustays_has_stay_id(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_icustays()
        assert "stay_id" in df.columns
        assert len(df) == 2

    def test_chartevents_filters_item_ids(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_chartevents(item_ids=[220045])
        assert set(df["itemid"].unique()) == {220045}
        assert 99999 not in df["itemid"].values

    def test_chartevents_filters_stay_ids(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_chartevents(stay_ids=[30001], item_ids=[220045])
        assert (df["stay_id"] == 30001).all()
        assert 30002 not in df["stay_id"].values

    def test_charttime_not_parsed_to_datetime(self, fake_root):
        """CRITICAL: charttime must NOT be a datetime dtype — Phase 2 concern.
        Pandas 3.x may use StringDtype or object; both are acceptable.
        Only datetime64 dtype is prohibited."""
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_chartevents(item_ids=[220045])
        import pandas.api.types as pat
        assert not pat.is_datetime64_any_dtype(df["charttime"]), (
            f"charttime must NOT be datetime64, got {df['charttime'].dtype}"
        )
        # Also verify it reads as a string (not numeric)
        assert isinstance(df["charttime"].iloc[0], str), (
            f"charttime values must be strings, got {type(df['charttime'].iloc[0])}"
        )

    def test_labevents_filters_item_ids(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        df = loader.load_labevents(item_ids=[50931])
        assert (df["itemid"] == 50931).all()
        assert 88888 not in df["itemid"].values

    def test_core_tables_returns_all_required_keys(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        tables = loader.load_core_tables()
        for k in ("patients", "admissions", "icustays", "d_items", "d_labitems"):
            assert k in tables
            assert isinstance(tables[k], pd.DataFrame)

    def test_check_availability_detects_present_tables(self, fake_root):
        loader = MIMICIVLoader(str(fake_root))
        av = loader.check_availability()
        assert av["hosp/patients.csv.gz"]
        assert av["hosp/admissions.csv.gz"]
        assert av["icu/icustays.csv.gz"]
        assert av["icu/chartevents.csv.gz"]

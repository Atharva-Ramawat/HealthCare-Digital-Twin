"""
tests/unit/test_synthetic_generator.py
Phase 1 — Tests for SyntheticClinicalGenerator.
Replaces Phase 0 placeholder.
"""
import warnings
import pytest
import numpy as np
import pandas as pd

from src.ingestion.synthetic_generator import (
    SyntheticClinicalGenerator,
    VALID_TRAJECTORIES,
    SYNTHETIC_ITEM_IDS,
    _VITAL_SPECS,
    _DISCLAIMER,
)

EXPECTED_COLS = {
    "subject_id", "hadm_id", "stay_id", "itemid",
    "charttime", "value", "valuenum", "valueuom",
    "canonical_name", "data_source", "trajectory",
}


@pytest.fixture
def gen():
    return SyntheticClinicalGenerator(seed=42)


@pytest.mark.parametrize("traj", sorted(VALID_TRAJECTORIES))
def test_all_trajectories_return_dataframe(gen, traj):
    df = gen.generate_patient_series(trajectory=traj, num_steps=20)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.parametrize("traj", sorted(VALID_TRAJECTORIES))
def test_schema_columns_present(gen, traj):
    df = gen.generate_patient_series(trajectory=traj, num_steps=10)
    missing = EXPECTED_COLS - set(df.columns)
    assert not missing, f"Missing columns for '{traj}': {missing}"


@pytest.mark.parametrize("traj", sorted(VALID_TRAJECTORIES))
def test_synthetic_label_in_data_source(gen, traj):
    df = gen.generate_patient_series(trajectory=traj, num_steps=10)
    assert all("SYNTHETIC" in str(v) for v in df["data_source"]), (
        f"All rows must contain 'SYNTHETIC' in data_source"
    )


def test_identifiers_preserved(gen):
    df = gen.generate_patient_series(
        patient_id=99, hadm_id=88, stay_id=77,
        trajectory="normal", num_steps=10,
    )
    assert (df["subject_id"] == 99).all()
    assert (df["hadm_id"] == 88).all()
    assert (df["stay_id"] == 77).all()


def test_item_ids_are_mimic_compatible(gen):
    df = gen.generate_patient_series(trajectory="normal", num_steps=10)
    assert set(df["itemid"].unique()) <= set(SYNTHETIC_ITEM_IDS.values())


def test_charttime_is_timestamp(gen):
    df = gen.generate_patient_series(trajectory="normal", num_steps=10)
    assert isinstance(df["charttime"].iloc[0], pd.Timestamp)


def test_missingness_stress_has_high_missing(gen):
    df = gen.generate_patient_series(
        trajectory="missingness_stress", num_steps=300
    )
    frac = df["valuenum"].isna().mean()
    assert frac >= 0.50, f"Expected >=50% missing, got {frac:.1%}"


def test_values_in_physiological_range(gen):
    df = gen.generate_patient_series(trajectory="normal", num_steps=100)
    for vital, spec in _VITAL_SPECS.items():
        sub = df[df["canonical_name"] == vital]["valuenum"].dropna()
        if sub.empty:
            continue
        assert (sub >= spec["lo"]).all(), f"{vital} has values below lo={spec['lo']}"
        assert (sub <= spec["hi"]).all(), f"{vital} has values above hi={spec['hi']}"


def test_invalid_trajectory_raises(gen):
    with pytest.raises(ValueError, match="Unknown trajectory"):
        gen.generate_patient_series(trajectory="cardiac_arrest", num_steps=5)


def test_invalid_trajectory_septic_shock_raises(gen):
    with pytest.raises(ValueError, match="Unknown trajectory"):
        gen.generate_patient_series(trajectory="septic_shock", num_steps=5)


def test_reproducibility_same_seed():
    g1 = SyntheticClinicalGenerator(seed=42)
    g2 = SyntheticClinicalGenerator(seed=42)
    df1 = g1.generate_patient_series(trajectory="normal", num_steps=20)
    df2 = g2.generate_patient_series(trajectory="normal", num_steps=20)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    g1 = SyntheticClinicalGenerator(seed=1)
    g2 = SyntheticClinicalGenerator(seed=2)
    df1 = g1.generate_patient_series(trajectory="normal", num_steps=50)
    df2 = g2.generate_patient_series(trajectory="normal", num_steps=50)
    assert not df1["valuenum"].equals(df2["valuenum"])


def test_ward_dataset_multi_patient(gen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = gen.generate_ward_dataset(n_patients=4, num_steps=10)
    assert df["subject_id"].nunique() == 4


def test_ward_dataset_carries_disclaimer(gen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = gen.generate_ward_dataset(n_patients=2, num_steps=10)
    assert all("SYNTHETIC" in str(v) for v in df["data_source"])


def test_ward_dataset_emits_user_warning(gen):
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        gen.generate_ward_dataset(n_patients=2, num_steps=5)


def test_synthetic_start_time_is_future(gen):
    """Start time should default to 2100 to be visually distinct from MIMIC data."""
    df = gen.generate_patient_series(trajectory="normal", num_steps=5)
    first_time = df["charttime"].iloc[0]
    assert first_time.year >= 2100, (
        f"Synthetic timestamps should default to year>=2100, got {first_time.year}"
    )

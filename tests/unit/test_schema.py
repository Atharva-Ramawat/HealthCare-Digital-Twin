"""
tests/unit/test_schema.py
Phase 1 — Tests for canonical variable schema.
"""
import pytest
from src.ingestion.schema import (
    ALL_VARIABLE_SPECS,
    VARIABLE_REGISTRY,
    CORE_VITAL_CANDIDATES,
    LAB_CANDIDATES,
    DEMOGRAPHIC_CANDIDATES,
    IDENTIFIERS,
    CHARTEVENTS_VITAL_ITEM_IDS,
    LABEVENTS_ITEM_IDS,
    VariableCategory,
    DataAvailabilityStatus,
    VariableSpec,
)


class TestVariableRegistry:
    def test_all_variable_specs_non_empty(self):
        assert len(ALL_VARIABLE_SPECS) > 0

    def test_registry_keys_match_canonical_names(self):
        for name, spec in VARIABLE_REGISTRY.items():
            assert name == spec.canonical_name

    def test_no_duplicate_canonical_names(self):
        names = [s.canonical_name for s in ALL_VARIABLE_SPECS]
        assert len(names) == len(set(names)), "Duplicate canonical names found"

    def test_core_vitals_non_empty(self):
        assert len(CORE_VITAL_CANDIDATES) > 0

    def test_lab_candidates_non_empty(self):
        assert len(LAB_CANDIDATES) > 0

    def test_demographic_candidates_non_empty(self):
        assert len(DEMOGRAPHIC_CANDIDATES) > 0

    def test_identifiers_non_empty(self):
        assert len(IDENTIFIERS) > 0

    def test_all_specs_have_required_fields(self):
        for spec in ALL_VARIABLE_SPECS:
            assert spec.canonical_name, f"Missing canonical_name"
            assert spec.category in VariableCategory.__members__.values()
            assert spec.source_table, f"Missing source_table for {spec.canonical_name}"
            assert spec.unit, f"Missing unit for {spec.canonical_name}"

    def test_chartevents_item_ids_are_ints(self):
        assert all(isinstance(i, int) for i in CHARTEVENTS_VITAL_ITEM_IDS)

    def test_labevents_item_ids_are_ints(self):
        assert all(isinstance(i, int) for i in LABEVENTS_ITEM_IDS)

    def test_heart_rate_item_id(self):
        assert "heart_rate" in VARIABLE_REGISTRY
        hr = VARIABLE_REGISTRY["heart_rate"]
        assert hr.item_id == 220045
        assert hr.category == VariableCategory.CORE_VITAL

    def test_lactate_item_id(self):
        assert "lactate" in VARIABLE_REGISTRY
        lact = VARIABLE_REGISTRY["lactate"]
        assert lact.item_id == 50813
        assert lact.category == VariableCategory.LAB_CANDIDATE

    def test_all_statuses_are_valid_enum(self):
        valid = set(DataAvailabilityStatus.__members__.values())
        for spec in ALL_VARIABLE_SPECS:
            assert spec.availability_status in valid

    def test_chartevents_ids_non_empty(self):
        assert len(CHARTEVENTS_VITAL_ITEM_IDS) >= 10

    def test_labevents_ids_non_empty(self):
        assert len(LABEVENTS_ITEM_IDS) >= 4

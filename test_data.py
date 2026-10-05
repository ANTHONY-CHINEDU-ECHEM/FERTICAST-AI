import numpy as np
import pandas as pd

from ferticast.data.cleaning import clean_register, temporal_split
from ferticast.data.schema import INFERTILITY_CAUSES, NUMERIC_SPECS, age_band
from ferticast.data.simulate import simulate_register


def test_simulation_is_deterministic():
    a = simulate_register(500, seed=3)
    b = simulate_register(500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_simulated_success_declines_with_age():
    clean = simulate_register(40_000, seed=5, messy=False)
    young = clean.loc[clean.female_age < 35, "live_birth"].mean()
    older = clean.loc[clean.female_age >= 40, "live_birth"].mean()
    assert young > 2 * older


def test_cleaning_removes_duplicates_and_harmonises_units(raw_register, clean):
    assert clean["cycle_id"].is_unique
    assert "amh_value" not in clean and "amh_unit" not in clean
    # After conversion no AMH value should look like a pmol/L reading for a typical patient.
    assert clean["amh_ng_ml"].median() < 5
    assert set(clean["infertility_cause"]) <= set(INFERTILITY_CAUSES)


def test_cleaning_blanks_out_of_range_values(clean):
    for name, spec in NUMERIC_SPECS.items():
        values = clean[name].dropna()
        assert values.between(spec.low, spec.high).all(), name


def test_cleaning_audit_counts_pmol_conversions():
    raw = simulate_register(3000, n_clinics=10, seed=2)
    _, audit = clean_register(raw)
    assert audit["amh_values_converted_from_pmol"] > 0
    assert audit["duplicate_rows_removed"] == len(raw) - raw["cycle_id"].nunique()


def test_temporal_split_has_no_leakage(clean):
    splits = temporal_split(clean, 2019, [2020, 2021], [2022, 2023])
    assert splits["train"]["treatment_year"].max() <= 2019
    assert splits["test"]["treatment_year"].min() >= 2022
    assert sum(len(s) for s in splits.values()) == len(clean)


def test_age_band_edges():
    assert age_band(34) == "18 to 34"
    assert age_band(35) == "35 to 37"
    assert age_band(47) == "43 and over"
    assert age_band(np.float64(40.0)) == "40 to 42"

"""Register cleaning: unit harmonisation, range checks and temporal splitting."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ferticast.data.schema import AMH_PMOL_PER_NG, INFERTILITY_CAUSES, NUMERIC_SPECS, RAW_FEATURES, TARGET

logger = logging.getLogger(__name__)


def clean_register(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a raw register and return it with an audit of every correction.

    Rules applied, in order:

    1. Remove duplicate cycle submissions.
    2. Harmonise AMH to ng/mL using the reporting unit of each record.
    3. Normalise category spellings to the canonical vocabulary.
    4. Replace values outside the plausible clinical range with missing, so
       that placeholder codes such as 999 never reach the model.
    """
    audit: dict[str, int] = {}
    df = raw.copy()

    before = len(df)
    df = df.drop_duplicates(subset="cycle_id") if "cycle_id" in df else df.drop_duplicates()
    audit["duplicate_rows_removed"] = before - len(df)

    if "amh_value" in df:
        is_pmol = (df["amh_unit"] == "pmol/L") if "amh_unit" in df else pd.Series(False, index=df.index)
        audit["amh_values_converted_from_pmol"] = int((is_pmol & df["amh_value"].notna()).sum())
        df["amh_ng_ml"] = np.where(is_pmol, df["amh_value"] / AMH_PMOL_PER_NG, df["amh_value"])
        df = df.drop(columns=["amh_value", "amh_unit"], errors="ignore")

    original = df["infertility_cause"].astype(str)
    cause = original.str.lower().str.strip().str.replace(" ", "_")
    audit["cause_labels_normalised"] = int((cause != original).sum())
    df["infertility_cause"] = cause.where(cause.isin(INFERTILITY_CAUSES), "unexplained")
    df["sperm_source"] = df["sperm_source"].astype(str).str.lower().str.strip()

    for name, spec in NUMERIC_SPECS.items():
        if name not in df:
            df[name] = np.nan
        values = pd.to_numeric(df[name], errors="coerce")
        invalid = values.notna() & ((values < spec.low) | (values > spec.high))
        if invalid.any():
            audit[f"out_of_range_{name}"] = int(invalid.sum())
        df[name] = values.mask(invalid)

    df["smoker"] = pd.to_numeric(df["smoker"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["female_age"]).reset_index(drop=True)
    logger.info("Cleaning audit: %s", audit)
    return df, audit


def temporal_split(df: pd.DataFrame, train_until_year: int, calibration_years: list[int],
                   test_years: list[int]) -> dict[str, pd.DataFrame]:
    """Split by treatment year so evaluation mimics prospective deployment."""
    return {
        "train": df[df["treatment_year"] <= train_until_year].reset_index(drop=True),
        "calibration": df[df["treatment_year"].isin(calibration_years)].reset_index(drop=True),
        "test": df[df["treatment_year"].isin(test_years)].reset_index(drop=True),
    }


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the raw model inputs and the outcome."""
    return df[RAW_FEATURES].copy(), df[TARGET].astype(int)

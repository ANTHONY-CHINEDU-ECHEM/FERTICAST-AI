"""Adapter for the public HFEA anonymised register.

The HFEA publishes an anonymised register of UK fertility treatment. It is a
valuable real world source for outcome rates, but it reports age in bands and
does not include AMH, FSH, AFC, BMI or semen parameters. This adapter maps the
fields that do exist onto the FertiCast schema so the same pipeline can be run
on real outcomes. Fields that the register does not carry are left missing and
are then handled by the domain imputer, which falls back to age conditioned
estimates. Column names differ slightly between HFEA releases, so the mapping
is passed in rather than hard coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ferticast.data.schema import RAW_FEATURES, TARGET

AGE_BAND_MIDPOINTS = {"18-34": 31, "35-37": 36, "38-39": 38.5, "40-42": 41, "43-44": 43.5, "45-50": 47}

DEFAULT_COLUMN_MAP = {
    "age": "Patient Age at Treatment",
    "partner_age": "Partner Age",
    "previous_cycles": "Total Number of Previous IVF cycles",
    "previous_births": "Total number of previous live births",
    "sperm_source": "Sperm From",
    "live_birth": "Live Birth Occurrence",
    "year": "Year of Treatment",
    "cause_male": "Cause of Infertility - Male Factor",
    "cause_tubal": "Cause of Infertility - Tubal disease",
    "cause_ovulatory": "Cause of Infertility - Ovulatory Disorder",
    "cause_endometriosis": "Cause of Infertility - Endometriosis",
    "cause_unexplained": "Cause of Infertility - Patient Unexplained",
}


def _first_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _flag(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw:
        return pd.Series(0, index=raw.index)
    return pd.to_numeric(raw[column], errors="coerce").fillna(0).astype(int)


def load_hfea_register(path: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Read an HFEA register CSV and return a frame in the FertiCast schema."""
    cols = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=raw.index)
    out["cycle_id"] = np.arange(1, len(raw) + 1)
    out["clinic_id"] = 0
    out["treatment_year"] = _first_number(raw[cols["year"]]) if cols["year"] in raw else 0
    out["female_age"] = raw[cols["age"]].astype(str).str.strip().map(AGE_BAND_MIDPOINTS)
    out["male_age"] = (raw[cols["partner_age"]].astype(str).str.strip().map(AGE_BAND_MIDPOINTS)
                       if cols["partner_age"] in raw else np.nan)
    out["previous_ivf_cycles"] = _first_number(raw[cols["previous_cycles"]]).fillna(0)
    out["previous_live_births"] = (_first_number(raw[cols["previous_births"]]).fillna(0)
                                   if cols["previous_births"] in raw else 0)

    male, tubal = _flag(raw, cols["cause_male"]), _flag(raw, cols["cause_tubal"])
    ovulatory, endo = _flag(raw, cols["cause_ovulatory"]), _flag(raw, cols["cause_endometriosis"])
    out["infertility_cause"] = np.select(
        [(male + tubal + ovulatory + endo) > 1, male == 1, tubal == 1, ovulatory == 1, endo == 1],
        ["combined", "male_factor", "tubal", "ovulatory_pcos", "endometriosis"],
        default="unexplained",
    )
    out["sperm_source"] = np.where(raw[cols["sperm_source"]].astype(str).str.lower().str.contains("donor"), "donor", "partner")
    out["smoker"] = 0
    out[TARGET] = pd.to_numeric(raw[cols["live_birth"]], errors="coerce").fillna(0).clip(0, 1).astype(int)
    for name in RAW_FEATURES:
        if name not in out:
            out[name] = np.nan  # Not captured by the register; handled by the imputer.
    return out.dropna(subset=["female_age"]).reset_index(drop=True)

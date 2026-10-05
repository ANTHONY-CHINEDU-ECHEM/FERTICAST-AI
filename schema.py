"""Single source of truth for the clinical feature schema.

Every component (simulator, cleaner, imputer, API, Streamlit form) reads the
plausible ranges and category lists from here, so validation rules cannot drift
apart between training and serving.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET = "live_birth"
ID_COLUMNS = ["cycle_id", "clinic_id", "treatment_year"]

INFERTILITY_CAUSES = [
    "unexplained",
    "male_factor",
    "tubal",
    "ovulatory_pcos",
    "endometriosis",
    "diminished_reserve",
    "combined",
]
SPERM_SOURCES = ["partner", "donor"]

AMH_PMOL_PER_NG = 7.14  # 1 ng/mL of AMH equals 7.14 pmol/L


@dataclass(frozen=True)
class NumericSpec:
    """Plausible clinical range and display metadata for a numeric input."""

    name: str
    label: str
    unit: str
    low: float
    high: float
    default: float
    step: float = 1.0


NUMERIC_SPECS: dict[str, NumericSpec] = {
    spec.name: spec
    for spec in [
        NumericSpec("female_age", "Female age", "years", 18, 50, 34, 1),
        NumericSpec("male_age", "Male partner age", "years", 18, 70, 36, 1),
        NumericSpec("bmi", "Body mass index", "kg/m2", 15, 55, 24.5, 0.1),
        NumericSpec("amh_ng_ml", "Anti Mullerian hormone", "ng/mL", 0.01, 25, 2.4, 0.1),
        NumericSpec("fsh_iu_l", "Basal FSH", "IU/L", 0.5, 40, 6.8, 0.1),
        NumericSpec("afc", "Antral follicle count", "follicles", 0, 60, 12, 1),
        NumericSpec("infertility_duration_years", "Duration of infertility", "years", 0, 20, 2.5, 0.5),
        NumericSpec("sperm_concentration_m_ml", "Sperm concentration", "million/mL", 0, 300, 55, 1),
        NumericSpec("sperm_progressive_motility_pct", "Progressive motility", "%", 0, 100, 50, 1),
        NumericSpec("previous_ivf_cycles", "Previous IVF cycles", "cycles", 0, 10, 0, 1),
        NumericSpec("previous_live_births", "Previous live births", "births", 0, 6, 0, 1),
    ]
}

RAW_NUMERIC = list(NUMERIC_SPECS)
RAW_CATEGORICAL = ["infertility_cause", "sperm_source"]
RAW_BINARY = ["smoker"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL + RAW_BINARY

# Columns that may legitimately be missing and receive a missingness indicator.
IMPUTED_COLUMNS = [
    "amh_ng_ml",
    "fsh_iu_l",
    "afc",
    "bmi",
    "infertility_duration_years",
    "sperm_concentration_m_ml",
    "sperm_progressive_motility_pct",
    "male_age",
]

AGE_BANDS = [(18, 34, "18 to 34"), (35, 37, "35 to 37"), (38, 39, "38 to 39"), (40, 42, "40 to 42"), (43, 50, "43 and over")]


def age_band(age: float) -> str:
    """Return the regulator style age band label for a female age."""
    for low, high, label in AGE_BANDS:
        if low <= age <= high:
            return label
    return AGE_BANDS[0][2] if age < AGE_BANDS[0][0] else AGE_BANDS[-1][2]

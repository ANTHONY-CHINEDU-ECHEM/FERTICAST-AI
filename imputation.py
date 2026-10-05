"""Domain informed imputation for baseline fertility work up data.

Generic imputers treat every gap the same way. Clinical gaps are not alike:

* A missing semen analysis in a donor sperm cycle is *structural*. The value
  does not exist, so it is replaced by a screened donor reference profile
  rather than by a population average that includes male factor patients.
* AMH and FSH are two views of the same ovarian reserve. When one is missing
  it is predicted from the other and from age, using relationships learned on
  complete training records, instead of a flat median.
* Antral follicle count follows AMH and age.
* BMI is conditioned on diagnosis because ovulatory disorder patients carry a
  different BMI distribution.

Every imputed column also receives a missingness indicator, because the fact
that a test was not ordered is itself informative (for example, AMH was not
routine in earlier treatment years).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from ferticast.data.schema import IMPUTED_COLUMNS, NUMERIC_SPECS, RAW_FEATURES, age_band

DONOR_REFERENCE_AGE = 30.0


class DomainImputer(BaseEstimator, TransformerMixin):
    """Impute clinical work up variables using reproductive medicine logic."""

    def __init__(self, add_indicators: bool = True):
        self.add_indicators = add_indicators

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None):
        X = X[RAW_FEATURES]
        age = X["female_age"]
        log_amh, log_fsh = np.log(X["amh_ng_ml"]), np.log(X["fsh_iu_l"])

        both = log_amh.notna() & log_fsh.notna()
        self.amh_from_age_fsh_ = LinearRegression().fit(np.c_[age[both], log_fsh[both]], log_amh[both])
        has_amh = log_amh.notna()
        self.amh_from_age_ = LinearRegression().fit(age[has_amh].to_frame(), log_amh[has_amh])
        self.fsh_from_age_amh_ = LinearRegression().fit(np.c_[age[both], log_amh[both]], log_fsh[both])
        afc_ok = has_amh & X["afc"].notna()
        self.afc_from_age_amh_ = LinearRegression().fit(np.c_[age[afc_ok], log_amh[afc_ok]], np.log1p(X.loc[afc_ok, "afc"]))

        self.bmi_by_cause_ = X.groupby("infertility_cause")["bmi"].median().to_dict()
        self.bmi_global_ = float(X["bmi"].median())
        bands = age.map(age_band)
        self.duration_by_band_ = X.groupby(bands)["infertility_duration_years"].median().to_dict()
        self.duration_global_ = float(X["infertility_duration_years"].median())

        partner = X["sperm_source"] == "partner"
        male_factor = X["infertility_cause"].isin(["male_factor", "combined"])
        sperm_cols = ["sperm_concentration_m_ml", "sperm_progressive_motility_pct"]
        self.sperm_normal_ = X.loc[partner & ~male_factor, sperm_cols].median().to_dict()
        self.sperm_male_factor_ = X.loc[partner & male_factor, sperm_cols].median().to_dict()
        # Donors are screened, so the reference profile is the upper quartile of normal partners.
        self.sperm_donor_ = X.loc[partner & ~male_factor, sperm_cols].quantile(0.75).to_dict()
        self.age_gap_ = float((X["male_age"] - age).median())
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[RAW_FEATURES].copy()
        for col in NUMERIC_SPECS:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        indicators = {f"{col}_missing": X[col].isna().astype(int) for col in IMPUTED_COLUMNS}
        age = X["female_age"]
        donor = X["sperm_source"] == "donor"
        male_factor = X["infertility_cause"].isin(["male_factor", "combined"])

        # Ovarian reserve block: AMH first (using FSH when available), then FSH, then AFC.
        amh_gap, fsh_known = X["amh_ng_ml"].isna(), X["fsh_iu_l"].notna()
        use_fsh = amh_gap & fsh_known
        if use_fsh.any():
            pred = self.amh_from_age_fsh_.predict(np.c_[age[use_fsh], np.log(X.loc[use_fsh, "fsh_iu_l"])])
            X.loc[use_fsh, "amh_ng_ml"] = np.exp(pred)
        age_only = X["amh_ng_ml"].isna()
        if age_only.any():
            X.loc[age_only, "amh_ng_ml"] = np.exp(self.amh_from_age_.predict(age[age_only].to_frame()))
        X["amh_ng_ml"] = X["amh_ng_ml"].clip(NUMERIC_SPECS["amh_ng_ml"].low, NUMERIC_SPECS["amh_ng_ml"].high)

        fsh_gap = X["fsh_iu_l"].isna()
        if fsh_gap.any():
            pred = self.fsh_from_age_amh_.predict(np.c_[age[fsh_gap], np.log(X.loc[fsh_gap, "amh_ng_ml"])])
            X.loc[fsh_gap, "fsh_iu_l"] = np.exp(pred)
        afc_gap = X["afc"].isna()
        if afc_gap.any():
            pred = self.afc_from_age_amh_.predict(np.c_[age[afc_gap], np.log(X.loc[afc_gap, "amh_ng_ml"])])
            X.loc[afc_gap, "afc"] = np.round(np.expm1(pred)).clip(0, None)

        X["bmi"] = X["bmi"].fillna(X["infertility_cause"].map(self.bmi_by_cause_)).fillna(self.bmi_global_)
        X["infertility_duration_years"] = (
            X["infertility_duration_years"].fillna(age.map(age_band).map(self.duration_by_band_)).fillna(self.duration_global_)
        )

        # Semen block: donor reference, then diagnosis specific partner medians.
        for col in ["sperm_concentration_m_ml", "sperm_progressive_motility_pct"]:
            gap = X[col].isna()
            X.loc[gap & donor, col] = self.sperm_donor_[col]
            X.loc[gap & ~donor & male_factor, col] = self.sperm_male_factor_[col]
            X.loc[gap & ~donor & ~male_factor, col] = self.sperm_normal_[col]
        male_gap = X["male_age"].isna()
        X.loc[male_gap & donor, "male_age"] = DONOR_REFERENCE_AGE
        X.loc[male_gap & ~donor, "male_age"] = age[male_gap & ~donor] + self.age_gap_

        for col in ["previous_ivf_cycles", "previous_live_births"]:
            X[col] = X[col].fillna(0)
        if self.add_indicators:
            for name, values in indicators.items():
                X[name] = values.to_numpy()
        return X

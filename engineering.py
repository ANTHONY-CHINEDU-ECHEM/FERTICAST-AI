"""Feature engineering on top of the imputed clinical record."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ferticast.data.schema import IMPUTED_COLUMNS, INFERTILITY_CAUSES, NUMERIC_SPECS

# Human readable labels and the display group each model column belongs to.
# SHAP contributions are summed within a group before they are shown to a
# patient, so that one clinical concept appears as one bar.
DISPLAY_GROUPS = {name: spec.label for name, spec in NUMERIC_SPECS.items()}
DISPLAY_GROUPS.update({
    "total_motile_concentration": "Sperm quality (combined)",
    "low_ovarian_reserve": "Ovarian reserve markers",
    "donor_sperm": "Sperm source",
    "smoker": "Smoking",
})
DISPLAY_GROUPS.update({f"cause_{c}": "Cause of infertility" for c in INFERTILITY_CAUSES})
DISPLAY_GROUPS.update({f"{c}_missing": DISPLAY_GROUPS[c] for c in IMPUTED_COLUMNS})
DISPLAY_GROUPS["sperm_concentration_m_ml"] = "Sperm quality (combined)"
DISPLAY_GROUPS["sperm_progressive_motility_pct"] = "Sperm quality (combined)"
DISPLAY_GROUPS["sperm_concentration_m_ml_missing"] = "Sperm quality (combined)"
DISPLAY_GROUPS["sperm_progressive_motility_pct_missing"] = "Sperm quality (combined)"

MODIFIABLE_GROUPS = {"Body mass index", "Smoking"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive clinically meaningful features and encode categoricals.

    Produces a purely numeric frame with a stable column order so that tree
    models, SHAP explainers and the serving layer always agree on the layout.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = list(self._build(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build(X).reindex(columns=self.feature_names_, fill_value=0)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    @staticmethod
    def _build(X: pd.DataFrame) -> pd.DataFrame:
        out = X[list(NUMERIC_SPECS)].astype(float).copy()
        out["total_motile_concentration"] = (
            out["sperm_concentration_m_ml"] * out["sperm_progressive_motility_pct"] / 100.0
        )
        # Bologna style poor reserve flag: low AMH or low antral follicle count.
        out["low_ovarian_reserve"] = ((out["amh_ng_ml"] < 1.1) | (out["afc"] < 5)).astype(int)
        out["donor_sperm"] = (X["sperm_source"] == "donor").astype(int)
        out["smoker"] = X["smoker"].astype(int)
        for cause in INFERTILITY_CAUSES:
            out[f"cause_{cause}"] = (X["infertility_cause"] == cause).astype(int)
        for col in X.columns:
            if col.endswith("_missing"):
                out[col] = X[col].astype(int)
        return out

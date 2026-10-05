"""SHAP explanations for the calibrated ensemble.

The ensemble averages the learners in log odds space, so its SHAP values are
exactly the weighted average of each learner's TreeSHAP values. Contributions
are then grouped into clinical concepts (for example all cause of infertility
indicator columns become one factor) and converted from log odds into
percentage points through the calibrator, so that the waterfall a patient sees
ends at precisely the probability the model reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from ferticast.features.engineering import DISPLAY_GROUPS


class EnsembleExplainer:
    """TreeSHAP for a weighted log odds ensemble of gradient boosted trees."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.explainers = {name: shap.TreeExplainer(model) for name, model in bundle.models.items()
                           if bundle.weights.get(name, 0) > 0}

    @staticmethod
    def _as_matrix(values) -> np.ndarray:
        if isinstance(values, list):  # Older LightGBM API returns one array per class.
            values = values[-1]
        values = np.asarray(values)
        return values[..., -1] if values.ndim == 3 else values

    @staticmethod
    def _as_scalar(expected) -> float:
        return float(np.atleast_1d(expected)[-1])

    def shap_values(self, X_raw: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
        """Return (values in log odds, base value, engineered feature frame)."""
        X = self.bundle.preprocessor.transform(X_raw)
        values, base = np.zeros(X.shape), 0.0
        for name, explainer in self.explainers.items():
            weight = self.bundle.weights[name]
            values += weight * self._as_matrix(explainer.shap_values(X))
            base += weight * self._as_scalar(explainer.expected_value)
        return values, base, X

    def grouped(self, X_raw: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP values summed into clinical concept groups, one row per patient."""
        values, base, X = self.shap_values(X_raw)
        frame = pd.DataFrame(values, columns=X.columns)
        groups = pd.Series({c: DISPLAY_GROUPS.get(c, c) for c in X.columns})
        return frame.T.groupby(groups).sum().T, base

    def probability_waterfall(self, record_raw: pd.DataFrame) -> dict:
        """Sequential waterfall in probability space for a single patient.

        Factors are applied from the largest to the smallest absolute log odds
        contribution. Each step is the change in calibrated probability when
        that factor is added, so the steps sum exactly to the difference
        between the population baseline and the patient's prediction.
        """
        grouped, base = self.grouped(record_raw)
        contributions = grouped.iloc[0].sort_values(key=np.abs, ascending=False)
        calibrate = self.bundle.calibrator.predict
        running = base
        previous = float(calibrate(np.array([running]))[0])
        baseline_probability, steps = previous, []
        for factor, value in contributions.items():
            running += value
            current = float(calibrate(np.array([running]))[0])
            steps.append({"factor": factor, "log_odds": float(value), "effect": current - previous})
            previous = current
        return {"baseline_probability": baseline_probability, "probability": previous, "steps": steps}

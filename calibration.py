"""Probability calibration on top of raw model log odds."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class MarginCalibrator:
    """Map uncalibrated log odds to calibrated probabilities.

    ``method`` is one of ``none``, ``platt`` (logistic recalibration of slope
    and intercept) or ``isotonic`` (non parametric, monotone).
    """

    def __init__(self, method: str = "platt"):
        if method not in {"none", "platt", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.model_ = None

    def fit(self, margins: np.ndarray, y: np.ndarray) -> "MarginCalibrator":
        margins = np.asarray(margins, dtype=float)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6).fit(margins.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(margins, y)
        return self

    def predict(self, margins: np.ndarray) -> np.ndarray:
        margins = np.asarray(margins, dtype=float)
        if self.method == "platt":
            return self.model_.predict_proba(margins.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model_.predict(margins)
        return sigmoid(margins)

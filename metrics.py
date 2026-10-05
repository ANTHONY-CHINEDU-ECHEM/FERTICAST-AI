"""Discrimination, calibration and clinical utility metrics."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal frequency bins."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    error = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            error += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(error)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the logistic recalibration model (ideal: 1 and 0)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6).fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Headline metric set used across training, selection and reporting."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, rounds: int = 200, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            values.append(metric(y[idx], p[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> dict[str, list[float]]:
    """Decision curve analysis: net benefit of acting on the model at each threshold."""
    y, p = np.asarray(y), np.asarray(p)
    n, prevalence = len(y), y.mean()
    model, treat_all = [], []
    for t in thresholds:
        positive = p >= t
        tp, fp = (positive & (y == 1)).sum(), (positive & (y == 0)).sum()
        weight = t / (1 - t)
        model.append(float(tp / n - fp / n * weight))
        treat_all.append(float(prevalence - (1 - prevalence) * weight))
    return {"thresholds": list(thresholds), "model": model, "treat_all": treat_all}

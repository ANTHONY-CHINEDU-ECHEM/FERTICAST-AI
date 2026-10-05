"""End to end training: tuning, ensembling, calibration and honest evaluation."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ferticast import __version__
from ferticast.config import Config, resolve
from ferticast.data.cleaning import clean_register, features_and_target, temporal_split
from ferticast.data.schema import AGE_BANDS, RAW_FEATURES, TARGET, age_band
from ferticast.features.engineering import FeatureEngineer
from ferticast.features.imputation import DomainImputer
from ferticast.models.calibration import MarginCalibrator, sigmoid
from ferticast.models.metrics import bootstrap_ci, classification_metrics

logger = logging.getLogger(__name__)

XGB_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_child_weight": [5, 20, 50, 100],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
    "gamma": [0.0, 0.5, 2.0],
}
LGBM_SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_child_samples": [50, 100, 200, 400],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
}


def build_preprocessor() -> Pipeline:
    """Imputation followed by feature engineering, fitted on training data only."""
    return Pipeline([("impute", DomainImputer()), ("engineer", FeatureEngineer())])


def make_model(kind: str, params: dict, n_estimators: int, seed: int, early_stopping: int | None = None):
    """Factory for the two gradient boosting learners."""
    if kind == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators, tree_method="hist", eval_metric="logloss", random_state=seed,
            n_jobs=0, early_stopping_rounds=early_stopping, **params,
        )
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, random_state=seed, subsample_freq=1, verbose=-1, n_jobs=-1, **params,
    )


def raw_margin(model, X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated log odds from either learner."""
    if isinstance(model, xgb.XGBClassifier):
        return model.predict(X, output_margin=True)
    return model.predict(X, raw_score=True)


def _fit(kind: str, model, X, y, eval_set=None, early_stopping: int | None = None):
    if kind == "xgboost":
        return model.fit(X, y, eval_set=eval_set, verbose=False) if eval_set else model.fit(X, y, verbose=False)
    callbacks = [lgb.early_stopping(early_stopping, verbose=False)] if eval_set and early_stopping else None
    return model.fit(X, y, eval_set=eval_set, callbacks=callbacks)


def random_search(kind: str, X: pd.DataFrame, y: pd.Series, cfg: Config, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Cross validated random search, selecting on log loss (a proper scoring rule)."""
    space = XGB_SPACE if kind == "xgboost" else LGBM_SPACE
    size = min(len(X), cfg.training.search_sample_size)
    idx = rng.choice(len(X), size=size, replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    folds = StratifiedKFold(cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    for trial in range(cfg.training.search_iterations):
        params = {name: rng.choice(values).item() for name, values in space.items()}
        aucs, losses = [], []
        for train_idx, valid_idx in folds.split(Xs, ys):
            model = make_model(kind, params, 250, cfg.seed)
            _fit(kind, model, Xs.iloc[train_idx], ys.iloc[train_idx])
            p = sigmoid(raw_margin(model, Xs.iloc[valid_idx]))
            aucs.append(roc_auc_score(ys.iloc[valid_idx], p))
            losses.append(log_loss(ys.iloc[valid_idx], p))
        rows.append({"model": kind, "trial": trial, **params, "cv_auc": np.mean(aucs), "cv_log_loss": np.mean(losses)})
        logger.info("%s trial %d: auc=%.4f logloss=%.4f", kind, trial, rows[-1]["cv_auc"], rows[-1]["cv_log_loss"])
    table = pd.DataFrame(rows).sort_values("cv_log_loss").reset_index(drop=True)
    best = {name: table.loc[0, name].item() if hasattr(table.loc[0, name], "item") else table.loc[0, name] for name in space}
    return best, table


def select_calibration(margins: np.ndarray, y: np.ndarray, seed: int) -> tuple[str, dict[str, float]]:
    """Choose a calibration method by cross validated Brier score on the calibration set."""
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for method in ["none", "platt", "isotonic"]:
        fold_scores = []
        for fit_idx, eval_idx in folds.split(margins, y):
            calibrator = MarginCalibrator(method).fit(margins[fit_idx], y[fit_idx])
            fold_scores.append(brier_score_loss(y[eval_idx], calibrator.predict(margins[eval_idx])))
        scores[method] = float(np.mean(fold_scores))
    return min(scores, key=scores.get), scores


@dataclass
class ModelBundle:
    """Everything the serving layer needs, saved as a single artifact."""

    preprocessor: Pipeline
    models: dict
    weights: dict[str, float]
    calibrator: MarginCalibrator
    feature_names: list[str]
    age_band_rates: dict[str, float]
    base_rate: float
    metadata: dict

    def margin(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(X_raw)
        return sum(w * raw_margin(self.models[name], X) for name, w in self.weights.items() if w > 0)

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.margin(X_raw))


def _logistic_design(X: pd.DataFrame) -> pd.DataFrame:
    """Add hinge terms so the linear baseline can express the age decline."""
    out = X.copy()
    out["age_over_35"] = np.clip(out["female_age"] - 35, 0, None)
    out["age_over_40"] = np.clip(out["female_age"] - 40, 0, None)
    out["log_amh"] = np.log(out["amh_ng_ml"])
    return out


def run_training(cfg: Config) -> dict:
    """Run the full training workflow and persist artifacts and reports."""
    started = time.time()
    rng = np.random.default_rng(cfg.seed)
    raw = pd.read_csv(resolve(cfg.data.raw_path))
    clean, audit = clean_register(raw)
    splits = temporal_split(clean, cfg.split.train_until_year, cfg.split.calibration_years, cfg.split.test_years)
    processed_dir = resolve(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(processed_dir / f"{name}.csv.gz", index=False)
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})

    (Xr_train, y_train), (Xr_cal, y_cal), (Xr_test, y_test) = (features_and_target(splits[k]) for k in ["train", "calibration", "test"])
    preprocessor = build_preprocessor().fit(Xr_train)
    X_train, X_cal, X_test = (preprocessor.transform(x) for x in [Xr_train, Xr_cal, Xr_test])

    # Early stopping uses the most recent training year, mirroring the temporal design.
    recent = (splits["train"]["treatment_year"] == cfg.split.train_until_year).to_numpy()
    X_fit, y_fit, X_stop, y_stop = X_train[~recent], y_train[~recent], X_train[recent], y_train[recent]

    models, search_tables, best_params = {}, [], {}
    for kind in ["xgboost", "lightgbm"]:
        params, table = random_search(kind, X_fit, y_fit, cfg, rng)
        search_tables.append(table)
        probe = make_model(kind, params, cfg.training.max_boost_rounds, cfg.seed, cfg.training.early_stopping_rounds)
        _fit(kind, probe, X_fit, y_fit, eval_set=[(X_stop, y_stop)], early_stopping=cfg.training.early_stopping_rounds)
        best_iter = getattr(probe, "best_iteration", None) or getattr(probe, "best_iteration_", None) or cfg.training.max_boost_rounds
        rounds = int(best_iter * 1.1) + 1
        models[kind] = _fit(kind, make_model(kind, params, rounds, cfg.seed), X_train, y_train)
        best_params[kind] = {**params, "n_estimators": rounds}
        logger.info("%s refit on full training window with %d rounds", kind, rounds)

    # Candidate final models: each learner alone and their log odds average.
    cal_margins = {k: raw_margin(m, X_cal) for k, m in models.items()}
    candidates = {
        "xgboost": {"xgboost": 1.0, "lightgbm": 0.0},
        "lightgbm": {"xgboost": 0.0, "lightgbm": 1.0},
        "ensemble": {"xgboost": 0.5, "lightgbm": 0.5},
    }
    y_cal_np = y_cal.to_numpy()
    selection = {}
    for name, weights in candidates.items():
        margin = sum(w * cal_margins[k] for k, w in weights.items())
        method, scores = select_calibration(margin, y_cal_np, cfg.seed)
        selection[name] = {"method": method, "cv_brier": scores[method], "all_methods": scores}
    chosen = min(selection, key=lambda k: selection[k]["cv_brier"])
    weights = candidates[chosen]
    final_cal_margin = sum(w * cal_margins[k] for k, w in weights.items())
    calibrator = MarginCalibrator(selection[chosen]["method"]).fit(final_cal_margin, y_cal_np)
    logger.info("Selected %s with %s calibration", chosen, selection[chosen]["method"])

    train_bands = splits["train"]["female_age"].map(age_band)
    bundle = ModelBundle(
        preprocessor=preprocessor, models=models, weights=weights, calibrator=calibrator,
        feature_names=list(X_train.columns),
        age_band_rates={label: float(y_train[train_bands == label].mean()) for _, _, label in AGE_BANDS},
        base_rate=float(y_train.mean()),
        metadata={"version": __version__, "final_model": chosen, "calibration": selection[chosen]["method"],
                  "best_params": best_params, "trained_rows": int(len(X_train)), "seed": cfg.seed},
    )

    # ------------------------------------------------------------ evaluation
    y_test_np = y_test.to_numpy()
    test_margins = {k: raw_margin(m, X_test) for k, m in models.items()}
    predictions = {
        "FertiCast final": bundle.predict_proba(Xr_test),
        "FertiCast uncalibrated": sigmoid(sum(w * test_margins[k] for k, w in weights.items())),
        "XGBoost": sigmoid(test_margins["xgboost"]),
        "LightGBM": sigmoid(test_margins["lightgbm"]),
    }
    age_only = make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=1000))
    age_only.fit(Xr_train[["female_age"]], y_train)
    predictions["Age only baseline"] = age_only.predict_proba(Xr_test[["female_age"]])[:, 1]
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logistic.fit(_logistic_design(X_train), y_train)
    predictions["Logistic regression"] = logistic.predict_proba(_logistic_design(X_test))[:, 1]

    # Ablation: let XGBoost route missing values natively instead of domain imputation.
    engineer_only = FeatureEngineer().fit(Xr_train)
    native = _fit("xgboost", make_model("xgboost", {k: v for k, v in best_params["xgboost"].items() if k != "n_estimators"},
                                         best_params["xgboost"]["n_estimators"], cfg.seed),
                  engineer_only.transform(Xr_train), y_train)
    predictions["XGBoost native missing"] = sigmoid(raw_margin(native, engineer_only.transform(Xr_test)))
    if "true_probability" in splits["test"]:
        predictions["Simulator ceiling"] = splits["test"]["true_probability"].to_numpy()

    metrics = {name: classification_metrics(y_test_np, p) for name, p in predictions.items()}
    final = predictions["FertiCast final"]
    rounds = cfg.evaluation.bootstrap_rounds
    metrics["FertiCast final"]["roc_auc_ci"] = bootstrap_ci(y_test_np, final, roc_auc_score, rounds, cfg.seed)
    metrics["FertiCast final"]["brier_ci"] = bootstrap_ci(y_test_np, final, brier_score_loss, rounds, cfg.seed)

    test = splits["test"].assign(prediction=final, band=splits["test"]["female_age"].map(age_band))
    subgroups = []
    for column in ["band", "infertility_cause", "sperm_source"]:
        for level, group in test.groupby(column):
            subgroups.append({
                "dimension": "age_band" if column == "band" else column, "level": level, "n": int(len(group)),
                "observed_rate": float(group[TARGET].mean()), "mean_predicted": float(group["prediction"].mean()),
                "roc_auc": float(roc_auc_score(group[TARGET], group["prediction"])) if group[TARGET].nunique() > 1 else None,
            })

    report = {
        "generated_seconds": round(time.time() - started, 1),
        "rows": {k: int(len(v)) for k, v in splits.items()},
        "live_birth_rate": {k: float(v[TARGET].mean()) for k, v in splits.items()},
        "cleaning_audit": audit,
        "missing_rate_train": Xr_train[RAW_FEATURES].isna().mean().round(4).to_dict(),
        "selection": selection, "final_model": chosen, "metrics": metrics, "subgroups": subgroups,
        "best_params": best_params,
    }

    model_dir, report_dir = resolve(cfg.artifacts.model_dir), resolve(cfg.artifacts.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "ferticast_bundle.joblib", compress=3)
    pd.concat(search_tables).to_csv(report_dir / "hyperparameter_search.csv", index=False)
    pd.DataFrame({"y": y_test_np, **predictions}).to_csv(report_dir / "test_predictions.csv.gz", index=False)
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Final test metrics: %s", metrics["FertiCast final"])
    return report

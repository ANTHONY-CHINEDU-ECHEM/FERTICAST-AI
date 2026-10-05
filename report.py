"""Figure generation for the README and the model card."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from ferticast.config import Config, resolve
from ferticast.data.cleaning import features_and_target
from ferticast.data.schema import AGE_BANDS, age_band
from ferticast.explain.shap_explainer import EnsembleExplainer
from ferticast.models.metrics import net_benefit

INK, TEAL, CORAL, SAND, SLATE = "#1F2A37", "#0E7C7B", "#E4572E", "#D8C99B", "#6B7A8F"
PALETTE = {"FertiCast final": TEAL, "Logistic regression": SLATE, "Age only baseline": CORAL, "Simulator ceiling": SAND}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans",
    })


def plot_waterfall(explanation: dict, path, title: str = "What drives this prediction") -> None:
    """Probability space waterfall for a single patient explanation."""
    _style()
    factors = explanation["factors"]
    base = 100 * explanation["baseline_probability"]
    fig, ax = plt.subplots(figsize=(8.2, 0.48 * len(factors) + 2.2))
    running = base
    for i, item in enumerate(factors):
        effect = item["effect_percentage_points"]
        ax.barh(i, effect, left=running, color=TEAL if effect > 0 else CORAL, height=0.62)
        anchor = running + effect
        ax.text(anchor + (0.25 if effect > 0 else -0.25), i, f"{effect:+.1f}", va="center",
                ha="left" if effect > 0 else "right", fontsize=9.5)
        ax.plot([anchor, anchor], [i - 0.31, i + 0.69], color=INK, lw=0.6, alpha=0.5)
        running = anchor
    ax.set_yticks(range(len(factors)), [f["factor"] for f in factors])
    ax.invert_yaxis()
    ax.axvline(base, color=SLATE, ls="--", lw=1)
    final = 100 * explanation["probability"]
    ax.axvline(final, color=INK, lw=1.2)
    ax.set_xlabel("Predicted chance of live birth (%)")
    ax.set_title(f"{title}\nAverage patient {base:.1f}%  to  this patient {final:.1f}%", loc="left")
    low, high = ax.get_xlim()
    ax.set_xlim(low - 3.0, high + 1.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_figures(cfg: Config) -> list[str]:
    """Recreate every figure in docs/images from saved artifacts."""
    _style()
    figure_dir, report_dir = resolve(cfg.artifacts.figure_dir), resolve(cfg.artifacts.report_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(resolve(cfg.artifacts.model_dir) / "ferticast_bundle.joblib")
    preds = pd.read_csv(report_dir / "test_predictions.csv.gz")
    test = pd.read_csv(resolve(cfg.data.processed_dir) / "test.csv.gz")
    train = pd.read_csv(resolve(cfg.data.processed_dir) / "train.csv.gz")
    with open(report_dir / "metrics.json", encoding="utf-8") as handle:
        report = json.load(handle)
    y = preds["y"].to_numpy()
    written = []

    # 1. Discrimination.
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name in ["Simulator ceiling", "FertiCast final", "Logistic regression", "Age only baseline"]:
        if name in preds:
            fpr, tpr, _ = roc_curve(y, preds[name])
            ax.plot(fpr, tpr, color=PALETTE[name], lw=2.2 if name == "FertiCast final" else 1.5,
                    ls=":" if name == "Simulator ceiling" else "-",
                    label=f"{name} ({report['metrics'][name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Discrimination on the temporal hold out")
    ax.legend(title="ROC AUC", frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figure_dir / "roc_curves.png"); plt.close(fig); written.append("roc_curves.png")

    # 2. Calibration before and after.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for name, color in [("FertiCast uncalibrated", CORAL), ("FertiCast final", TEAL)]:
        observed, predicted = calibration_curve(y, preds[name], n_bins=12, strategy="quantile")
        axes[0].plot(100 * predicted, 100 * observed, "o-", color=color, lw=1.8, ms=5,
                     label=f"{name.replace('FertiCast ', '').capitalize()} (ECE {100 * report['metrics'][name]['ece']:.2f} pts)")
    axes[0].plot([0, 60], [0, 60], color=INK, lw=0.8, alpha=0.5)
    axes[0].set(xlabel="Predicted chance (%)", ylabel="Observed live birth rate (%)", title="Reliability on unseen years")
    axes[0].legend(frameon=False)
    axes[1].hist(100 * preds["FertiCast final"], bins=40, color=TEAL, alpha=0.85)
    axes[1].set(xlabel="Predicted chance (%)", ylabel="Cycles", title="Spread of individual predictions")
    fig.tight_layout(); fig.savefig(figure_dir / "calibration.png"); plt.close(fig); written.append("calibration.png")

    # 3. Age band: observed versus predicted.
    bands = test["female_age"].map(age_band)
    labels = [label for _, _, label in AGE_BANDS]
    observed = [100 * y[bands == b].mean() for b in labels]
    predicted = [100 * preds.loc[bands == b, "FertiCast final"].mean() for b in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, observed, 0.4, color=SAND, label="Observed")
    ax.bar(x + 0.2, predicted, 0.4, color=TEAL, label="Predicted")
    for xi, (o, p) in enumerate(zip(observed, predicted)):
        ax.text(xi - 0.2, o + 0.5, f"{o:.1f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, p + 0.5, f"{p:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set(xlabel="Female age band", ylabel="Live birth rate per cycle (%)", title="Calibration within age bands")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "age_band_calibration.png"); plt.close(fig); written.append("age_band_calibration.png")

    # 4. Decision curve.
    thresholds = list(cfg.evaluation.decision_thresholds)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    curve = net_benefit(y, preds["FertiCast final"].to_numpy(), thresholds)
    age_curve = net_benefit(y, preds["Age only baseline"].to_numpy(), thresholds)
    ax.plot(100 * np.array(thresholds), curve["model"], "o-", color=TEAL, label="FertiCast final")
    ax.plot(100 * np.array(thresholds), age_curve["model"], "s-", color=CORAL, label="Age only baseline")
    ax.plot(100 * np.array(thresholds), curve["treat_all"], color=SLATE, ls="--", label="Treat everyone")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xlabel="Threshold chance at which a couple would proceed (%)", ylabel="Net benefit",
           title="Decision curve analysis", ylim=(-0.05, None))
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "decision_curve.png"); plt.close(fig); written.append("decision_curve.png")

    # 5. Global SHAP summary on grouped clinical concepts.
    explainer = EnsembleExplainer(bundle)
    sample = test.sample(min(4000, len(test)), random_state=cfg.seed)
    X_sample, _ = features_and_target(sample)
    grouped, _ = explainer.grouped(X_sample)
    importance = grouped.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(importance.index, importance.values, color=TEAL)
    ax.set(xlabel="Mean absolute contribution to log odds", title="What the model relies on, by clinical concept")
    fig.tight_layout(); fig.savefig(figure_dir / "shap_global_importance.png"); plt.close(fig); written.append("shap_global_importance.png")

    values, _, X_eng = explainer.shap_values(X_sample)
    keep = [c for c in X_eng.columns if not c.endswith("_missing") and not c.startswith("cause_")]
    idx = [X_eng.columns.get_loc(c) for c in keep]
    plt.figure(figsize=(8, 5.6))
    shap.summary_plot(values[:, idx], X_eng[keep], show=False, max_display=12, plot_size=None)
    plt.title("SHAP summary: direction and size of each feature effect", loc="left")
    plt.tight_layout(); plt.savefig(figure_dir / "shap_beeswarm.png"); plt.close(); written.append("shap_beeswarm.png")

    # 6. Age and AMH dependence learned by the model.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feature, label in [(axes[0], "female_age", "Female age (years)"), (axes[1], "amh_ng_ml", "AMH (ng/mL)")]:
        col = X_eng.columns.get_loc(feature)
        ax.scatter(X_eng[feature], values[:, col], s=5, alpha=0.25, color=TEAL)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set(xlabel=label, ylabel="Contribution to log odds", title=f"Learned effect of {label.split(' (')[0].lower()}")
    axes[1].set_xlim(0, 10)
    fig.tight_layout(); fig.savefig(figure_dir / "shap_dependence.png"); plt.close(fig); written.append("shap_dependence.png")

    # 7. Patient level waterfalls for two contrasting profiles.
    from ferticast.inference.predictor import LiveBirthPredictor
    predictor = LiveBirthPredictor(bundle)
    profiles = {
        "waterfall_favourable.png": ("Patient A: 31 years, good reserve", {
            "female_age": 31, "male_age": 33, "bmi": 22.4, "amh_ng_ml": 3.8, "fsh_iu_l": 6.1, "afc": 16,
            "infertility_duration_years": 1.5, "infertility_cause": "tubal", "sperm_source": "partner",
            "sperm_concentration_m_ml": 62, "sperm_progressive_motility_pct": 55, "previous_ivf_cycles": 0,
            "previous_live_births": 1, "smoker": 0}),
        "waterfall_challenging.png": ("Patient B: 41 years, low reserve, raised BMI", {
            "female_age": 41, "male_age": 44, "bmi": 32.5, "amh_ng_ml": 0.6, "fsh_iu_l": 11.8, "afc": 4,
            "infertility_duration_years": 5, "infertility_cause": "diminished_reserve", "sperm_source": "partner",
            "sperm_concentration_m_ml": 38, "sperm_progressive_motility_pct": 41, "previous_ivf_cycles": 2,
            "previous_live_births": 0, "smoker": 1}),
    }
    for filename, (title, record) in profiles.items():
        plot_waterfall(predictor.explain(record), figure_dir / filename, title)
        written.append(filename)

    # 8. Data quality story: assay availability by year.
    raw_missing = train.assign(amh_missing=train["amh_ng_ml"].isna(), fsh_missing=train["fsh_iu_l"].isna())
    by_year = 100 * raw_missing.groupby("treatment_year")[["amh_missing", "fsh_missing"]].mean()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(by_year.index, by_year["amh_missing"], "o-", color=CORAL, label="AMH not recorded")
    ax.plot(by_year.index, by_year["fsh_missing"], "s-", color=TEAL, label="FSH not recorded")
    ax.set(xlabel="Treatment year", ylabel="Share of cycles (%)", title="Missingness is driven by practice, not chance")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "missingness_by_year.png"); plt.close(fig); written.append("missingness_by_year.png")
    return written

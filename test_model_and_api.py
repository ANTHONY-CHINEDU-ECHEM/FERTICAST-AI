import numpy as np
import pytest
from fastapi.testclient import TestClient

from ferticast.api.main import app, predictor_dependency
from ferticast.inference.predictor import LiveBirthPredictor
from ferticast.models.calibration import MarginCalibrator
from ferticast.models.metrics import classification_metrics, expected_calibration_error, net_benefit

RECORD = {"female_age": 33, "male_age": 35, "bmi": 23.0, "amh_ng_ml": 3.0, "fsh_iu_l": 6.5, "afc": 14,
          "infertility_duration_years": 2, "infertility_cause": "unexplained", "sperm_source": "partner",
          "sperm_concentration_m_ml": 60, "sperm_progressive_motility_pct": 50, "previous_ivf_cycles": 0,
          "previous_live_births": 0, "smoker": 0}


@pytest.fixture(scope="module")
def predictor(trained_bundle):
    return LiveBirthPredictor(trained_bundle[0])


@pytest.fixture(scope="module")
def client(predictor):
    app.dependency_overrides[predictor_dependency] = lambda: predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_training_beats_chance_and_reports_all_models(trained_bundle):
    _, report = trained_bundle
    assert report["metrics"]["FertiCast final"]["roc_auc"] > 0.6
    assert {"Age only baseline", "Logistic regression", "XGBoost native missing"} <= set(report["metrics"])


def test_age_lowers_predicted_probability(predictor):
    young, older = predictor.predict([RECORD, {**RECORD, "female_age": 43}])
    assert young > older


def test_waterfall_sums_to_prediction(predictor):
    explanation = predictor.explain(RECORD, top_k=50)
    total = sum(f["effect_percentage_points"] for f in explanation["factors"]) / 100
    assert explanation["baseline_probability"] + total == pytest.approx(explanation["probability"], abs=2e-3)
    assert explanation["probability"] == pytest.approx(float(predictor.predict([RECORD])[0]), abs=1e-4)


def test_missing_inputs_are_reported_as_imputed(predictor):
    sparse = {"female_age": 37, "infertility_cause": "tubal"}
    explanation = predictor.explain(sparse)
    assert 0 < explanation["probability"] < 1
    assert "amh_ng_ml" in explanation["imputed_fields"]


@pytest.mark.parametrize("method", ["none", "platt", "isotonic"])
def test_calibrators_are_monotone(method):
    rng = np.random.default_rng(0)
    margins = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-0.6 * margins + 0.4))).astype(int)
    calibrator = MarginCalibrator(method).fit(margins, y)
    grid = calibrator.predict(np.linspace(-4, 4, 50))
    assert np.all(np.diff(grid) >= -1e-9)
    assert grid.min() >= 0 and grid.max() <= 1


def test_metrics_on_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 30_000)
    y = (rng.random(30_000) < p).astype(int)
    metrics = classification_metrics(y, p)
    assert metrics["calibration_slope"] == pytest.approx(1.0, abs=0.08)
    assert expected_calibration_error(y, p) < 0.02
    curve = net_benefit(y, p, [0.2, 0.3])
    assert curve["model"][0] >= curve["treat_all"][0]


def test_api_contract(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/predict", json={"records": [RECORD, {"female_age": 41}]}).json()
    assert len(body["probabilities"]) == 2 and body["probabilities"][0] > body["probabilities"][1]
    explained = client.post("/explain", json=RECORD).json()
    assert explained["age_band"] == "18 to 34" and explained["factors"]
    scenario = client.post("/scenario", json={"record": {**RECORD, "bmi": 36}, "changes": {"bmi": 25}}).json()
    assert scenario["difference_percentage_points"] > 0


def test_api_rejects_implausible_values(client):
    assert client.post("/explain", json={**RECORD, "female_age": 12}).status_code == 422
    assert client.post("/explain", json={**RECORD, "bmi": 99}).status_code == 422
    assert client.post("/explain", json={**RECORD, "unknown_field": 1}).status_code == 422
    assert client.post("/scenario", json={"record": RECORD, "changes": {"bmi": 500}}).status_code == 422

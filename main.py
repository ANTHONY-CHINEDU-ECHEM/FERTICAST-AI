"""FastAPI service exposing prediction, explanation and scenario endpoints."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ferticast import __version__
from ferticast.data.schema import NUMERIC_SPECS
from ferticast.inference.predictor import LiveBirthPredictor, get_predictor

app = FastAPI(
    title="FertiCast AI",
    version=__version__,
    description="Calibrated pre treatment IVF live birth probability with SHAP explanations. "
                "Research and decision support prototype. Not a medical device.",
)


def _bounded(name: str, required: bool = False):
    spec = NUMERIC_SPECS[name]
    return Field(... if required else None, ge=spec.low, le=spec.high, description=f"{spec.label} ({spec.unit})")


class PatientRecord(BaseModel):
    """Baseline work up for one couple. Optional fields may be omitted and are imputed."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "female_age": 36, "male_age": 38, "bmi": 26.1, "amh_ng_ml": 1.6, "fsh_iu_l": 8.2, "afc": 9,
        "infertility_duration_years": 3, "infertility_cause": "unexplained", "sperm_source": "partner",
        "sperm_concentration_m_ml": 48, "sperm_progressive_motility_pct": 45, "previous_ivf_cycles": 1,
        "previous_live_births": 0, "smoker": 0}})

    female_age: float = _bounded("female_age", required=True)
    male_age: Optional[float] = _bounded("male_age")
    bmi: Optional[float] = _bounded("bmi")
    amh_ng_ml: Optional[float] = _bounded("amh_ng_ml")
    fsh_iu_l: Optional[float] = _bounded("fsh_iu_l")
    afc: Optional[float] = _bounded("afc")
    infertility_duration_years: Optional[float] = _bounded("infertility_duration_years")
    sperm_concentration_m_ml: Optional[float] = _bounded("sperm_concentration_m_ml")
    sperm_progressive_motility_pct: Optional[float] = _bounded("sperm_progressive_motility_pct")
    previous_ivf_cycles: float = _bounded("previous_ivf_cycles")
    previous_live_births: float = _bounded("previous_live_births")
    infertility_cause: Literal["unexplained", "male_factor", "tubal", "ovulatory_pcos", "endometriosis",
                               "diminished_reserve", "combined"] = "unexplained"
    sperm_source: Literal["partner", "donor"] = "partner"
    smoker: int = Field(0, ge=0, le=1)


class BatchRequest(BaseModel):
    records: list[PatientRecord] = Field(..., min_length=1, max_length=1000)


class ScenarioRequest(BaseModel):
    record: PatientRecord
    changes: dict


def predictor_dependency() -> LiveBirthPredictor:
    try:
        return get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info(predictor: LiveBirthPredictor = Depends(predictor_dependency)) -> dict:
    bundle = predictor.bundle
    return {"metadata": bundle.metadata, "age_band_rates": bundle.age_band_rates, "features": bundle.feature_names}


@app.post("/predict")
def predict(request: BatchRequest, predictor: LiveBirthPredictor = Depends(predictor_dependency)) -> dict:
    probabilities = predictor.predict([r.model_dump() for r in request.records])
    return {"probabilities": [round(float(p), 4) for p in probabilities]}


@app.post("/explain")
def explain(record: PatientRecord, predictor: LiveBirthPredictor = Depends(predictor_dependency)) -> dict:
    return predictor.explain(record.model_dump())


@app.post("/scenario")
def scenario(request: ScenarioRequest, predictor: LiveBirthPredictor = Depends(predictor_dependency)) -> dict:
    try:
        changed = PatientRecord(**{**request.record.model_dump(), **request.changes})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid scenario: {exc}") from exc
    return predictor.what_if(request.record.model_dump(), changed.model_dump())

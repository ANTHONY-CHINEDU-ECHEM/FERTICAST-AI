"""Clinician facing Streamlit application for FertiCast AI.

Run with ``streamlit run app/streamlit_app.py`` after training a model.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from ferticast.data.schema import INFERTILITY_CAUSES, NUMERIC_SPECS
from ferticast.inference.predictor import LiveBirthPredictor
from ferticast.models.report import plot_waterfall

st.set_page_config(page_title="FertiCast AI", layout="wide")

CAUSE_LABELS = {
    "unexplained": "Unexplained", "male_factor": "Male factor", "tubal": "Tubal disease",
    "ovulatory_pcos": "Ovulatory disorder or PCOS", "endometriosis": "Endometriosis",
    "diminished_reserve": "Diminished ovarian reserve", "combined": "Combined female and male factors",
}


@st.cache_resource(show_spinner="Loading model")
def load_predictor() -> LiveBirthPredictor:
    return LiveBirthPredictor.load()


def number(name: str, known: bool = True):
    """Numeric input bound to the shared clinical schema. Returns None when not measured."""
    spec = NUMERIC_SPECS[name]
    if not known:
        return None
    is_whole = float(spec.step).is_integer()
    cast = int if is_whole else float
    return st.number_input(f"{spec.label} ({spec.unit})", min_value=cast(spec.low), max_value=cast(spec.high),
                           value=cast(spec.default), step=cast(spec.step), key=name)


st.title("FertiCast AI")
st.caption("Pre treatment estimate of the chance of a live birth from one IVF cycle, with the reasons behind it.")

try:
    predictor = load_predictor()
except FileNotFoundError as error:
    st.error(f"{error} Train a model with `make all`, then reload this page.")
    st.stop()

with st.sidebar:
    st.header("Baseline work up")
    st.subheader("Female partner")
    record = {"female_age": number("female_age"), "bmi": number("bmi")}
    amh_known = st.checkbox("AMH measured", value=True)
    record["amh_ng_ml"] = number("amh_ng_ml", amh_known)
    fsh_known = st.checkbox("Basal FSH measured", value=True)
    record["fsh_iu_l"] = number("fsh_iu_l", fsh_known)
    afc_known = st.checkbox("Antral follicle count measured", value=True)
    record["afc"] = number("afc", afc_known)
    record["smoker"] = int(st.checkbox("Current smoker", value=False))

    st.subheader("History")
    record["infertility_cause"] = st.selectbox("Main cause of infertility", INFERTILITY_CAUSES, format_func=CAUSE_LABELS.get)
    record["infertility_duration_years"] = number("infertility_duration_years")
    record["previous_ivf_cycles"] = number("previous_ivf_cycles")
    record["previous_live_births"] = number("previous_live_births")

    st.subheader("Sperm")
    record["sperm_source"] = st.radio("Sperm source", ["partner", "donor"], horizontal=True, format_func=str.capitalize)
    partner = record["sperm_source"] == "partner"
    record["male_age"] = number("male_age", partner)
    record["sperm_concentration_m_ml"] = number("sperm_concentration_m_ml", partner)
    record["sperm_progressive_motility_pct"] = number("sperm_progressive_motility_pct", partner)

explanation = predictor.explain(record)
probability, band_average = explanation["probability"], explanation["age_band_average"]

left, middle, right = st.columns(3)
left.metric("Predicted chance of live birth", f"{100 * probability:.1f}%")
middle.metric(f"Average for age {explanation['age_band']}", f"{100 * band_average:.1f}%",
              delta=f"{100 * (probability - band_average):+.1f} points versus peers", delta_color="normal")
right.metric("Average across all patients", f"{100 * explanation['baseline_probability']:.1f}%")

if explanation["imputed_fields"]:
    labels = ", ".join(NUMERIC_SPECS[f].label for f in explanation["imputed_fields"])
    st.info(f"Not measured and estimated from related findings: {labels}. Entering measured values will sharpen the estimate.")

chart, table = st.columns([3, 2])
with chart:
    st.subheader("Why the estimate differs from the average patient")
    buffer = io.BytesIO()
    plot_waterfall(explanation, buffer, "Contribution of each baseline factor")
    st.image(buffer.getvalue(), width="stretch")
with table:
    st.subheader("Factors in plain terms")
    frame = pd.DataFrame(explanation["factors"])
    frame["Effect"] = frame["effect_percentage_points"].map(lambda v: f"{v:+.1f} points")
    frame["Can change"] = frame["modifiable"].map({True: "Yes", False: "No"})
    st.dataframe(frame.rename(columns={"factor": "Factor"})[["Factor", "Effect", "Can change"]],
                 hide_index=True, width="stretch")

st.subheader("Explore a change before treatment")
col_bmi, col_smoke, col_result = st.columns(3)
bmi_spec = NUMERIC_SPECS["bmi"]
target_bmi = col_bmi.slider("Body mass index at cycle start", float(bmi_spec.low), float(bmi_spec.high), float(record["bmi"]), 0.5)
stops_smoking = col_smoke.checkbox("Stops smoking", value=False, disabled=not record["smoker"])
scenario = predictor.what_if(record, {"bmi": target_bmi, "smoker": 0 if stops_smoking else record["smoker"]})
col_result.metric("Chance under this scenario", f"{100 * scenario['scenario']:.1f}%",
                  delta=f"{scenario['difference_percentage_points']:+.1f} points")
st.caption("Scenario estimates describe associations learned from past cycles. They are not a guarantee that changing a "
           "factor will change the outcome for an individual.")

st.divider()
st.caption("FertiCast AI is a research prototype trained on synthetic register data. It is not a medical device and must "
           "not be used to make treatment decisions.")

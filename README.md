# FertiCast AI

**Calibrated, explainable prediction of live birth before an IVF cycle begins**

<p align="center">
  <img width="1440" height="1250" alt="streamlit_app" src="https://github.com/user-attachments/assets/7fac94cd-710b-47a3-bfef-f22b078edd22" />
</p>

## Project brief

A couple considering in vitro fertilisation is asked to commit money, months of treatment and a great deal of emotional energy before a single egg has been collected. The question they bring to the first consultation is simple: what are our chances? The answer they usually receive is a national success rate for the woman's age band. That figure is honest, but it is an average over thousands of very different couples. Two women of 33 can sit in the same age band while one has a strong ovarian reserve and a previous child, and the other has low AMH, a raised BMI and two failed cycles behind her. Quoting both the same number serves neither of them.

Clinics feel the same gap from the other side. Counselling time is limited, treatment funding rules are increasingly tied to expected benefit, and regulators expect clinics to give patients information that is accurate for them rather than flattering in aggregate. An individual estimate is only useful, though, if it can be trusted in two specific ways. It must be calibrated, meaning that among couples told 30 percent roughly 30 in 100 really do take a baby home, and it must be explainable, so that a clinician can show which baseline findings moved the estimate and which of those the couple can still influence before treatment starts.

FertiCast AI is a complete machine learning product built around that need. It takes the baseline work up that already exists at the first consultation (female age, AMH, FSH, antral follicle count, BMI, duration and cause of infertility, semen analysis and treatment history) and returns a calibrated probability of live birth from one cycle, a waterfall chart that shows how each factor moved the estimate away from the average patient, and a scenario tool for modifiable factors. The repository covers the whole path from raw register data to a served model: data cleaning with an audit trail, domain informed imputation, tuned gradient boosting ensembles, probability calibration on recent years, temporal validation, SHAP explanations, a REST API, a clinician facing web application, automated tests, containers and continuous integration.

One thing should be stated plainly. Real fertility registers that contain hormone assays cannot be published, and the open HFEA register does not include AMH, BMI or semen parameters. The model in this repository is therefore trained on a large synthetic register of 120,000 cycles whose structure follows published reproductive medicine relationships and whose data quality problems mirror real clinical systems. Every number below describes performance on that synthetic register. The engineering is real and reusable; the clinical performance figures are not claims about real patients.

## What the system does

<table>
  <tr><th align="left">Capability</th><th align="left">How it is delivered</th></tr>
  <tr><td>Individual live birth probability</td><td>Log odds ensemble of XGBoost and LightGBM, recalibrated on the most recent treatment years</td></tr>
  <tr><td>Reasons behind each estimate</td><td>TreeSHAP values grouped into clinical concepts and expressed in percentage points that sum exactly to the prediction</td></tr>
  <tr><td>Handling of incomplete work ups</td><td>A purpose built imputer that treats donor sperm cycles, unordered assays and keyed in errors differently</td></tr>
  <tr><td>Scenario exploration</td><td>What if comparison for modifiable factors such as BMI and smoking</td></tr>
  <tr><td>Integration</td><td>FastAPI service with validated inputs, plus a Streamlit application for consultations</td></tr>
  <tr><td>Assurance</td><td>Temporal hold out, bootstrap confidence intervals, subgroup audit, decision curve analysis, 22 automated tests</td></tr>
</table>

## Key findings

### The model recovers almost all of the signal that exists

On 20,064 cycles from the two most recent years, which the model never saw during development, FertiCast AI reaches a ROC AUC of 0.697 (95 percent interval 0.689 to 0.705). Because the register is simulated, the best score any model could achieve is known: 0.705. The pipeline therefore captures essentially all of the learnable signal, and the remaining gap is clinic level variation that is deliberately withheld from the model so that it transfers between clinics. A discrimination figure near 0.70 is also typical of published pre treatment IVF models, which is a useful reminder that much of the outcome is decided by events after stimulation begins and cannot be known at the first consultation.

<table>
  <tr><th align="left">Model</th><th>ROC AUC</th><th>PR AUC</th><th>Brier score</th><th>Calibration error (points)</th></tr>
  <tr><td>Age only baseline (what age band tables offer)</td><td align="center">0.671</td><td align="center">0.391</td><td align="center">0.1859</td><td align="center">1.60</td></tr>
  <tr><td>Logistic regression with hand built age hinges</td><td align="center">0.698</td><td align="center">0.438</td><td align="center">0.1813</td><td align="center">1.33</td></tr>
  <tr><td>XGBoost</td><td align="center">0.697</td><td align="center">0.436</td><td align="center">0.1815</td><td align="center">1.44</td></tr>
  <tr><td>LightGBM</td><td align="center">0.697</td><td align="center">0.436</td><td align="center">0.1815</td><td align="center">1.48</td></tr>
  <tr><td><b>FertiCast AI (calibrated ensemble)</b></td><td align="center"><b>0.697</b></td><td align="center"><b>0.436</b></td><td align="center"><b>0.1812</b></td><td align="center"><b>0.71</b></td></tr>
  <tr><td>Theoretical ceiling of the simulator</td><td align="center">0.705</td><td align="center">0.449</td><td align="center">0.1797</td><td align="center">0.80</td></tr>
</table>

<p align="center"><img width="896" height="832" alt="roc_curves" src="https://github.com/user-attachments/assets/05b8b22e-3eb5-4310-b3d9-981d69ca8d50" /></p>

Two results in this table deserve an honest reading. First, a carefully specified logistic regression matches the boosted ensemble on discrimination. That is expected when the underlying process is close to additive, and it is reported here rather than hidden. The ensemble earns its place because it finds the age decline, the saturating AMH benefit and the BMI tails without anyone specifying them by hand, and because after recalibration it has the lowest calibration error of any candidate. Second, the lift over the age only baseline (0.671 to 0.697 in ROC AUC, 0.391 to 0.436 in PR AUC) looks modest as a summary statistic but is large for the individual, as the next finding shows.

### Age band averages hide a twofold range of individual chances

Among women aged 18 to 34 in the hold out years, the model's estimates run from 25 percent at the tenth percentile to 49 percent at the ninetieth. For women aged 40 to 42 the same range is 5 percent to 16 percent. In both groups the couples at the top are roughly two to three times more likely to succeed than those at the bottom, yet an age band table would quote each group a single figure. This is the core business case: personalised estimates change the conversation for the patients who are furthest from their band average, in either direction.

<p align="center"><img width="1152" height="672" alt="age_band_calibration" src="https://github.com/user-attachments/assets/787b973d-5751-4b25-954c-5b3d212bd14c" /></p>

The chart above confirms that the individual estimates still add up correctly. Within every age band the mean prediction is close to the observed live birth rate, including the oldest group where rates are very low and a small absolute error would be a large relative one.

### Calibration has to be maintained, not assumed

Success rates in the register improve slowly over calendar time, as they have in real practice. A model trained on cycles up to 2019 therefore underestimates outcomes in later years. Before recalibration the ensemble's calibration error on the hold out years is 1.51 points; after a logistic recalibration fitted on 2020 and 2021 it falls to 0.71 points, which is as good as the simulator's own true probabilities achieve on a sample of this size. Recalibration is chosen automatically from three options (none, Platt scaling, isotonic regression) by cross validated Brier score. The practical lesson is that a clinic deploying a model like this needs a yearly recalibration step in its governance process, and the pipeline is built so that step is a single command.

<p align="center"><img width="1600" height="736" alt="calibration" src="https://github.com/user-attachments/assets/89cac780-146e-4bf8-b314-d5d8c21915cf" /></p>

### Female age dominates, but it is far from the whole story

Grouped SHAP values show that female age accounts for 39 percent of the model's total attribution, AMH for 15 percent, and antral follicle count, previous live births, previous IVF cycles and BMI for 6 to 7 percent each. Semen quality, duration and cause of infertility and smoking make up most of the remainder. The dependence plots show the model has learned the clinically expected shapes without being told them: a gentle decline to age 35 followed by an accelerating fall, and an AMH benefit that flattens at higher values.

<p align="center"><img width="1184" height="768" alt="shap_global_importance" src="https://github.com/user-attachments/assets/ddd726e8-a337-4c11-8c6c-2e003642d311" /></p>
<p align="center"><img width="1600" height="640" alt="shap_dependence" src="https://github.com/user-attachments/assets/02be8e43-8c42-4a5d-88f8-bb09568405e3" /></p>
<p align="center"><img width="1280" height="896" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/bf6cb840-9889-4980-93cc-3fe3cbbfcf1d" /></p>

### Explanations a patient can read

Standard SHAP waterfalls are drawn in log odds, which mean little in a consultation. FertiCast AI converts them into percentage points by passing the cumulative log odds through the calibrator one factor at a time, so every bar is a change in the chance of live birth and the bars sum exactly to the final estimate (this property is enforced by a unit test). Indicator columns are merged into the clinical concept they belong to, so "cause of infertility" is one bar rather than seven.

<table>
  <tr>
    <td><img width="1312" height="1043" alt="waterfall_favourable" src="https://github.com/user-attachments/assets/fcb23064-6972-4dcb-aec7-54a25e2ae509" /></td>
    <td><img width="1312" height="1043" alt="waterfall_challenging" src="https://github.com/user-attachments/assets/029e3a83-cfea-4b7a-b63c-a87616005cb3" /></td>
  </tr>
</table>

### Modifiable factors matter most where the baseline prognosis is good

The scenario tool makes a point that is easy to miss in aggregate statistics. For a 36 year old with good ovarian reserve, a BMI of 34 and a smoking habit, moving to a BMI of 25 and stopping smoking is associated with a rise from 30 percent to 43 percent, a gain of 13 points. For a 41 year old with low reserve the same changes are associated with a gain of about 1 point, from 3.4 percent to 4.4 percent. Lifestyle counselling has its greatest expected return in patients whose biology is otherwise favourable, which is useful when a clinic decides where to focus pre treatment optimisation programmes. These are associations in the training data and the application says so on screen.

### The model adds decision value across realistic thresholds

Decision curve analysis asks whether acting on the model leaves couples better off than a blanket policy. The calibrated ensemble beats a treat everyone policy at every threshold tested. Against a decision made on age alone it is indistinguishable at very low thresholds (5 to 10 percent) and pulls ahead from 15 percent upward, with the widest margin between 30 and 40 percent. In plain terms, the extra inputs matter most for couples whose decision is finely balanced, and add little when almost anyone would proceed.

<p align="center"><img width="1056" height="672" alt="decision_curve" src="https://github.com/user-attachments/assets/145fa9e6-3e47-4ef4-bab2-8f57c3ff3cd1" /></p>

### Data quality work was as important as modelling

The raw register arrives the way real clinical extracts do. The cleaning stage removed 600 duplicate submissions, converted 18,180 AMH results that one clinic in five had reported in pmol/L rather than ng/mL, normalised 5,263 inconsistent diagnosis labels and blanked 943 placeholder or impossible values such as a BMI of 99. Left uncorrected, the unit mix up alone would have told the model that a fifth of patients had an ovarian reserve seven times higher than reality.

Missingness is driven by practice rather than chance. AMH is absent for more than half of cycles before 2015, when the assay was not routine, and semen parameters are absent for every donor sperm cycle because no partner sample exists. The imputer handles each case on its merits: AMH is predicted from age and FSH, FSH from age and AMH, donor cycles receive a screened donor reference profile rather than a population median dragged down by male factor patients, and every imputed field carries an indicator. An ablation in which XGBoost routes missing values natively scores the same on discrimination (0.698), so the domain imputer is not claimed to improve accuracy. Its value is that the logistic baseline, the scenario tool and the explanations all receive complete, clinically sensible records, and that the application can tell the clinician exactly which inputs were estimated.

<p align="center"><img width="1088" height="608" alt="missingness_by_year" src="https://github.com/user-attachments/assets/5aa025a5-958e-41a2-aa4e-efd8ce67f89b" /></p>

### Performance is consistent across patient groups

ROC AUC by cause of infertility ranges from 0.67 (endometriosis, ovulatory disorder) to 0.71 (diminished ovarian reserve), and mean predictions sit within two points of observed rates for every diagnosis and for both partner and donor sperm cycles. Within a single age band discrimination is lower (0.59 to 0.61) because the strongest predictor has been held fixed; this is expected and is the fairest measure of what the non age inputs contribute.

## Architecture

```
raw register (CSV)
      |
      v
clean_register        duplicates, unit harmonisation, label normalisation, range checks, audit trail
      |
      v
temporal_split        train to 2019 | calibrate on 2020 and 2021 | test on 2022 and 2023
      |
      v
DomainImputer  >  FeatureEngineer          fitted on training years only
      |
      v
random search with cross validation        XGBoost and LightGBM, selected on log loss
      |
      v
log odds ensemble  >  MarginCalibrator     none, Platt or isotonic, chosen by cross validated Brier score
      |
      v
ModelBundle (single joblib artifact)
      |
      +====> FastAPI service        /predict  /explain  /scenario  /model  /health
      +====> Streamlit application  consultation view with waterfall and scenario tool
      +====> reports and figures    metrics.json, subgroup audit, docs/images
```

## Repository layout

```
ferticast_ai
    app/streamlit_app.py              clinician application
    configs/config.yaml               data, split, training and evaluation settings
    data/sample                       two thousand example rows of the raw register
    docs/images                       every figure in this document, regenerated by make report
    docs/model_card.md                intended use, evaluation and limitations
    models/ferticast_bundle.joblib    trained preprocessing, models and calibrator
    reports/metrics.json              full evaluation output
    src/ferticast
        data/schema.py                feature ranges and vocabularies shared by every layer
        data/simulate.py              synthetic register generator with realistic data quality faults
        data/cleaning.py              cleaning rules, audit and temporal split
        data/hfea_adapter.py          mapping from the public HFEA register to this schema
        features/imputation.py        domain informed imputer
        features/engineering.py       derived features and display grouping
        models/train.py               tuning, ensembling, calibration, evaluation
        models/calibration.py         margin calibrator
        models/metrics.py             discrimination, calibration and net benefit metrics
        models/report.py              figure generation
        explain/shap_explainer.py     ensemble TreeSHAP and probability space waterfall
        inference/predictor.py        serving facade
        api/main.py                   FastAPI service
        cli.py                        command line entry point
    tests                             22 tests covering data, features, model behaviour and the API
    Dockerfile, compose.yaml    container images for the API and the application
    .github/workflows/ci.yml          lint and test on every push
```

## Getting started

Python 3.10 or later is required.

```
make install
make all
make test
```

`make all` generates the register, trains and calibrates the models, evaluates them and rebuilds every figure. On a single CPU core the full run takes about two minutes. A trained bundle is already included, so the application and API also work straight after `make install`.

Open the consultation application:

```
make app
```

Start the API and read the interactive documentation at `http://localhost:8000/docs`:

```
make api
```

Or run both in containers:

```
docker compose up
```

### Calling the API

```python
import httpx

record = {
    "female_age": 36, "bmi": 26.1, "amh_ng_ml": 1.6, "afc": 9,
    "infertility_duration_years": 3, "infertility_cause": "unexplained",
    "sperm_source": "partner", "sperm_concentration_m_ml": 48,
    "sperm_progressive_motility_pct": 45, "previous_ivf_cycles": 1,
    "previous_live_births": 0, "smoker": 0,
}
explanation = httpx.post("http://localhost:8000/explain", json=record).json()
print(explanation["probability"], explanation["age_band_average"])
for factor in explanation["factors"]:
    print(factor["factor"], factor["effect_percentage_points"])
```

Fields that were not measured can simply be left out. The response lists them under `imputed_fields`. Values outside plausible clinical ranges are rejected with a validation error, using the same range definitions that drive cleaning and the application form.

## Methodology notes

* **Temporal validation.** Random splits flatter clinical models because they let the model see the future. All selection decisions here use cycles up to 2019, calibration uses 2020 and 2021, and the reported figures come from 2022 and 2023.
* **Selection on proper scoring rules.** Hyperparameters are chosen on cross validated log loss and the calibration method on cross validated Brier score, since the product is a probability rather than a ranking.
* **Ensembling in log odds.** Averaging the two learners' margins rather than their probabilities keeps the ensemble's SHAP values exactly equal to the average of the learners' SHAP values, so explanations stay additive.
* **Uncertainty.** Headline metrics carry percentile bootstrap intervals from 200 resamples.
* **Single schema.** Ranges, units and vocabularies live in one module that is imported by the simulator, the cleaner, the API validators and the application form, so training and serving cannot drift apart.

## Using real data

`ferticast.data.hfea_adapter` maps the public HFEA anonymised register onto the same schema. That register reports age in bands and omits hormone assays, BMI and semen analysis, so a model trained on it relies on age, diagnosis and treatment history. A clinic with its own records can export them in the column layout shown in `data/sample` and run `make train` unchanged.

## Limitations and responsible use

* The training register is synthetic. Performance on real patients is unknown and would require validation on local data before any clinical use.
* The model predicts the outcome of one cycle from information available before treatment. It does not use stimulation response, embryo quality or transfer details.
* Scenario estimates are associations, not causal effects.
* Protected characteristics such as ethnicity are not modelled. A real deployment would need a fairness audit on those dimensions.
* FertiCast AI is a research and portfolio prototype. It is not a medical device and must not be used to make treatment decisions.

## Licence

Released under the MIT licence.

# Model card: FertiCast AI

## Intended use
Decision support prototype that estimates the probability of a live birth from one IVF cycle using baseline
findings available at the first consultation. Intended audience: fertility clinicians and data science reviewers.
Not intended for unsupervised patient use or for any treatment decision.

## Model
Log odds average of an XGBoost classifier and a LightGBM classifier, followed by Platt recalibration. Inputs are
fourteen baseline fields; missing assays are imputed with a domain informed imputer and flagged.

## Data
Synthetic register of 120,000 cycles from 64 simulated clinics, treatment years 2012 to 2023. Train: years up to
2019 (79,857 cycles). Calibration: 2020 and 2021 (20,079). Test: 2022 and 2023 (20,064).

## Evaluation on the temporal hold out
ROC AUC 0.697 (95 percent interval 0.689 to 0.705). Brier score 0.1812. Expected calibration error 0.71 points.
Calibration slope 1.05. Simulator ceiling ROC AUC 0.705. Subgroup results by age band, diagnosis and sperm source
are stored in `reports/metrics.json`.

## Ethical considerations
Predictions can influence whether a couple proceeds with treatment. Estimates must be communicated with their
uncertainty, must be recalibrated regularly, and must be validated on local real world data before use.

## Caveats
Synthetic training data; no post stimulation information; no modelling of protected characteristics; scenario
outputs are associational.

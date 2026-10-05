"""Synthetic national register simulator.

Real IVF registers cannot be redistributed with a public repository, and the
open HFEA extract does not carry hormone assays. This module therefore
generates a large register with the same statistical character as the real
thing: correlated ovarian reserve markers, age driven decline in success,
clinic level variation, a secular improvement over time and, importantly, the
messy data quality problems that make clinical modelling hard (unit mix ups,
structurally missing semen analyses for donor cycles, assays that were not
routinely ordered in earlier years, keyed in outliers and duplicate rows).

The outcome model is a documented logistic process, which means the true
discrimination ceiling is known and the pipeline can be audited against it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ferticast.data.schema import AMH_PMOL_PER_NG, INFERTILITY_CAUSES, TARGET


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_logit(df: pd.DataFrame, clinic_effect: np.ndarray | float = 0.0) -> np.ndarray:
    """Ground truth log odds of live birth for complete, clean covariates."""
    age = df["female_age"].to_numpy(float)
    amh = df["amh_ng_ml"].to_numpy(float)
    bmi = df["bmi"].to_numpy(float)
    cause = df["infertility_cause"].to_numpy()
    donor = (df["sperm_source"].to_numpy() == "donor").astype(float)
    tmc = df["sperm_concentration_m_ml"].to_numpy(float) * df["sperm_progressive_motility_pct"].to_numpy(float) / 100.0

    logit = np.full(len(df), -0.42)
    # Female age: gentle slope to 35, then an accelerating decline.
    logit += -0.02 * (age - 30) - 0.16 * np.clip(age - 35, 0, None) - 0.14 * np.clip(age - 40, 0, None)
    # Ovarian reserve: saturating benefit, with an extra penalty for very low AMH.
    logit += 0.38 * np.tanh(np.log(amh) - np.log(2.0)) - 0.35 * (amh < 0.5)
    logit += -0.15 * ((age > 38) & (amh < 1.0))
    logit += 0.012 * np.clip(df["afc"].to_numpy(float) - 10, -8, 12)
    # Body mass index: penalties on both tails.
    logit += -0.045 * np.clip(bmi - 25, 0, None) - 0.06 * np.clip(18.5 - bmi, 0, None)
    logit += -0.05 * df["infertility_duration_years"].to_numpy(float)
    cause_effect = {
        "unexplained": 0.0, "male_factor": 0.05, "tubal": -0.05, "ovulatory_pcos": 0.08,
        "endometriosis": -0.18, "diminished_reserve": -0.30, "combined": -0.22,
    }
    logit += np.array([cause_effect[c] for c in cause])
    # Semen quality matters mainly when total motile concentration is very low.
    logit += np.where(donor == 1, 0.10, -0.30 * _sigmoid((5.0 - tmc) / 2.0))
    logit += -0.012 * np.clip(df["male_age"].to_numpy(float) - 40, 0, None) * (1 - donor)
    logit += 0.28 * (df["previous_live_births"].to_numpy(float) > 0)
    logit += -0.12 * df["previous_ivf_cycles"].to_numpy(float)
    logit += -0.22 * df["smoker"].to_numpy(float)
    logit += 0.015 * (df["treatment_year"].to_numpy(float) - 2012)
    return logit + clinic_effect


def simulate_register(n_cycles: int = 120_000, n_clinics: int = 64, year_min: int = 2012,
                      year_max: int = 2023, seed: int = 42, messy: bool = True) -> pd.DataFrame:
    """Generate a synthetic IVF register.

    Parameters
    ----------
    messy:
        When true, inject realistic data quality problems after outcomes are
        drawn. Set to false to obtain the clean latent covariates.
    """
    rng = np.random.default_rng(seed)
    n = n_cycles
    age = np.clip(rng.normal(35.0, 4.6, n), 20, 46).round()

    # Cause of infertility depends on age: diminished reserve becomes more common.
    base = np.array([0.30, 0.27, 0.12, 0.12, 0.07, 0.07, 0.05])
    weights = np.tile(base, (n, 1))
    weights[:, 5] *= np.exp(0.14 * (age - 35))
    weights[:, 3] *= np.exp(-0.05 * (age - 35))
    weights /= weights.sum(axis=1, keepdims=True)
    cause_idx = (rng.random(n)[:, None] > weights.cumsum(axis=1)).sum(axis=1).clip(0, 6)
    cause = np.array(INFERTILITY_CAUSES)[cause_idx]
    pcos, dor, male = cause == "ovulatory_pcos", cause == "diminished_reserve", np.isin(cause, ["male_factor", "combined"])

    log_amh = 1.25 - 0.085 * (age - 30) + 0.9 * pcos - 1.1 * dor + rng.normal(0, 0.65, n)
    amh = np.clip(np.exp(log_amh), 0.02, 22)
    fsh = np.clip(np.exp(1.85 + 0.012 * (age - 30) - 0.22 * (log_amh - 1) + rng.normal(0, 0.25, n)), 1, 35)
    afc = rng.poisson(np.exp(2.2 + 0.45 * (log_amh - 1) - 0.02 * (age - 30))).clip(0, 55)
    bmi = np.clip(21 + rng.gamma(3.2, 1.6, n) + 2.5 * pcos, 16.5, 48)
    duration = np.clip(0.5 + rng.gamma(2.2, 1.4, n) + 0.05 * (age - 35), 0.5, 15).round(1)

    donor = rng.random(n) < 0.07
    conc = np.where(male, rng.lognormal(np.log(9), 0.9, n), rng.lognormal(np.log(55), 0.55, n)).clip(0.1, 280)
    motility = np.where(male, rng.normal(24, 11, n), rng.normal(52, 12, n)).clip(1, 95)
    male_age = np.clip(age + rng.normal(2.2, 4.5, n), 20, 65).round()

    prev_cycles = rng.poisson(0.7 + 0.04 * np.clip(age - 30, 0, None)).clip(0, 6)
    prev_births = (rng.random(n) < 0.16 + 0.01 * np.clip(age - 30, 0, None)).astype(int)
    prev_births += (rng.random(n) < 0.03).astype(int)
    smoker = (rng.random(n) < 0.09).astype(int)
    year = rng.integers(year_min, year_max + 1, n)
    clinic = rng.integers(1, n_clinics + 1, n)
    clinic_effects = rng.normal(0, 0.18, n_clinics + 1)

    df = pd.DataFrame({
        "cycle_id": np.arange(1, n + 1),
        "clinic_id": clinic,
        "treatment_year": year,
        "female_age": age,
        "male_age": male_age,
        "bmi": bmi.round(1),
        "amh_ng_ml": amh.round(2),
        "fsh_iu_l": fsh.round(1),
        "afc": afc.astype(float),
        "infertility_duration_years": duration,
        "infertility_cause": cause,
        "sperm_source": np.where(donor, "donor", "partner"),
        "sperm_concentration_m_ml": conc.round(1),
        "sperm_progressive_motility_pct": motility.round(0),
        "previous_ivf_cycles": prev_cycles.astype(float),
        "previous_live_births": prev_births.astype(float),
        "smoker": smoker,
    })
    probability = _sigmoid(true_logit(df, clinic_effects[clinic]))
    df[TARGET] = (rng.random(n) < probability).astype(int)
    df["true_probability"] = probability.round(4)

    if messy:
        df = _inject_data_quality_issues(df, rng, n_clinics)
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator, n_clinics: int) -> pd.DataFrame:
    """Corrupt a clean register the way real clinical data is corrupted."""
    df = df.copy()
    n = len(df)
    donor = (df["sperm_source"] == "donor").to_numpy()

    # 1. One clinic in five reports AMH in pmol/L rather than ng/mL.
    pmol_clinics = rng.choice(np.arange(1, n_clinics + 1), size=max(1, n_clinics // 5), replace=False)
    is_pmol = df["clinic_id"].isin(pmol_clinics).to_numpy()
    df["amh_unit"] = np.where(is_pmol, "pmol/L", "ng/mL")
    df["amh_value"] = np.where(is_pmol, df["amh_ng_ml"] * AMH_PMOL_PER_NG, df["amh_ng_ml"]).round(2)
    df = df.drop(columns="amh_ng_ml")

    # 2. AMH was not ordered routinely before 2015; FSH is often skipped when AMH exists.
    early = (df["treatment_year"] < 2015).to_numpy()
    amh_missing = rng.random(n) < np.where(early, 0.55, 0.08)
    df.loc[amh_missing, "amh_value"] = np.nan
    df.loc[(rng.random(n) < 0.25) & ~amh_missing, "fsh_iu_l"] = np.nan
    df.loc[rng.random(n) < 0.15, "afc"] = np.nan
    df.loc[rng.random(n) < 0.06, "bmi"] = np.nan
    df.loc[rng.random(n) < 0.05, "infertility_duration_years"] = np.nan

    # 3. Donor sperm cycles have no partner semen analysis: structural missingness.
    df.loc[donor, ["sperm_concentration_m_ml", "sperm_progressive_motility_pct", "male_age"]] = np.nan
    df.loc[(rng.random(n) < 0.04) & ~donor, ["sperm_concentration_m_ml", "sperm_progressive_motility_pct"]] = np.nan

    # 4. Keyed in outliers and placeholder codes.
    df.loc[rng.random(n) < 0.004, "bmi"] = 99.0
    df.loc[rng.random(n) < 0.002, "fsh_iu_l"] = 999.0
    df.loc[rng.random(n) < 0.002, "sperm_progressive_motility_pct"] = 150.0

    # 5. Inconsistent category spellings and a handful of duplicated submissions.
    spelling = rng.random(n)
    df.loc[spelling < 0.03, "infertility_cause"] = df.loc[spelling < 0.03, "infertility_cause"].str.upper()
    df.loc[spelling > 0.97, "infertility_cause"] = df.loc[spelling > 0.97, "infertility_cause"].str.replace("_", " ")
    duplicates = df.sample(frac=0.005, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, duplicates], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)

import numpy as np

from ferticast.data.cleaning import features_and_target
from ferticast.features.engineering import DISPLAY_GROUPS, FeatureEngineer
from ferticast.features.imputation import DONOR_REFERENCE_AGE, DomainImputer


def test_imputer_leaves_no_gaps_and_adds_indicators(clean):
    X, _ = features_and_target(clean)
    out = DomainImputer().fit(X).transform(X)
    assert not out.isna().any().any()
    assert out["amh_ng_ml_missing"].sum() == X["amh_ng_ml"].isna().sum()


def test_donor_cycles_receive_reference_profile_not_population_median(clean):
    X, _ = features_and_target(clean)
    imputer = DomainImputer().fit(X)
    out = imputer.transform(X)
    donor = (X["sperm_source"] == "donor").to_numpy()
    assert donor.any()
    assert np.allclose(out.loc[donor, "sperm_concentration_m_ml"], imputer.sperm_donor_["sperm_concentration_m_ml"])
    assert (out.loc[donor, "male_age"] == DONOR_REFERENCE_AGE).all()
    assert imputer.sperm_donor_["sperm_concentration_m_ml"] > imputer.sperm_male_factor_["sperm_concentration_m_ml"]


def test_imputed_amh_falls_with_age(clean):
    X, _ = features_and_target(clean)
    imputer = DomainImputer().fit(X)
    probe = X.head(2).copy()
    probe[["amh_ng_ml", "fsh_iu_l"]] = np.nan
    probe["female_age"] = [28, 43]
    out = imputer.transform(probe)
    assert out["amh_ng_ml"].iloc[0] > out["amh_ng_ml"].iloc[1]


def test_imputer_uses_fsh_when_available(clean):
    X, _ = features_and_target(clean)
    imputer = DomainImputer().fit(X)
    probe = X.head(2).copy()
    probe["amh_ng_ml"] = np.nan
    probe["female_age"] = 35
    probe["fsh_iu_l"] = [5.0, 16.0]
    out = imputer.transform(probe)
    assert out["amh_ng_ml"].iloc[0] > out["amh_ng_ml"].iloc[1]


def test_feature_engineer_is_stable_for_single_rows(clean):
    X, _ = features_and_target(clean)
    imputed = DomainImputer().fit(X).transform(X)
    engineer = FeatureEngineer().fit(imputed)
    one = engineer.transform(imputed.head(1))
    assert list(one.columns) == engineer.feature_names_
    assert set(engineer.feature_names_) <= set(DISPLAY_GROUPS)
    assert one.filter(like="cause_").sum(axis=1).iloc[0] == 1

"""Shared fixtures: a small register and a quickly trained bundle."""
from __future__ import annotations

import pytest

from ferticast.config import Config, load_config
from ferticast.data.cleaning import clean_register
from ferticast.data.simulate import simulate_register


@pytest.fixture(scope="session")
def raw_register():
    return simulate_register(n_cycles=6000, n_clinics=20, seed=11)


@pytest.fixture(scope="session")
def clean(raw_register):
    return clean_register(raw_register)[0]


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory, raw_register):
    """Train a miniature model end to end in a temporary workspace."""
    import joblib

    from ferticast.models.train import run_training

    root = tmp_path_factory.mktemp("workspace")
    cfg = dict(load_config())
    cfg["data"] = {**cfg["data"], "raw_path": str(root / "raw.csv"), "processed_dir": str(root / "processed")}
    cfg["training"] = {**cfg["training"], "search_iterations": 2, "cv_folds": 2, "max_boost_rounds": 80, "search_sample_size": 2000}
    cfg["evaluation"] = {**cfg["evaluation"], "bootstrap_rounds": 20}
    cfg["artifacts"] = {"model_dir": str(root / "models"), "report_dir": str(root / "reports"), "figure_dir": str(root / "figures")}
    raw_register.to_csv(root / "raw.csv", index=False)
    report = run_training(Config(cfg))
    return joblib.load(root / "models" / "ferticast_bundle.joblib"), report

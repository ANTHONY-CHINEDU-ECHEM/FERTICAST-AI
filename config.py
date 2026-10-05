"""Configuration loading with dotted attribute access."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FERTICAST_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dictionary that also exposes keys as attributes, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration file."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))


def resolve(path: str | Path) -> Path:
    """Resolve a repository relative path to an absolute path."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

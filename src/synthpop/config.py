"""Config loading (YAML -> validated pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models.config import (
    AgeMixtureSpec,
    CategoricalSpec,
    ConditionalSpec,
    PopulationConfig,
    VariableSpec,
)


def _wrap_variable(entry: dict[str, Any]) -> dict[str, Any]:
    """YAML variable entries are flat ({name, type, ...}); wrap the spec part."""
    name = entry["name"]
    spec_data = {k: v for k, v in entry.items() if k != "name"}
    return {"name": name, "spec": spec_data}


def load_population_config(path: str | Path) -> PopulationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw["variables"] = [_wrap_variable(v) for v in raw.get("variables", [])]
    return PopulationConfig.model_validate(raw)


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

"""Scenario loading: YAML/JSON files -> validated, immutable Scenario models."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..models.scenario import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a YAML or JSON file.

    Accepted shapes:
    - a bare list of scenario objects
    - a mapping with a top-level ``scenarios`` list
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix in (".json",):
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of scenarios")
    return [Scenario.model_validate(s) for s in data]

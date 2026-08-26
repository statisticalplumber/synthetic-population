#!/usr/bin/env python
"""Export JSON schemas from pydantic models (single source of truth).

schemas/*.json are generated artifacts — do not edit them by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthpop.models.persona import LatentAttributes, Persona, PersonaSkeleton  # noqa: E402
from synthpop.models.scenario import Scenario                                   # noqa: E402
from synthpop.models.simulation import EnsemblePrediction, SimulationResult     # noqa: E402

PAIRS = [
    ("persona.schema.json", [Persona, PersonaSkeleton, LatentAttributes]),
    ("scenario.schema.json", [Scenario]),
    ("simulation.schema.json", [SimulationResult, EnsemblePrediction]),
]


def main() -> int:
    out_dir = ROOT / "schemas"
    out_dir.mkdir(exist_ok=True)
    for filename, models in PAIRS:
        defs: dict[str, dict] = {}
        for m in models:
            schema = m.model_json_schema()
            # merge $defs from every model into one document
            for k, v in schema.get("$defs", {}).items():
                defs.setdefault(k, v)
            schema.pop("$defs", None)
            defs[m.__name__] = schema
        doc = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": filename,
            "$defs": defs,
        }
        path = out_dir / filename
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

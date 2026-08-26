"""Deterministic/probabilistic population sampler.

Design principles:
- The LLM never invents the population. Distributions come from
  ``PopulationConfig`` only.
- Fully reproducible: same (config, seed, n) => identical output.
- Vectorized with numpy (one draw per variable, not per row).
- Every row is validated into a frozen ``PersonaSkeleton`` (pydantic).

Scaling note: this is O(variables * n) in numpy — fine up to ~10M rows.
For 100M+ consider chunked streaming; not needed for current scale plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.config import (
    AgeMixtureSpec,
    CategoricalSpec,
    ConditionalSpec,
    PopulationConfig,
)
from ..models.persona import PersonaSkeleton
from ..models.provenance import GenerationMetadata


class PopulationSampler:
    """Samples demographic skeletons from a validated PopulationConfig."""

    def __init__(self, config: PopulationConfig, seed: int | None = None):
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    # per-variable draws (vectorized over n rows)
    # ------------------------------------------------------------------ #

    def _draw_categorical(self, spec: CategoricalSpec, n: int) -> np.ndarray:
        idx = self.rng.choice(len(spec.categories), size=n, p=spec.probabilities)
        return np.asarray(spec.categories, dtype=object)[idx]

    def _draw_conditional(self, spec: ConditionalSpec, n: int, rows: dict[str, np.ndarray]) -> np.ndarray:
        """First-matching-rule-wins; rows per rule are drawn in one vectorized call."""
        cats = list(spec.categories)
        k = len(cats)
        out = np.empty(n, dtype=object)
        assigned = np.zeros(n, dtype=bool)

        def draw(vec: list[float], mask: np.ndarray) -> None:
            m = int(mask.sum())
            if m == 0:
                return
            out[mask] = np.asarray(cats, dtype=object)[self.rng.choice(k, size=m, p=vec)]

        for rule in spec.rules:
            mask = np.ones(n, dtype=bool)
            for var, vals in rule.when.items():
                mask &= np.isin(rows[var], np.asarray(vals, dtype=object))
            open_rows = mask & ~assigned
            draw([rule.distribution.get(c, 0.0) for c in cats], open_rows)
            assigned |= open_rows

        # rows never matched by any rule keep the default distribution
        draw([spec.default.get(c, 0.0) for c in cats], ~assigned)
        return out

    def _draw_age_mixture(self, spec: AgeMixtureSpec, n: int, rows: dict[str, np.ndarray]) -> None:
        band_idx = self.rng.choice(len(spec.bands), size=n, p=[b.weight for b in spec.bands])
        ages = np.empty(n, dtype=np.int64)
        bands = np.empty(n, dtype=object)
        for bi, b in enumerate(spec.bands):
            m = band_idx == bi
            if m.any():
                ages[m] = self.rng.integers(b.lo, b.hi + 1, size=int(m.sum()))
                bands[m] = f"{b.lo}-{b.hi}"
        rows["age"] = ages
        rows["age_band"] = bands

    # ------------------------------------------------------------------ #

    def sample(self, n: int) -> list[PersonaSkeleton]:
        cfg = self.config
        rows: dict[str, np.ndarray] = {}

        for var in cfg.variables:
            spec = var.spec
            if isinstance(spec, CategoricalSpec):
                rows[var.name] = self._draw_categorical(spec, n)
            elif isinstance(spec, ConditionalSpec):
                rows[var.name] = self._draw_conditional(spec, n, rows)
            elif isinstance(spec, AgeMixtureSpec):
                self._draw_age_mixture(spec, n, rows)
            else:  # pragma: no cover - pydantic union guarantees the above
                raise TypeError(f"unknown variable spec: {type(spec)}")

        config_version = cfg.config_hash()
        skeletons: list[PersonaSkeleton] = []
        for i in range(n):
            row: dict[str, Any] = {
                "persona_id": f"{cfg.country_code}_{i:06d}",
                "country": cfg.country,
                "data_label": cfg.data_label,
            }
            for name in ("emirate", "city", "urban_rural", "age", "age_band",
                         "gender", "marital_status", "education",
                         "employment_status", "occupation_group",
                         "income_band", "household_size", "housing_status"):
                row[name] = _py(rows[name][i])
            row["provenance"] = GenerationMetadata(
                pipeline_stage="skeleton",
                provider="deterministic_sampler",
                model=None,
                prompt_version=None,
                population_config_version=config_version,
                seed=self.seed,
                data_label=cfg.data_label,
            )
            skeletons.append(PersonaSkeleton.model_validate(row))
        return skeletons


def _py(v: Any) -> Any:
    """Convert numpy scalars to plain python (int/str) for pydantic."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.str_):
        return str(v)
    return v

"""Config models for population sampling.

The population config is the *only* place where distributions live. The LLM is
never allowed to invent the population distribution; the sampler consumes this
config deterministically.

YAML files are validated into these models before use.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance import DataLabel

PROB_SUM_TOL = 1e-6

# Fields the persona skeleton must end up with. `country` comes from the
# config itself; everything else must be produced by exactly one variable spec.
REQUIRED_SAMPLED_FIELDS = frozenset(
    {
        "emirate", "city", "urban_rural", "age", "age_band", "gender",
        "marital_status", "education", "employment_status",
        "occupation_group", "income_band", "household_size", "housing_status",
    }
)


def _check_prob_map(dist: dict[str | int, float], where: str) -> None:
    if not dist:
        raise ValueError(f"{where}: empty distribution")
    keys = list(dist.keys())
    if len(keys) != len(set(keys)):
        raise ValueError(f"{where}: duplicate categories in distribution")
    if any(p < 0 for p in dist.values()):
        raise ValueError(f"{where}: negative probability")
    total = sum(dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{where}: probabilities sum to {total}, expected 1.0")


class CategoricalSpec(BaseModel):
    """Flat categorical distribution."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical"] = "categorical"
    categories: list[str | int] = Field(min_length=2)
    probabilities: list[float]

    @model_validator(mode="after")
    def _check(self) -> "CategoricalSpec":
        cats = list(self.categories)
        if len(cats) != len(set(cats)):
            raise ValueError("categorical: duplicate categories")
        if len(cats) != len(self.probabilities):
            raise ValueError(
                f"categorical: {len(cats)} categories but "
                f"{len(self.probabilities)} probabilities"
            )
        _check_prob_map(dict(zip(map(str, cats), self.probabilities)), "categorical")
        return self


class ConditionalRule(BaseModel):
    """Rule: if every `when` variable is in the given value set, use
    `distribution`. First matching rule wins (evaluated in order)."""

    model_config = ConfigDict(extra="forbid")

    when: dict[str, list[str | int]]
    distribution: dict[str | int, float]

    @model_validator(mode="after")
    def _check(self) -> "ConditionalRule":
        if not self.when:
            raise ValueError("conditional rule: 'when' must not be empty")
        _check_prob_map(self.distribution, "conditional rule")
        return self


class ConditionalSpec(BaseModel):
    """Conditional distribution with ordered rules over previously sampled
    variables, plus a default for unmatched rows."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["conditional"] = "conditional"
    categories: list[str | int] = Field(min_length=2)
    rules: list[ConditionalRule] = Field(default_factory=list)
    default: dict[str | int, float]

    @model_validator(mode="after")
    def _check(self) -> "ConditionalSpec":
        cats = set(map(str, self.categories))
        _check_prob_map(self.default, "conditional default")
        for i, rule in enumerate(self.rules):
            for var, vals in rule.when.items():
                if not vals:
                    raise ValueError(f"conditional rule {i}: empty value set for {var}")
            for k in rule.distribution:
                if str(k) not in cats:
                    raise ValueError(
                        f"conditional rule {i}: category {k!r} not in {sorted(cats)}"
                    )
        for k in self.default:
            if str(k) not in cats:
                raise ValueError(
                    f"conditional default: category {k!r} not in {sorted(cats)}"
                )
        return self


class AgeBandSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lo: int
    hi: int
    weight: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "AgeBandSpec":
        if self.lo > self.hi:
            raise ValueError(f"age band {self.lo}-{self.hi}: lo > hi")
        return self


class AgeMixtureSpec(BaseModel):
    """Age sampled as a mixture of uniform integer bands.

    Produces both `age` (int) and `age_band` (e.g. '25-34') fields.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["age_mixture"] = "age_mixture"
    bands: list[AgeBandSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "AgeMixtureSpec":
        total = sum(b.weight for b in self.bands)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"age_mixture: band weights sum to {total}, expected 1.0")
        for a, b in zip(self.bands, self.bands[1:]):
            if a.hi + 1 != b.lo:
                raise ValueError(
                    f"age_mixture: bands must be contiguous and ordered, "
                    f"got ...{a.lo}-{a.hi} then {b.lo}-{b.hi}..."
                )
        return self


class VariableSpec(BaseModel):
    """One variable in the sampling DAG: name + typed distribution spec."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    spec: Annotated[
        Union[CategoricalSpec, ConditionalSpec, AgeMixtureSpec],
        Field(discriminator="type"),
    ]


class PopulationConfig(BaseModel):
    """Full configuration for one population sampling run."""

    model_config = ConfigDict(extra="forbid")

    config_version: str
    country: str
    country_code: str = Field(pattern=r"^[A-Z]{2,3}$")
    data_label: DataLabel = "synthetic_mock"
    source_note: str = ""
    sample_size: int = Field(ge=1)
    seed: int
    variables: list[VariableSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "PopulationConfig":
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("duplicate variable names in config")
        known: set[str] = set()
        for v in self.variables:
            spec = v.spec
            if isinstance(spec, ConditionalSpec):
                for rule in spec.rules:
                    for parent in rule.when:
                        if parent not in known:
                            raise ValueError(
                                f"variable {v.name!r} references {parent!r} "
                                "which is not defined earlier in 'variables'"
                            )
            known.add(v.name)
            if isinstance(spec, AgeMixtureSpec):
                known.add("age_band")
        produced = set(names) | {"age_band"}
        missing = REQUIRED_SAMPLED_FIELDS - produced
        if missing:
            raise ValueError(f"config does not produce required fields: {sorted(missing)}")
        return self

    # ---- derived helpers -------------------------------------------------

    def config_hash(self) -> str:
        """Stable hash of the full config (used as population_config_version)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def variable(self, name: str) -> VariableSpec:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def categorical_categories(self, name: str) -> list[str | int]:
        """Categories of a categorical/conditional variable (for validation)."""
        spec = self.variable(name).spec
        if isinstance(spec, (CategoricalSpec, ConditionalSpec)):
            return list(spec.categories)
        raise ValueError(f"{name} is not categorical")

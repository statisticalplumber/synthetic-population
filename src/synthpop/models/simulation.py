"""Layer 3 — behavioral observation models.

Simulations are generated independently of persona creation. Outcomes are
probabilistic, never forced categorical certainty: a ``SimulationResult``
carries a full probability distribution over the scenario options, a
confidence score, and a small set of *declared* behavioral factors (labels +
direction + strength) — NOT a chain of reasoning.

Ensemble-readiness: ``SimulationResult`` is one independent model prediction.
Multiple predictions for the same ``(persona_id, scenario_id, scenario_version)``
triple (see ``ensemble_key()``) can later be aggregated into mean probability,
variance, entropy, and model disagreement without schema changes. Agreement
across models is NOT treated as ground truth.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .persona import LatentAttributes
from .provenance import SimulationProvenance

PROB_SUM_TOL = 0.01


class BehavioralFactor(BaseModel):
    """One declared behavioral factor behind a simulation.

    ``factor`` should be a known persona latent name (see
    ``LatentAttributes.names()``); ``direction`` says which way it pushed the
    outcome; ``strength`` is its magnitude in [0, 1]. This is a concise,
    auditable label — deliberately not free-text reasoning.

    Deliberately permissive at this layer (raw model output): the
    deterministic validator and ``SimulationResult`` enforce that the factor
    name is known and strength is in [0, 1].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    factor: str = Field(min_length=1)
    direction: Literal["positive", "negative", "neutral"]
    strength: float


class SimulationOutput(BaseModel):
    """Raw model response for one (persona, scenario) simulation.

    This is the schema handed to the LLM (or produced by the offline mock).
    The model must NOT be asked for reasoning or chain-of-thought — only the
    distribution, a confidence score, and a small set of factor labels.

    Deliberately PERMISSIVE on value ranges: a model may return out-of-range
    or non-normalized numbers, and the deterministic validator
    (``synthpop.simulation.validation.validate_simulation_output``) is the
    gate that rejects them (as retryable malformed output). Strictness lives
    on ``SimulationResult``, which is only ever built from validated output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    probabilities: dict[str, float] = Field(min_length=2)
    confidence: float
    behavioral_factors: list[BehavioralFactor] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Probabilistic behavioral outcome for one (persona, scenario) pair.

    ``choice`` is derived (argmax of the distribution) — it is a convenience
    label, the distribution is the payload. One record = one model's
    prediction; ensemble aggregation happens downstream over
    ``ensemble_key()`` groups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona_id: str
    scenario_id: str
    scenario_version: str
    choice: str
    probabilities: dict[str, float] = Field(min_length=2)
    confidence: float = Field(ge=0.0, le=1.0)
    behavioral_factors: list[BehavioralFactor] = Field(default_factory=list)
    provenance: SimulationProvenance

    @model_validator(mode="after")
    def _check(self) -> "SimulationResult":
        for k, v in self.probabilities.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"probability {k}={v} outside [0,1]")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probabilities sum to {total:.4f}, expected ~1.0")
        # The reported choice must be the argmax of the distribution.
        top = max(self.probabilities, key=self.probabilities.get)
        if self.choice != top:
            raise ValueError(
                f"choice {self.choice!r} is not argmax of probabilities (top={top!r})"
            )
        # Provenance must agree with the record identity.
        if self.provenance.scenario_id != self.scenario_id:
            raise ValueError(
                f"provenance scenario_id {self.provenance.scenario_id!r} != "
                f"record scenario_id {self.scenario_id!r}"
            )
        if self.provenance.scenario_version != self.scenario_version:
            raise ValueError(
                f"provenance scenario_version {self.provenance.scenario_version!r} != "
                f"record scenario_version {self.scenario_version!r}"
            )
        # Strictness for the final record: factors must be known latents with
        # bounded strength (raw SimulationOutput is permissive).
        known = set(LatentAttributes.names())
        for f in self.behavioral_factors:
            if f.factor not in known:
                raise ValueError(f"unknown behavioral factor {f.factor!r}")
            if not (0.0 <= f.strength <= 1.0):
                raise ValueError(f"behavioral factor strength {f.strength} outside [0,1]")
        return self

    @property
    def record_id(self) -> str:
        """Stable idempotency key for this (persona, scenario) record."""
        return f"{self.persona_id}|{self.scenario_id}@{self.scenario_version}"

    def ensemble_key(self) -> tuple[str, str, str]:
        """Grouping key for multi-model (ensemble) aggregation."""
        return (self.persona_id, self.scenario_id, self.scenario_version)

    def entropy(self, base: float = 2.0) -> float:
        """Shannon entropy of the option distribution (bits for base=2)."""
        h = 0.0
        for v in self.probabilities.values():
            if v > 0:
                h -= v * math.log(v, base)
        return h


class EnsemblePrediction(BaseModel):
    """Multi-model simulation output with disagreement score.

    `model_disagreement` = std-dev of per-model probabilities for the chosen
    option. High disagreement => uncertain record; do NOT read agreement as
    ground-truth human behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona_id: str
    scenario_id: str
    scenario_version: str
    choice: str
    predictions: dict[str, float]  # model name -> P(choice)
    ensemble_probability: float = Field(ge=0.0, le=1.0)
    model_disagreement: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_disagreement(self) -> "EnsemblePrediction":
        if len(self.predictions) < 2:
            raise ValueError("ensemble requires at least 2 models")
        vals = list(self.predictions.values())
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if abs(std - self.model_disagreement) > 0.01:
            raise ValueError(
                f"model_disagreement {self.model_disagreement} inconsistent "
                f"with predictions (std={std:.4f})"
            )
        if abs(self.ensemble_probability - mean) > 0.01:
            raise ValueError(
                f"ensemble_probability {self.ensemble_probability} inconsistent "
                f"with mean of predictions ({mean:.4f})"
            )
        return self

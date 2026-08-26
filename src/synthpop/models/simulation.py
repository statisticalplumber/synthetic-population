"""Layer 3 — behavioral observation models.

Simulations are generated independently of persona creation. Outcomes are
probabilistic, never forced categorical certainty.

Ensemble models support multi-model (monoculture-mitigation) simulation:
agreement across models is NOT treated as ground truth.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance import SimulationProvenance


class SimulationResult(BaseModel):
    """Probabilistic behavioral outcome for one (persona, scenario) pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona_id: str
    scenario_id: str
    scenario_version: str
    choice: str
    probabilities: dict[str, float] = Field(min_length=2)
    confidence: float = Field(ge=0.0, le=1.0)
    behavioral_factors: list[str] = Field(default_factory=list)
    provenance: SimulationProvenance

    @model_validator(mode="after")
    def _check_probabilities(self) -> "SimulationResult":
        for k, v in self.probabilities.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"probability {k}={v} outside [0,1]")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"probabilities sum to {total:.4f}, expected ~1.0")
        # The reported choice must be the argmax of the distribution.
        top = max(self.probabilities, key=self.probabilities.get)
        if self.choice != top:
            raise ValueError(
                f"choice {self.choice!r} is not argmax of probabilities (top={top!r})"
            )
        return self


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

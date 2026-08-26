"""Provenance metadata.

Every generated record carries provenance so that any problematic generation
can later be traced, reproduced, and selectively regenerated.

Provenance is immutable once written (frozen model).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

DataLabel = Literal["synthetic_mock", "synthetic_real_stats", "real"]
PipelineStage = Literal["skeleton", "persona", "simulation", "enrichment"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GenerationMetadata(BaseModel):
    """Provenance for a single generated record (skeleton or enriched persona)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_stage: PipelineStage
    provider: str = "deterministic_sampler"
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    population_config_version: str
    seed: int
    created_at: str = Field(default_factory=utcnow_iso)
    data_label: DataLabel = "synthetic_mock"


class SimulationProvenance(BaseModel):
    """Provenance for a behavioral simulation record (Layer 3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: Optional[str]
    simulator_prompt_version: str
    scenario_id: str
    scenario_version: str
    persona_version: str
    model_params: dict[str, float | int | str] = Field(default_factory=dict)
    seed: int
    created_at: str = Field(default_factory=utcnow_iso)
    data_label: DataLabel = "synthetic_mock"

"""Persona models.

Layer 1 — PersonaSkeleton: observable population variables. Immutable (frozen);
produced by the deterministic sampler, NOT by an LLM.

Layer 2 — LatentAttributes: normalized 0-1 behavioral latents. Produced by
LLM enrichment (or the mock provider during development).

Persona — Layer 1 + Layer 2 + enrichment provenance.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .provenance import DataLabel, GenerationMetadata

PERSONA_ID_RE = re.compile(r"^[A-Z]{2,3}_\d{6}$")


class PersonaSkeleton(BaseModel):
    """Layer 1: observable demographic variables (statistically sampled)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona_id: str
    country: str
    emirate: str
    city: str
    urban_rural: Literal["urban", "rural"]
    age: int = Field(ge=18, le=80)
    age_band: str
    gender: Literal["female", "male"]
    marital_status: str
    education: str
    employment_status: str
    occupation_group: str
    income_band: str
    household_size: int = Field(ge=1, le=15)
    housing_status: str
    data_label: DataLabel = "synthetic_mock"
    provenance: GenerationMetadata

    @field_validator("persona_id")
    @classmethod
    def _check_persona_id(cls, v: str) -> str:
        if not PERSONA_ID_RE.match(v):
            raise ValueError(
                f"persona_id {v!r} does not match {PERSONA_ID_RE.pattern}"
            )
        return v

    @field_validator("age_band")
    @classmethod
    def _check_age_band(cls, v: str) -> str:
        m = re.fullmatch(r"(\d+)-(\d+)", v)
        if not m or int(m.group(1)) >= int(m.group(2)):
            raise ValueError(f"age_band {v!r} must look like '18-24'")
        return v

    @model_validator(mode="after")
    def _age_in_band(self) -> "PersonaSkeleton":
        lo, hi = (int(x) for x in self.age_band.split("-"))
        if not (lo <= self.age <= hi):
            raise ValueError(
                f"age {self.age} outside age_band {self.age_band} ({lo}-{hi})"
            )
        return self

    def demographic_key(self) -> tuple:
        """Full demographic signature, used for duplicate detection."""
        return (
            self.country, self.emirate, self.city, self.urban_rural,
            self.age, self.gender, self.marital_status, self.education,
            self.employment_status, self.occupation_group,
            self.income_band, self.household_size, self.housing_status,
        )


class LatentAttributes(BaseModel):
    """Layer 2: latent behavioral variables, normalized to [0, 1]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_sensitivity: float = Field(ge=0.0, le=1.0)
    risk_tolerance: float = Field(ge=0.0, le=1.0)
    brand_loyalty: float = Field(ge=0.0, le=1.0)
    technology_affinity: float = Field(ge=0.0, le=1.0)
    novelty_seeking: float = Field(ge=0.0, le=1.0)
    social_influence_sensitivity: float = Field(ge=0.0, le=1.0)
    convenience_preference: float = Field(ge=0.0, le=1.0)
    environmental_concern: float = Field(ge=0.0, le=1.0)
    status_orientation: float = Field(ge=0.0, le=1.0)
    financial_conservatism: float = Field(ge=0.0, le=1.0)
    impulsivity: float = Field(ge=0.0, le=1.0)
    trust_propensity: float = Field(ge=0.0, le=1.0)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.model_fields.keys())


class Persona(PersonaSkeleton):
    """Full persona: immutable demographics + latent attributes + provenance."""

    latent: Optional[LatentAttributes] = None
    enrichment: Optional[GenerationMetadata] = None

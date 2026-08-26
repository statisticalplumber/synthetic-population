"""Scenario model (kept separate from personas by design).

Scenarios are versioned and immutable: a scenario is identified by
``(scenario_id, scenario_version)`` and never mutated in place — a change is
a new version, so historical simulation results remain reproducible.

The model is intentionally generic (no consumer-only assumptions):
``category`` is a free-form label, ``price``/``currency`` are optional
numeric attributes, and arbitrary structured detail goes in ``context``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCENARIO_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SCENARIO_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class Scenario(BaseModel):
    """A behavioral scenario presented to personas during simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    scenario_version: str = "v1"
    category: str
    question: str = Field(min_length=1)
    description: Optional[str] = None
    options: list[str] = Field(min_length=2)
    # Optional numeric/structured attributes (generic, not consumer-specific).
    price: Optional[float] = Field(default=None, ge=0.0)
    currency: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("scenario_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not SCENARIO_ID_RE.match(v):
            raise ValueError(f"scenario_id {v!r} must match {SCENARIO_ID_RE.pattern}")
        return v

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("scenario options must be unique")
        if any(not o.strip() for o in v):
            raise ValueError("scenario options must be non-empty strings")
        return v

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not CURRENCY_RE.match(v):
            raise ValueError(f"currency {v!r} must be a 3-letter code (e.g. AED)")
        return v

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("scenario tags must be unique")
        return v

    @model_validator(mode="after")
    def _version_format(self) -> "Scenario":
        if not SCENARIO_VERSION_RE.fullmatch(self.scenario_version):
            raise ValueError(
                f"scenario_version {self.scenario_version!r} must look like 'v1' or 'v1.2'"
            )
        return self

    def full_id(self) -> str:
        return f"{self.scenario_id}@{self.scenario_version}"

"""Scenario model (kept separate from personas by design)."""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCENARIO_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class Scenario(BaseModel):
    """A behavioral scenario presented to personas during simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    version: str = "v1"
    category: str
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    context: dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None

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

    @model_validator(mode="after")
    def _version_format(self) -> "Scenario":
        if not re.fullmatch(r"v\d+(\.\d+)*", self.version):
            raise ValueError(f"version {self.version!r} must look like 'v1' or 'v1.2'")
        return self

    def full_id(self) -> str:
        return f"{self.scenario_id}@{self.version}"

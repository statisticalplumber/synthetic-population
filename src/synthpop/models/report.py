"""Validation / quality report models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .provenance import utcnow_iso


class VariableCheck(BaseModel):
    """Result of one distribution check (marginal, conditional, or age bins).

    Pass/fail is decided by a chi-squared goodness-of-fit test against the
    target distribution (sample-size aware). JS divergence and max abs diff
    are reported as informational distance metrics.
    """

    model_config = ConfigDict(frozen=True)

    variable: str
    kind: Literal["marginal", "conditional", "age_bins"]
    n: int
    target: dict[str, float]
    observed: dict[str, float]
    js_divergence: float
    max_abs_diff: float
    chi2_stat: float = 0.0
    chi2_pvalue: float = 1.0
    status: Literal["pass", "fail", "insufficient_data"]
    detail: str = ""


class ValidationReport(BaseModel):
    """Deterministic validation report for a population sample."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_version: str
    config_hash: str
    seed: int
    n: int
    checks: list[VariableCheck]
    duplicate_rate: float
    missing_rate: float
    overall: Literal["pass", "fail"]
    created_at: str = Field(default_factory=utcnow_iso)
    thresholds: dict[str, float] = Field(default_factory=dict)

    @property
    def failures(self) -> list[VariableCheck]:
        return [c for c in self.checks if c.status == "fail"]

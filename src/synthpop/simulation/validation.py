"""Deterministic (no-LLM) validation of simulation outputs.

Every model result is checked in ordinary Python BEFORE it is accepted as a
``SimulationResult``. Failures raise ``SimulationValidationError`` (a
retryable, transient malformed-output condition at the batch level).
"""

from __future__ import annotations

from ..models.persona import LatentAttributes, Persona
from ..models.scenario import Scenario
from ..models.simulation import PROB_SUM_TOL, SimulationOutput, SimulationResult


class SimulationValidationError(ValueError):
    """A model output failed deterministic validation."""


def validate_simulation_output(
    out: SimulationOutput,
    persona: Persona,
    scenario: Scenario,
) -> None:
    """Validate a raw model output against the (persona, scenario) pair.

    Raises SimulationValidationError on the first violated constraint:
    - all scenario options present, no unexpected options
    - probabilities within [0, 1] and summing to ~1 (within PROB_SUM_TOL)
    - confidence within [0, 1]
    - factor strengths within [0, 1]; factor names are known latent names
    """
    expected = set(scenario.options)
    got = set(out.probabilities)

    missing = expected - got
    if missing:
        raise SimulationValidationError(
            f"missing options {sorted(missing)}; expected exactly {sorted(expected)}"
        )
    unexpected = got - expected
    if unexpected:
        raise SimulationValidationError(
            f"unexpected options {sorted(unexpected)}; expected exactly {sorted(expected)}"
        )

    for opt, p in out.probabilities.items():
        if not (0.0 <= p <= 1.0):
            raise SimulationValidationError(
                f"probability {opt}={p} outside [0,1]"
            )
    total = sum(out.probabilities.values())
    if abs(total - 1.0) > PROB_SUM_TOL:
        raise SimulationValidationError(
            f"probabilities sum to {total:.4f}, expected ~1.0 (tol {PROB_SUM_TOL})"
        )

    if not (0.0 <= out.confidence <= 1.0):
        raise SimulationValidationError(
            f"confidence {out.confidence} outside [0,1]"
        )

    known_factors = set(LatentAttributes.names())
    for f in out.behavioral_factors:
        if not (0.0 <= f.strength <= 1.0):
            raise SimulationValidationError(
                f"factor {f.factor!r} strength {f.strength} outside [0,1]"
            )
        if f.factor not in known_factors:
            raise SimulationValidationError(
                f"unknown factor {f.factor!r}; known latent names: "
                f"{sorted(known_factors)}"
            )


def validate_simulation_result(
    result: SimulationResult,
    persona: Persona,
    scenario: Scenario,
) -> None:
    """Cross-check a built result against its inputs (ID consistency)."""
    if result.persona_id != persona.persona_id:
        raise SimulationValidationError(
            f"persona_id mismatch: result {result.persona_id!r} != "
            f"input {persona.persona_id!r}"
        )
    if result.scenario_id != scenario.scenario_id:
        raise SimulationValidationError(
            f"scenario_id mismatch: result {result.scenario_id!r} != "
            f"input {scenario.scenario_id!r}"
        )
    if result.scenario_version != scenario.scenario_version:
        raise SimulationValidationError(
            f"scenario_version mismatch: result {result.scenario_version!r} != "
            f"input {scenario.scenario_version!r}"
        )
    if result.provenance.scenario_id != scenario.scenario_id:
        raise SimulationValidationError(
            f"provenance scenario_id {result.provenance.scenario_id!r} != "
            f"{scenario.scenario_id!r}"
        )
    if result.provenance.scenario_version != scenario.scenario_version:
        raise SimulationValidationError(
            f"provenance scenario_version {result.provenance.scenario_version!r} != "
            f"{scenario.scenario_version!r}"
        )

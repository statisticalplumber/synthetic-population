"""Deterministic offline simulation (mock path).

``mock_simulation_output`` computes a valid ``SimulationOutput`` directly from
(persona, scenario, seed) — no LLM, no network. Used by the simulator when
the configured provider is the offline ``MockProvider``, so mock runs are:

- deterministic (same inputs => same output),
- valid by construction (options, ranges, sum-to-1),
- persona-sensitive (option scores are driven by the persona's latents),
- cheap (500-record runs finish in well under a second).

This is a development/testing stand-in, NOT a behavioral model: it produces
plausible variance for pipeline, metrics, and checkpoint testing only.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..models.persona import LatentAttributes, Persona
from ..models.scenario import Scenario
from ..models.simulation import BehavioralFactor, SimulationOutput

MOCK_MODEL_LABEL = "mock/deterministic"


def mock_simulation_output(
    persona: Persona, scenario: Scenario, seed: int = 0
) -> SimulationOutput:
    """Deterministic probability distribution over ``scenario.options``.

    Each option gets a score from a deterministic subset of the persona's
    latent attributes (option-specific pseudo-random selection + signs),
    then a temperature-scaled softmax produces the distribution. Confidence
    is the peak probability; factors are the persona's strongest latents.
    """
    names = LatentAttributes.names()
    latent = persona.latent
    vals = {
        n: (getattr(latent, n) if latent is not None else 0.5) for n in names
    }

    scores: dict[str, float] = {}
    for opt in scenario.options:
        h = int.from_bytes(
            hashlib.sha256(
                f"{seed}|{scenario.full_id()}|{opt}".encode()
            ).digest()[:8],
            "big",
        )
        r = np.random.default_rng(h)
        idx = r.choice(len(names), size=3, replace=False)
        signs = r.choice([-1.0, 1.0], size=3)
        scores[opt] = float(
            sum(signs[i] * (2.0 * (vals[names[j]] - 0.5)) for i, j in enumerate(idx))
        )

    # temperature-scaled softmax
    temp = 2.0
    exps = {k: math.exp(v / temp) for k, v in scores.items()}
    z = sum(exps.values())
    probs = {k: round(v / z, 4) for k, v in exps.items()}

    top = max(probs.values())
    spread = top - 1.0 / len(probs)  # 0 = uniform, ~1 = certain
    confidence = round(min(1.0, 0.5 + spread), 4)

    # strongest latents (furthest from 0.5) as declared factors
    ranked = sorted(names, key=lambda n: -abs(vals[n] - 0.5))[:3]
    factors = [
        BehavioralFactor(
            factor=n,
            direction="positive" if vals[n] >= 0.5 else "negative",
            strength=round(abs(vals[n] - 0.5) * 2.0, 4),
        )
        for n in ranked
    ]

    return SimulationOutput(
        probabilities=probs, confidence=confidence, behavioral_factors=factors
    )

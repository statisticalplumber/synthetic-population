"""Versioned prompts for behavioral scenario simulation (Layer 3).

The prompt explicitly asks ONLY for:
- a probability for every option,
- a confidence score,
- a small set of labeled behavioral factors (name / direction / strength).

It explicitly does NOT ask for chain-of-thought, explanations, or
step-by-step reasoning — factor labels are concise, auditable declarations.
"""

from __future__ import annotations

import json

from ..models.persona import LatentAttributes, Persona
from ..models.scenario import Scenario

SIM_PROMPT_VERSION = "sim_v1"

SIM_SYSTEM_PROMPT = (
    "You simulate the decision of a synthetic persona in a behavioral scenario. "
    "Return ONLY the JSON object defined by the schema — no prose, no markdown. "
    "Requirements: give a probability for EVERY option (all probabilities in [0,1] "
    "and summing to 1); give an overall confidence in [0,1]; give at most 3 "
    "behavioral factors using EXACTLY the provided latent attribute names, each "
    "with direction (positive|negative|neutral) and strength in [0,1]. "
    "Do NOT provide reasoning, explanations, justifications, or step-by-step "
    "thought processes. Do not produce any text outside the JSON object."
)

MAX_FACTORS = 3


def build_simulation_prompt(persona: Persona, scenario: Scenario) -> str:
    """User prompt: persona (demographics + latents) + scenario definition."""
    latent = persona.latent
    payload = {
        "persona_id": persona.persona_id,
        "demographics": {
            "country": persona.country,
            "emirate": persona.emirate,
            "city": persona.city,
            "urban_rural": persona.urban_rural,
            "age": persona.age,
            "gender": persona.gender,
            "marital_status": persona.marital_status,
            "education": persona.education,
            "employment_status": persona.employment_status,
            "occupation_group": persona.occupation_group,
            "income_band": persona.income_band,
            "household_size": persona.household_size,
            "housing_status": persona.housing_status,
        },
        "latent_attributes": (
            {n: getattr(latent, n) for n in LatentAttributes.names()}
            if latent is not None
            else None
        ),
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "scenario_version": scenario.scenario_version,
            "category": scenario.category,
            "question": scenario.question,
            "description": scenario.description,
            "options": scenario.options,
            "price": scenario.price,
            "currency": scenario.currency,
            "context": scenario.context,
            "tags": scenario.tags,
        },
        "instructions": {
            "options_to_score": scenario.options,
            "allowed_factor_names": LatentAttributes.names(),
            "max_factors": MAX_FACTORS,
        },
    }
    return json.dumps(payload, ensure_ascii=False)

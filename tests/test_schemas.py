import pytest
from pydantic import ValidationError

from synthpop.models.persona import LatentAttributes, Persona, PersonaSkeleton
from synthpop.models.provenance import GenerationMetadata, SimulationProvenance
from synthpop.models.scenario import Scenario
from synthpop.models.simulation import EnsemblePrediction, SimulationResult


def make_skeleton(**over) -> PersonaSkeleton:
    base = {
        "persona_id": "AE_000001",
        "country": "UAE",
        "emirate": "Dubai",
        "city": "Dubai City",
        "urban_rural": "urban",
        "age": 34,
        "age_band": "25-34",
        "gender": "female",
        "marital_status": "married",
        "education": "bachelors",
        "employment_status": "employed",
        "occupation_group": "professional",
        "income_band": "upper_middle",
        "household_size": 3,
        "housing_status": "owned",
        "provenance": GenerationMetadata(
            pipeline_stage="skeleton",
            population_config_version="abc",
            seed=42,
        ),
    }
    base.update(over)
    return PersonaSkeleton.model_validate(base)


def test_skeleton_ok():
    s = make_skeleton()
    assert s.persona_id == "AE_000001"


def test_skeleton_bad_persona_id():
    with pytest.raises(ValidationError):
        make_skeleton(persona_id="AE_1")


def test_skeleton_age_outside_band():
    with pytest.raises(ValidationError):
        make_skeleton(age=40, age_band="25-34")


def test_skeleton_is_immutable():
    s = make_skeleton()
    with pytest.raises(ValidationError):
        s.age = 99  # frozen


def test_latent_bounds():
    with pytest.raises(ValidationError):
        LatentAttributes.model_validate(
            {k: 0.5 for k in LatentAttributes.names()} | {"impulsivity": 1.5}
        )


def test_persona_inherits_skeleton():
    s = make_skeleton()
    p = Persona.model_validate(
        {**s.model_dump(), "latent": {k: 0.5 for k in LatentAttributes.names()}}
    )
    assert p.latent.technology_affinity == 0.5
    assert p.persona_id == s.persona_id


def test_scenario_basic():
    sc = Scenario.model_validate(
        {
            "scenario_id": "grocery_subscription_v1",
            "category": "consumer",
            "question": "Would you subscribe to grocery delivery for AED 39/month?",
            "options": ["yes", "no", "maybe"],
        }
    )
    assert sc.version == "v1"


def test_scenario_duplicate_options():
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {"scenario_id": "x", "question": "q", "options": ["a", "a"]}
        )


def sim_provenance() -> SimulationProvenance:
    return SimulationProvenance(
        provider="mock", model=None, simulator_prompt_version="sim_v1",
        scenario_id="grocery_subscription_v1", scenario_version="v1",
        persona_version="abc", seed=1,
    )


def test_simulation_result_ok():
    r = SimulationResult.model_validate(
        {
            "persona_id": "AE_000001",
            "scenario_id": "grocery_subscription_v1",
            "scenario_version": "v1",
            "choice": "yes",
            "probabilities": {"yes": 0.52, "no": 0.31, "maybe": 0.17},
            "confidence": 0.68,
            "behavioral_factors": ["price sensitivity", "convenience preference"],
            "provenance": sim_provenance(),
        }
    )
    assert r.choice == "yes"


def test_simulation_probability_sum():
    with pytest.raises(ValidationError):
        SimulationResult.model_validate(
            {
                "persona_id": "AE_000001", "scenario_id": "s", "scenario_version": "v1",
                "choice": "yes",
                "probabilities": {"yes": 0.9, "no": 0.5},
                "confidence": 0.5,
                "provenance": sim_provenance(),
            }
        )


def test_simulation_choice_must_be_argmax():
    with pytest.raises(ValidationError):
        SimulationResult.model_validate(
            {
                "persona_id": "AE_000001", "scenario_id": "s", "scenario_version": "v1",
                "choice": "no",
                "probabilities": {"yes": 0.6, "no": 0.4},
                "confidence": 0.5,
                "provenance": sim_provenance(),
            }
        )


def test_ensemble_consistency():
    e = EnsemblePrediction.model_validate(
        {
            "persona_id": "AE_000001", "scenario_id": "s", "scenario_version": "v1",
            "choice": "yes",
            "predictions": {"model_a": 0.68, "model_b": 0.72, "model_c": 0.59},
            "ensemble_probability": 0.6633,
            "model_disagreement": 0.055,
        }
    )
    assert abs(e.ensemble_probability - 0.6633) < 1e-3


def test_ensemble_requires_two_models():
    with pytest.raises(ValidationError):
        EnsemblePrediction.model_validate(
            {
                "persona_id": "AE_000001", "scenario_id": "s", "scenario_version": "v1",
                "choice": "yes",
                "predictions": {"model_a": 0.6},
                "ensemble_probability": 0.6,
                "model_disagreement": 0.0,
            }
        )

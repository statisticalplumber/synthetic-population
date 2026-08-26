"""Layer 3 — behavioral scenario simulation tests.

Covers: deterministic mock simulation, probability constraints, malformed /
missing / unexpected options, checkpoint/resume, provider errors,
retryable vs non-retryable classification, nested JSON-schema handling,
provenance, and inference-parameter forwarding.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from synthpop.llm.batch import arun_batch, backoff_delay, run_batch
from synthpop.llm.cost import CostTracker
from synthpop.llm.provider import (
    LLMError,
    MockProvider,
    NonRetryableLLMError,
    OpenAICompatibleProvider,
    RetryableLLMError,
    StructuredResponse,
    build_provider,
    classify_http_status,
    classify_transport_error,
    extract_json_object,
    inline_local_refs,
    prepare_prompt_schema,
)
from synthpop.models.persona import LatentAttributes, Persona
from synthpop.models.provenance import GenerationMetadata
from synthpop.models.scenario import Scenario
from synthpop.models.simulation import (
    PROB_SUM_TOL,
    SimulationOutput,
    SimulationResult,
)
from synthpop.simulation import (
    SIM_PROMPT_VERSION,
    BehavioralSimulator,
    SimulationValidationError,
    compute_simulation_metrics,
    load_scenarios,
    mock_simulation_output,
    pair_key,
    validate_simulation_output,
)
from synthpop.simulation.mock import MOCK_MODEL_LABEL

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_persona(pid: str = "AE_000001", **over) -> Persona:
    base = {
        "persona_id": pid,
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
            population_config_version="pop_v1",
            seed=42,
        ),
        "latent": {
            "price_sensitivity": 0.8,
            "risk_tolerance": 0.55,
            "brand_loyalty": 0.3,
            "technology_affinity": 0.9,
            "novelty_seeking": 0.5,
            "social_influence_sensitivity": 0.5,
            "convenience_preference": 0.7,
            "environmental_concern": 0.4,
            "status_orientation": 0.6,
            "financial_conservatism": 0.35,
            "impulsivity": 0.4,
            "trust_propensity": 0.65,
        },
    }
    base.update(over)
    return Persona.model_validate(base)


def make_scenario(sid: str = "grocery_subscription", **over) -> Scenario:
    base = {
        "scenario_id": sid,
        "scenario_version": "v1",
        "category": "consumer",
        "question": "Would you subscribe to a weekly grocery delivery service?",
        "description": "Recurring grocery delivery.",
        "options": ["subscribe", "maybe", "decline"],
        "price": 39.0,
        "currency": "AED",
        "tags": ["subscription"],
    }
    base.update(over)
    return Scenario.model_validate(base)


# ---------------------------------------------------------------------------
# deterministic mock simulation
# ---------------------------------------------------------------------------


def test_mock_output_deterministic():
    p = make_persona()
    s = make_scenario()
    a = mock_simulation_output(p, s, seed=42)
    b = mock_simulation_output(p, s, seed=42)
    assert a == b
    c = mock_simulation_output(p, s, seed=7)
    assert c.probabilities != a.probabilities or c.confidence != a.confidence


def test_mock_output_valid_by_construction():
    p = make_persona()
    s = make_scenario()
    out = mock_simulation_output(p, s, seed=42)
    assert set(out.probabilities) == set(s.options)
    assert all(0.0 <= v <= 1.0 for v in out.probabilities.values())
    assert abs(sum(out.probabilities.values()) - 1.0) <= PROB_SUM_TOL
    assert 0.0 <= out.confidence <= 1.0
    for f in out.behavioral_factors:
        assert 0.0 <= f.strength <= 1.0
    validate_simulation_output(out, p, s)  # must not raise


def test_mock_output_persona_sensitive():
    s = make_scenario()
    p1 = make_persona("AE_000001", latent={n: 0.95 for n in LatentAttributes.names()})
    p2 = make_persona("AE_000002", latent={n: 0.05 for n in LatentAttributes.names()})
    o1 = mock_simulation_output(p1, s, seed=42)
    o2 = mock_simulation_output(p2, s, seed=42)
    assert o1.probabilities != o2.probabilities


# ---------------------------------------------------------------------------
# deterministic validation: malformed / missing / unexpected
# ---------------------------------------------------------------------------


def _valid_out(**over) -> SimulationOutput:
    base = {
        "probabilities": {"subscribe": 0.5, "maybe": 0.3, "decline": 0.2},
        "confidence": 0.7,
        "behavioral_factors": [
            {"factor": "price_sensitivity", "direction": "positive", "strength": 0.6},
        ],
    }
    base.update(over)
    return SimulationOutput.model_validate(base)


def test_validation_missing_option():
    out = _valid_out(probabilities={"subscribe": 0.6, "maybe": 0.4})
    with pytest.raises(SimulationValidationError, match="missing options"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_unexpected_option():
    out = _valid_out(probabilities={"subscribe": 0.5, "maybe": 0.3, "decline": 0.2, "other": 0.0})
    with pytest.raises(SimulationValidationError, match="unexpected options"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_probability_out_of_range():
    out = _valid_out(probabilities={"subscribe": 1.5, "maybe": -0.5, "decline": 0.0})
    with pytest.raises(SimulationValidationError, match="outside \\[0,1\\]"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_probability_sum():
    out = _valid_out(probabilities={"subscribe": 0.9, "maybe": 0.9, "decline": 0.9})
    with pytest.raises(SimulationValidationError, match="sum to"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_confidence_range():
    out = _valid_out(confidence=1.5)
    with pytest.raises(SimulationValidationError, match="confidence"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_unknown_factor():
    out = _valid_out(behavioral_factors=[
        {"factor": "vibes", "direction": "positive", "strength": 0.5},
    ])
    with pytest.raises(SimulationValidationError, match="unknown factor"):
        validate_simulation_output(out, make_persona(), make_scenario())


def test_validation_factor_strength_range():
    out = _valid_out(behavioral_factors=[
        {"factor": "impulsivity", "direction": "positive", "strength": 2.0},
    ])
    with pytest.raises(SimulationValidationError, match="strength"):
        validate_simulation_output(out, make_persona(), make_scenario())


# ---------------------------------------------------------------------------
# simulator with MockProvider: provenance, choice, batch
# ---------------------------------------------------------------------------


def test_simulator_mock_provenance():
    sim = BehavioralSimulator(MockProvider(), seed=42)
    p = make_persona()
    s = make_scenario()
    r = sim.simulate_one(p, s)
    assert isinstance(r, SimulationResult)
    prov = r.provenance
    assert prov.provider == "mock"
    assert prov.model == MOCK_MODEL_LABEL
    assert prov.simulator_prompt_version == SIM_PROMPT_VERSION
    assert prov.scenario_id == s.scenario_id
    assert prov.scenario_version == s.scenario_version
    assert prov.persona_version == "pop_v1"
    assert prov.seed == 42
    assert prov.data_label == "synthetic_mock"
    assert prov.created_at is not None
    assert r.choice == max(r.probabilities, key=r.probabilities.get)
    assert r.record_id == f"{p.persona_id}|{s.full_id()}"


def test_simulator_mock_batch():
    sim = BehavioralSimulator(MockProvider(), seed=42)
    personas = [make_persona(f"AE_{i:06d}") for i in range(1, 4)]
    scenarios = [make_scenario("s1"), make_scenario("s2", options=["a", "b"])]
    results, batch = sim.simulate_batch(personas, scenarios)
    assert len(results) == 6
    assert not batch.failed
    assert batch.attempts == 6
    assert batch.retries == 0
    # incremental checkpoint hook fired for each result
    seen: list[str] = []
    sim2 = BehavioralSimulator(MockProvider(), seed=42)
    sim2.simulate_batch(personas, scenarios, on_result=lambda r: seen.append(r.record_id))
    assert len(seen) == 6


# ---------------------------------------------------------------------------
# checkpoint / resume
# ---------------------------------------------------------------------------


def test_batch_checkpoint_resume():
    calls = []

    def fn(item):
        calls.append(item)
        return StructuredResponse(
            data={"x": 1}, raw="{}", model="m",
            prompt_tokens=1, completion_tokens=1,
        )

    items = [("k1", 1), ("k2", 2), ("k3", 3)]
    r1 = run_batch(
        MockProvider(), items, fn,
        max_retries=0, backoff_base_s=0.0,
        done={"k2"},
    )
    assert [k for k, _ in r1.ok] == ["k1", "k3"]
    assert calls == [1, 3]  # k2 skipped

    # resume: nothing left to do
    r2 = run_batch(
        MockProvider(), items, fn,
        max_retries=0, backoff_base_s=0.0,
        done={k for k, _ in r1.ok} | {"k2"},
    )
    assert r2.ok == [] and r2.attempts == 0


# ---------------------------------------------------------------------------
# provider errors: retryable vs non-retryable
# ---------------------------------------------------------------------------


class FlakyProvider(MockProvider):
    """Fails N times (per distinct user prompt) with a retryable error, then succeeds."""

    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.calls: dict[str, int] = {}

    def complete_structured(self, **kwargs):
        user = kwargs.get("user", "")
        self.calls[user] = self.calls.get(user, 0) + 1
        if self.calls[user] <= self.fail_times:
            raise RetryableLLMError("transient 503")
        return StructuredResponse(
            data=SimOut(x=1), raw="{}", model="flaky",
            prompt_tokens=1, completion_tokens=1,
        )


class DeadProvider(MockProvider):
    """Always fails with a non-retryable error."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def complete_structured(self, **kwargs):
        self.calls += 1
        raise NonRetryableLLMError("401 unauthorized")


def test_batch_retryable_then_succeeds():
    prov = FlakyProvider(fail_times=2)
    r = run_batch(
        prov, [("a", 1), ("b", 2)],
        lambda item: prov.complete_structured(system="s", user=f"item-{item}"),
        max_retries=3, backoff_base_s=0.0,
    )
    assert len(r.ok) == 2 and not r.failed
    assert r.retries == 4  # 2 per item
    assert r.attempts == 6


def test_batch_nonretryable_fails_immediately():
    prov = DeadProvider()
    r = run_batch(
        prov, [("a", 1)],
        lambda item: prov.complete_structured(system="s", user="a"),
        max_retries=3, backoff_base_s=0.0,
    )
    assert not r.ok
    assert len(r.failed) == 1
    assert isinstance(r.failed[0][1], NonRetryableLLMError)
    assert prov.calls == 1  # no retries
    assert r.attempts == 1 and r.retries == 0


def test_batch_permanent_retryable_failure():
    prov = FlakyProvider(fail_times=99)
    tracker = CostTracker("flaky")
    r = run_batch(
        prov, [("a", 1)],
        lambda item: prov.complete_structured(system="s", user="a"),
        max_retries=2, backoff_base_s=0.0, tracker=tracker,
    )
    assert not r.ok and len(r.failed) == 1
    assert r.attempts == 3  # 1 + 2 retries
    assert r.retries == 2
    assert tracker.failures == 3


def test_backoff_exponential_with_jitter():
    import random
    d0 = backoff_delay(0, backoff_base_s=1.0, backoff_cap_s=60.0, jitter=0.0)
    d1 = backoff_delay(1, backoff_base_s=1.0, backoff_cap_s=60.0, jitter=0.0)
    d2 = backoff_delay(2, backoff_base_s=1.0, backoff_cap_s=60.0, jitter=0.0)
    assert (d0, d1, d2) == (1.0, 2.0, 4.0)
    # capped
    d10 = backoff_delay(10, backoff_base_s=1.0, backoff_cap_s=8.0, jitter=0.0)
    assert d10 == 8.0
    # jitter stays within [1-jitter, 1] of the base delay
    rng = random.Random(0)
    for _ in range(50):
        d = backoff_delay(3, backoff_base_s=1.0, backoff_cap_s=60.0, jitter=0.5, rng=rng)
        assert 8.0 * 0.5 <= d <= 8.0


# ---------------------------------------------------------------------------
# retry classification
# ---------------------------------------------------------------------------


def test_http_status_classification():
    assert classify_http_status(429) is RetryableLLMError
    assert classify_http_status(500) is RetryableLLMError
    assert classify_http_status(503) is RetryableLLMError
    assert classify_http_status(408) is RetryableLLMError
    assert classify_http_status(401) is NonRetryableLLMError
    assert classify_http_status(403) is NonRetryableLLMError
    assert classify_http_status(404) is NonRetryableLLMError
    assert classify_http_status(400) is NonRetryableLLMError
    assert classify_transport_error() is RetryableLLMError


def test_malformed_json_is_retryable():
    with pytest.raises(RetryableLLMError):
        extract_json_object("this is not json at all")
    with pytest.raises(RetryableLLMError):
        extract_json_object("```json\n{oops}\n```")


# ---------------------------------------------------------------------------
# nested JSON-schema handling + parameter forwarding (mock transport)
# ---------------------------------------------------------------------------


def test_inline_local_refs_nested_schema():
    from pydantic import BaseModel, Field
    from typing import Optional

    class Inner(BaseModel):
        a: int
        b: Optional[str] = None

    class Outer(BaseModel):
        name: str
        inner: Inner
        nested_list: list[Inner] = Field(default_factory=list)

    schema = Outer.model_json_schema()
    assert "$defs" in schema and any("$ref" in json.dumps(v) for v in schema["properties"].values())
    prepared = prepare_prompt_schema(Outer)
    # no $defs remain, and no dangling $ref anywhere
    assert "$defs" not in prepared
    assert "$ref" not in json.dumps(prepared)
    # inlined structure is intact
    assert prepared["properties"]["inner"]["properties"]["a"]["type"] == "integer"
    assert prepared["required"] == ["name", "inner"]


def _mock_transport(handler):
    def transport(request: httpx.Request) -> httpx.Response:
        return handler(json.loads(request.content))
    return httpx.MockTransport(transport)


def _openai_provider(role: str = "luna", base_url: str = "http://test/v1",
                     transport=None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name=role,
        base_url=base_url,
        model="test-model",
        timeout_s=5.0,
        default_params={"temperature": 0.3, "max_tokens": 1234, "top_p": 0.9},
        response_format="json_schema",
        transport=transport,
    )


def test_param_forwarding_and_nested_schema_end_to_end():
    """max_tokens/temperature are forwarded; nested schema is inlined; the
    returned data parses into the (nested) pydantic model."""

    class Inner(BaseModel):
        a: int

    class Outer(BaseModel):
        name: str
        inner: Inner

    seen: dict = {}

    def handler(body: dict) -> httpx.Response:
        seen.update(body)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "name": "x", "inner": {"a": 1},
            })}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "test-model-echo",
        })

    prov = _openai_provider(transport=_mock_transport(handler))
    resp = prov.complete_structured(
        system="sys", user="usr", schema=Outer,
    )
    # inference params forwarded
    assert seen["max_tokens"] == 1234
    assert seen["temperature"] == 0.3
    assert seen["top_p"] == 0.9
    # schema inlined in the request (no $defs / $ref)
    schema = seen["response_format"]["json_schema"]["schema"]
    assert "$defs" not in json.dumps(schema)
    assert "$ref" not in json.dumps(schema)
    # response parses into the nested model
    assert isinstance(resp.data, Outer)
    assert resp.data.inner.a == 1
    assert resp.model == "test-model-echo"


class SimOut(BaseModel):
    x: int


def test_provider_401_nonretryable():
    def handler(body: dict) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    prov = _openai_provider(transport=_mock_transport(handler))
    with pytest.raises(NonRetryableLLMError):
        prov.complete_structured(system="s", user="u", schema=SimOut)


def test_provider_429_retryable():
    def handler(body: dict) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    prov = _openai_provider(transport=_mock_transport(handler))
    with pytest.raises(RetryableLLMError):
        prov.complete_structured(system="s", user="u", schema=SimOut)


def test_provider_timeout_retryable():
    def handler(body: dict) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    prov = _openai_provider(transport=_mock_transport(handler))
    with pytest.raises(RetryableLLMError):
        prov.complete_structured(system="s", user="u", schema=SimOut)


def test_provider_400_invalid_schema_nonretryable():
    def handler(body: dict) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid json schema"}})

    prov = _openai_provider(transport=_mock_transport(handler))
    with pytest.raises(NonRetryableLLMError):
        prov.complete_structured(system="s", user="u", schema=SimOut)


# ---------------------------------------------------------------------------
# async batch: bounded concurrency, input-order results, clean shutdown
# ---------------------------------------------------------------------------


def test_async_batch_concurrency_and_order():
    in_flight = 0
    max_in_flight = 0

    async def fn(item):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return StructuredResponse(
            data={"v": item}, raw="{}", model="m",
            prompt_tokens=1, completion_tokens=1,
        )

    items = [(f"k{i}", i) for i in range(10)]
    r = asyncio.run(
        arun_batch(
            MockProvider(), items, fn,
            max_retries=0, max_concurrency=3, backoff_base_s=0.0,
        )
    )
    assert [k for k, _ in r.ok] == [f"k{i}" for i in range(10)]  # input order
    assert max_in_flight <= 3  # bounded
    assert max_in_flight > 1  # actually concurrent


def test_async_batch_clean_shutdown():
    async def fn(item):
        await asyncio.sleep(1.0)
        return StructuredResponse(data={}, raw="{}", model="m",
                                  prompt_tokens=1, completion_tokens=1)

    async def scenario():
        task = asyncio.create_task(
            arun_batch(MockProvider(), [(f"k{i}", i) for i in range(5)], fn,
                       max_retries=0, max_concurrency=2, backoff_base_s=0.0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(scenario())


# ---------------------------------------------------------------------------
# scenario loading + metrics
# ---------------------------------------------------------------------------


def test_load_scenarios_yaml(tmp_path):
    import yaml
    p = tmp_path / "sc.yaml"
    p.write_text(yaml.safe_dump({"scenarios": [
        {"scenario_id": "a", "scenario_version": "v1", "category": "c",
         "question": "q", "options": ["x", "y"]},
    ]}))
    scs = load_scenarios(p)
    assert scs[0].full_id() == "a@v1"


def test_metrics_on_mock_run():
    sim = BehavioralSimulator(MockProvider(), seed=42)
    personas = [make_persona(f"AE_{i:06d}") for i in range(1, 11)]
    scenarios = [make_scenario(f"s{i}", options=[f"o{j}" for j in range(3)])
                 for i in range(1, 4)]
    results, batch = sim.simulate_batch(personas, scenarios)
    m = compute_simulation_metrics(results, scenarios, batch, sim.tracker)
    assert m["n"] == 30
    assert m["prob_sum_validation_failure_rate"] == 0.0
    assert 0.0 <= m["confidence"]["mean"] <= 1.0
    assert set(m["option_selection"]) == {s.full_id() for s in scenarios}
    for dist in m["option_selection"].values():
        assert abs(sum(dist.values()) - 1.0) < 0.01
    assert 0.0 <= m["probability_entropy"]["mean_normalized"] <= 1.0
    assert m["retry_failure"]["failures"] == 0


def test_dataset_file_loads():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    scs = load_scenarios(repo / "config" / "scenarios" / "mock_scenarios.yaml")
    assert len(scs) == 5
    assert {s.scenario_id for s in scs} == {
        "grocery_subscription", "premium_smartphone_purchase", "ev_purchase",
        "travel_subscription", "streaming_service",
    }
    for s in scs:
        assert s.price is not None and s.currency == "AED"


# ---------------------------------------------------------------------------
# build_provider (models.yaml) still works
# ---------------------------------------------------------------------------


def test_build_provider_from_yaml():
    cfg = {
        "roles": {
            "mock": {"provider": "mock"},
            "luna": {
                "provider": "openai_compatible",
                "default_base_url": "http://127.0.0.1:9/v1",
                "default_model": "luna",
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        },
    }
    assert isinstance(build_provider("mock", cfg), MockProvider)
    luna = build_provider("luna", cfg)
    assert isinstance(luna, OpenAICompatibleProvider)
    assert luna.default_params == {"temperature": 0.3, "max_tokens": 2048}
    with pytest.raises(KeyError):
        build_provider("nope", cfg)

import pytest

from synthpop.llm import CostTracker, MockProvider, build_provider
from synthpop.llm.batch import run_batch
from synthpop.models.persona import LatentAttributes


def test_mock_provider_deterministic_and_conformant():
    p = MockProvider()
    r1 = p.complete_structured(
        system="s", user='{"age": 34}', schema=LatentAttributes
    )
    r2 = p.complete_structured(
        system="s", user='{"age": 34}', schema=LatentAttributes
    )
    assert r1.data == r2.data
    assert isinstance(r1.data, LatentAttributes)
    for name in LatentAttributes.names():
        assert 0.0 <= getattr(r1.data, name) <= 1.0


def test_mock_provider_varies_with_input():
    p = MockProvider()
    r1 = p.complete_structured(system="s", user="input-a", schema=LatentAttributes)
    r2 = p.complete_structured(system="s", user="input-b", schema=LatentAttributes)
    assert r1.data != r2.data


MODELS_CFG = {
    "roles": {
        "mock": {"provider": "mock"},
        "luna": {
            "provider": "openai_compatible",
            "default_base_url": "http://127.0.0.1:9/v1",
            "default_model": "luna",
        },
    }
}


def test_build_provider_roles():
    assert isinstance(build_provider("mock", MODELS_CFG), MockProvider)
    luna = build_provider("luna", MODELS_CFG)
    assert luna.model == "luna"
    with pytest.raises(KeyError):
        build_provider("nope", MODELS_CFG)


def test_batch_retries_then_succeeds():
    calls = {"n": 0}

    def fn(item):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("simulated transient failure")
        return MockProvider().complete_structured(
            system="s", user=item, schema=LatentAttributes
        )

    p = MockProvider()
    result = run_batch(
        p, [("k1", "x")], fn,
        max_retries=3, backoff_base_s=0.0,
        tracker=CostTracker("mock"),
    )
    assert len(result.ok) == 1
    assert result.retries == 2
    assert calls["n"] == 3


def test_batch_permanent_failure_and_resume():
    def fn(item):
        raise ValueError("always fails")

    tracker = CostTracker("mock")
    result = run_batch(
        MockProvider(), [("k1", "x"), ("k2", "y")], fn,
        max_retries=1, backoff_base_s=0.0, tracker=tracker,
    )
    assert len(result.failed) == 2
    assert tracker.failures == 4  # 2 items x (1 attempt + 1 retry)

    # resume: k1 already done -> only k2 attempted
    tracker2 = CostTracker("mock")
    result2 = run_batch(
        MockProvider(), [("k1", "x"), ("k2", "y")], fn,
        max_retries=1, backoff_base_s=0.0, tracker=tracker2,
        done={"k1"},
    )
    assert tracker2.failures == 2  # only k2, with 1 retry


def test_cost_tracker_math():
    t = CostTracker("luna", price_per_1k_prompt=1.0, price_per_1k_completion=4.0)
    t.record_success(prompt_tokens=1000, completion_tokens=500)
    t.record_failure(prompt_tokens=100, completion_tokens=0)
    assert t.estimated_cost_usd == pytest.approx(1.0 + 2.0 + 0.1)

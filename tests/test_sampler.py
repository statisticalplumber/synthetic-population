from collections import Counter

import pytest

from synthpop.config import load_population_config
from synthpop.population import PopulationSampler

CFG = "config/populations/uae_mock.yaml"


@pytest.fixture(scope="module")
def config():
    return load_population_config(CFG)


def test_reproducible_with_same_seed(config):
    a = PopulationSampler(config, seed=7).sample(200)
    b = PopulationSampler(config, seed=7).sample(200)
    assert [s.persona_id for s in a] == [s.persona_id for s in b]
    assert a[0].model_dump() == b[0].model_dump()


def test_different_seed_changes_sample(config):
    a = PopulationSampler(config, seed=7).sample(200)
    b = PopulationSampler(config, seed=8).sample(200)
    assert a[0].demographic_key() != b[0].demographic_key() or a[50].demographic_key() != b[50].demographic_key()


def test_sample_size_and_ids(config):
    skels = PopulationSampler(config, seed=1).sample(100)
    assert len(skels) == 100
    assert skels[0].persona_id == "AE_000000"
    assert skels[99].persona_id == "AE_000099"


def test_age_within_band(config):
    for s in PopulationSampler(config, seed=3).sample(300):
        lo, hi = map(int, s.age_band.split("-"))
        assert lo <= s.age <= hi


def test_marginal_close_to_target(config):
    skels = PopulationSampler(config, seed=42).sample(2000)
    counts = Counter(s.emirate for s in skels)
    target = dict(zip(config.variable("emirate").spec.categories,
                      config.variable("emirate").spec.probabilities))
    for cat, p in target.items():
        assert abs(counts.get(cat, 0) / 2000 - p) < 0.03


def test_conditional_preserved(config):
    skels = PopulationSampler(config, seed=42).sample(3000)
    # students should have occupation 'student'
    students = [s for s in skels if s.employment_status == "student"]
    assert students
    assert all(s.occupation_group == "student" for s in students)
    # retired -> retired
    retired = [s for s in skels if s.employment_status == "retired"]
    assert all(s.occupation_group == "retired" for s in retired)


def test_all_rows_valid_personas(config):
    skels = PopulationSampler(config, seed=5).sample(50)
    for s in skels:
        assert s.provenance.pipeline_stage == "skeleton"
        assert s.provenance.seed == 5
        assert s.data_label == "synthetic_mock"

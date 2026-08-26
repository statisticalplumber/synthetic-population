import copy

import pytest

from synthpop.config import load_population_config
from synthpop.models.config import PopulationConfig

CFG = "config/populations/uae_mock.yaml"


def test_load_uae_mock_config():
    cfg = load_population_config(CFG)
    assert cfg.country_code == "AE"
    assert cfg.data_label == "synthetic_mock"
    assert len(cfg.variables) == 12
    assert cfg.config_hash()  # non-empty


def test_config_hash_stable():
    a = load_population_config(CFG)
    b = load_population_config(CFG)
    assert a.config_hash() == b.config_hash()


def test_probabilities_must_sum_to_one():
    cfg = load_population_config(CFG)
    bad = copy.deepcopy(cfg.model_dump())
    bad["variables"][0]["spec"]["probabilities"][0] += 0.1
    with pytest.raises(Exception):
        PopulationConfig.model_validate(bad)


def test_forward_reference_rejected():
    cfg = load_population_config(CFG)
    bad = cfg.model_dump()
    # make 'city' (index 1) reference 'gender' (index 3) — defined later
    bad["variables"][1]["spec"]["rules"][0]["when"] = {"gender": ["male"]}
    with pytest.raises(Exception):
        PopulationConfig.model_validate(bad)


def test_missing_required_field_rejected():
    cfg = load_population_config(CFG)
    bad = cfg.model_dump()
    bad["variables"] = [v for v in bad["variables"] if v["name"] != "housing_status"]
    with pytest.raises(Exception):
        PopulationConfig.model_validate(bad)

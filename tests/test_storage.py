import json

from synthpop.config import load_population_config
from synthpop.population import PopulationSampler
from synthpop.storage import (
    load_skeletons,
    make_run_id,
    read_parquet_dir,
    write_jsonl,
    write_manifest,
    write_partitioned_parquet,
)


def test_parquet_roundtrip_partitioned(tmp_path):
    config = load_population_config("config/populations/uae_mock.yaml")
    skels = PopulationSampler(config, seed=9).sample(50)
    records = [s.model_dump(mode="json") for s in skels]

    files = write_partitioned_parquet(records, tmp_path, partition_by=["country"])
    assert len(files) == 1  # single country
    back = read_parquet_dir(tmp_path)
    assert len(back) == 50
    assert back[0]["persona_id"] == "AE_000000"
    assert back[0]["provenance"]["seed"] == 9

    # re-validate into pydantic models
    loaded = load_skeletons(back)
    assert loaded[0].age == skels[0].age


def test_jsonl_and_manifest(tmp_path):
    records = [{"a": 1}, {"a": 2}]
    p = write_jsonl(records, tmp_path / "r.jsonl")
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["a"] == 1
    m = write_manifest(tmp_path, stage="skeletons", run_id="r1", extra={"n": 2})
    assert json.loads(m.read_text())["run_id"] == "r1"


def test_run_id_format():
    assert make_run_id("AE", "uae_mock_v1", 1000, 42) == "ae_uae_mock_v1_n1000_s42"

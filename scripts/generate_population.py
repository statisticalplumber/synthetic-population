#!/usr/bin/env python
"""Stage 1 — generate demographic skeletons from a population config.

Usage:
    python scripts/generate_population.py \
        --config config/populations/uae_mock.yaml \
        --n 1000 --seed 42 \
        --out data/generated/skeletons
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthpop.config import load_population_config          # noqa: E402
from synthpop.population import PopulationSampler           # noqa: E402
from synthpop.storage import (                              # noqa: E402
    make_run_id, write_jsonl, write_manifest, write_partitioned_parquet,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/populations/uae_mock.yaml")
    ap.add_argument("--n", type=int, default=None, help="override sample size")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--out", default="data/generated/skeletons")
    args = ap.parse_args()

    config = load_population_config(args.config)
    n = args.n or config.sample_size
    seed = config.seed if args.seed is None else args.seed
    sampler = PopulationSampler(config, seed=seed)

    skeletons = sampler.sample(n)
    records = [s.model_dump(mode="json") for s in skeletons]

    run_id = make_run_id(config.country_code, config.config_version, n, seed)
    out_dir = Path(args.out) / run_id
    files = write_partitioned_parquet(records, out_dir, partition_by=["country"])
    write_jsonl(records, out_dir / "records.jsonl")
    write_manifest(
        out_dir, stage="skeletons", run_id=run_id,
        extra={
            "config_path": str(args.config),
            "config_version": config.config_version,
            "config_hash": config.config_hash(),
            "seed": seed,
            "n": n,
            "data_label": config.data_label,
            "source_note": config.source_note,
        },
    )

    print(json.dumps({
        "run_id": run_id,
        "n": n,
        "seed": seed,
        "config_hash": config.config_hash(),
        "data_label": config.data_label,
        "parquet_files": [str(f) for f in files],
        "out_dir": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

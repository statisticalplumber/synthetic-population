#!/usr/bin/env python
"""Stage 2 — enrich skeletons with latent behavioral attributes.

Uses the configured provider role (mock by default; set to luna/terra/sol in
config/generation.yaml for real LLM enrichment).

Usage:
    python scripts/enrich_personas.py --run-dir data/generated/skeletons/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthpop.config import load_population_config, load_yaml_dict      # noqa: E402
from synthpop.llm import CostTracker, build_provider                   # noqa: E402
from synthpop.persona import PROMPT_VERSION, enrich_skeletons          # noqa: E402
from synthpop.storage import (                                        # noqa: E402
    load_skeletons,
    read_parquet_dir,
    write_jsonl,
    write_manifest,
    write_partitioned_parquet,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--base", default="data/generated/skeletons")
    ap.add_argument("--gen-config", default="config/generation.yaml")
    ap.add_argument("--models-config", default="config/models.yaml")
    ap.add_argument("--role", default=None, help="override enrichment role")
    ap.add_argument("--out", default="data/generated/personas")
    args = ap.parse_args()

    base = Path(args.base)
    run_dir = Path(args.run_dir) if args.run_dir else sorted(
        p for p in base.iterdir() if p.is_dir()
    )[-1]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pop_cfg_path = manifest["extra"]["config_path"]
    config = load_population_config(pop_cfg_path)

    gen_cfg = load_yaml_dict(args.gen_config)
    models_cfg = load_yaml_dict(args.models_config)
    role = args.role or gen_cfg.get("enrichment", {}).get("role", "luna")
    provider = build_provider(role, models_cfg)
    pricing = models_cfg.get("pricing_usd_per_1k", {}).get(role, {})
    tracker = CostTracker(
        provider.name,
        price_per_1k_prompt=pricing.get("prompt", 0.0),
        price_per_1k_completion=pricing.get("completion", 0.0),
    )

    records = read_parquet_dir(run_dir)
    skeletons = load_skeletons(records)

    out_dir = Path(args.out) / manifest["run_id"]
    personas, batch = enrich_skeletons(
        skeletons, provider, config, tracker=tracker,
    )

    persona_records = [p.model_dump(mode="json") for p in personas]
    files = write_partitioned_parquet(persona_records, out_dir, partition_by=["country"])
    jsonl = write_jsonl(persona_records, out_dir / "records.jsonl")
    write_manifest(
        out_dir, stage="personas", run_id=manifest["run_id"],
        extra={
            "n": len(persona_records),
            "files": [str(f) for f in files] + [str(jsonl)],
            "provider": provider.name,
            "role": role,
            "prompt_version": PROMPT_VERSION,
            "estimated_cost_usd": tracker.estimated_cost_usd,
            "requests": tracker.requests,
            "batch_failed": len(batch.failed),
        },
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "n": len(persona_records),
        "failed": len(batch.failed),
        "role": role,
        "provider": provider.name,
        "requests": tracker.requests,
        "estimated_cost_usd": round(tracker.estimated_cost_usd, 4),
        "out_dir": str(out_dir),
    }, indent=2))
    return 0 if not batch.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

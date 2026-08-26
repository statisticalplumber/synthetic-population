#!/usr/bin/env python
"""Test a configured provider role against its real endpoint.

Checks: /v1/models reachability, then runs a small batch of real
enrichment calls through the production path (provider -> batch ->
pydantic validation).

Usage:
    LUNA_BASE_URL=http://127.0.0.1:1234/v1 LUNA_MODEL=luna \
        python scripts/test_provider.py --role luna --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from synthpop.config import load_population_config, load_yaml_dict  # noqa: E402
from synthpop.llm import CostTracker, build_provider               # noqa: E402
from synthpop.llm.provider import OpenAICompatibleProvider         # noqa: E402
from synthpop.persona import enrich_skeletons                      # noqa: E402
from synthpop.population import PopulationSampler                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--gen-config", default="config/generation.yaml")
    ap.add_argument("--models-config", default="config/models.yaml")
    ap.add_argument("--pop-config", default="config/populations/uae_mock.yaml")
    ap.add_argument("--out", default=None, help="persist enriched personas to <out>.jsonl")
    args = ap.parse_args()

    models_cfg = load_yaml_dict(args.models_config)
    provider = build_provider(args.role, models_cfg)
    if not isinstance(provider, OpenAICompatibleProvider):
        raise SystemExit(f"role {args.role!r} is not an openai_compatible provider")

    # 1. reachability + model list
    print(f"probing {provider.base_url}/models ...")
    r = httpx.get(f"{provider.base_url}/models", timeout=30)
    r.raise_for_status()
    models = r.json().get("data", [])
    available = [m["id"] for m in models]
    print(f"models on server: {available}")
    if provider.model not in available:
        print(f"WARNING: configured model {provider.model!r} not in server list")

    # 2. small real batch through the production enrichment path
    config = load_population_config(args.pop_config)
    skeletons = PopulationSampler(config, seed=0).sample(args.n)
    tracker = CostTracker(provider.name)
    t0 = time.time()
    personas, batch = enrich_skeletons(
        skeletons, provider, config, tracker=tracker, max_retries=2
    )
    dt = time.time() - t0

    print(json.dumps({
        "role": args.role,
        "model": provider.model,
        "response_format": provider.response_format,
        "requested": args.n,
        "ok": len(batch.ok),
        "failed": len(batch.failed),
        "attempts": batch.attempts,
        "retries": batch.retries,
        "elapsed_s": round(dt, 1),
        "s_per_ok": round(dt / max(len(batch.ok), 1), 1),
        "prompt_tokens": tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "estimated_cost_usd": round(tracker.estimated_cost_usd, 4),
    }, indent=2))

    if args.out:
        import statistics
        from pathlib import Path as P

        out = P(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for p in personas:
                f.write(p.model_dump_json() + "\n")
        stats = {}
        for k in personas[0].latent.model_dump():
            vals = [p.latent.model_dump()[k] for p in personas]
            stats[k] = {"mean": round(statistics.fmean(vals), 3),
                        "std": round(statistics.pstdev(vals), 3)}
        print(f"persisted {len(personas)} personas -> {out}")
        print("latent stats:")
        for k, v in stats.items():
            print(f"  {k:28s} mean={v['mean']:.3f} std={v['std']:.3f}")

    for key, resp in batch.ok[:2]:
        print(f"--- {key} (model={resp.model})")
        print(resp.raw[:300])
    for key, err in batch.failed:
        print(f"FAILED {key}: {type(err).__name__}: {err}")
    return 0 if not batch.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

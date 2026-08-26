#!/usr/bin/env python
"""Layer 3 — behavioral scenario simulation.

Simulates (persona, scenario) pairs through the configured provider role and
writes probabilistic SimulationResult records with full provenance.

Checkpoint/resume: results are appended to <out>/<run_id>/simulations.jsonl
as each pair completes; the file doubles as the checkpoint. Re-running the
same command resumes from where it stopped (idempotent by
(persona_id, scenario_id@scenario_version) key).

Usage:
    python scripts/simulate.py \
        --personas data/generated/personas/<run_id> \
        --scenarios config/scenarios/mock_scenarios.yaml \
        --role mock --out data/generated/simulations

    # real provider (async, bounded concurrency):
    python scripts/simulate.py --role luna --concurrency 8 ...
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthpop.config import load_yaml_dict                                  # noqa: E402
from synthpop.llm import CostTracker, MockProvider, build_provider           # noqa: E402
from synthpop.models.persona import Persona                                  # noqa: E402
from synthpop.models.simulation import SimulationResult                      # noqa: E402
from synthpop.simulation import (                                           # noqa: E402
    SIM_PROMPT_VERSION,
    BehavioralSimulator,
    compute_simulation_metrics,
    load_scenarios,
    pair_key,
)
from synthpop.storage import read_parquet_dir, write_manifest                # noqa: E402


def load_personas(path: Path) -> tuple[list[Persona], str]:
    """Load personas from a run dir (parquet + manifest) or a .jsonl file."""
    if path.is_dir():
        records = read_parquet_dir(path)
        run_id = json.loads((path / "manifest.json").read_text())["run_id"]
    elif path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_id = path.parent.name if path.parent.name != "records.jsonl" else path.stem
    else:
        raise SystemExit(f"unsupported personas path: {path} (run dir or .jsonl)")
    return [Persona.model_validate(r) for r in records], run_id


def load_checkpoint(path: Path) -> tuple[set[str], list[dict]]:
    """Resume support: existing results file => done keys + stored results."""
    if not path.exists():
        return set(), []
    stored = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = {
        f"{r['persona_id']}|{r['scenario_id']}@{r['scenario_version']}"
        for r in stored
    }
    return done, stored


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Layer 3 behavioral scenario simulation"
    )
    ap.add_argument("--personas", required=True,
                    help="run dir (parquet+manifest) or .jsonl of personas")
    ap.add_argument("--scenarios", required=True,
                    help="YAML/JSON file with a list of scenarios")
    ap.add_argument("--role", default="mock",
                    help="provider role from config/models.yaml (mock/luna/...)")
    ap.add_argument("--out", default="data/generated/simulations")
    ap.add_argument("--limit", type=int, default=None,
                    help="simulate only the first N personas")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="bounded in-flight requests (async runner)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sync", action="store_true",
                    help="force the sequential sync runner (default: async)")
    ap.add_argument("--models-config", default="config/models.yaml")
    ap.add_argument("--gen-config", default="config/generation.yaml")
    args = ap.parse_args()

    personas, personas_run_id = load_personas(Path(args.personas))
    if args.limit is not None:
        personas = personas[: args.limit]
    scenarios = load_scenarios(args.scenarios)

    models_cfg = load_yaml_dict(args.models_config)
    gen_cfg = load_yaml_dict(args.gen_config)
    sim_cfg = gen_cfg.get("simulation", {})
    max_retries = args.max_retries if args.max_retries != 3 else sim_cfg.get("max_retries", 3)
    concurrency = args.concurrency if args.concurrency != 8 else sim_cfg.get("concurrency", 8)

    role = args.role
    provider = build_provider(role, models_cfg)
    pricing = models_cfg.get("pricing_usd_per_1k", {}).get(role, {})
    tracker = CostTracker(
        provider.name,
        price_per_1k_prompt=pricing.get("prompt", 0.0),
        price_per_1k_completion=pricing.get("completion", 0.0),
    )
    simulator = BehavioralSimulator(
        provider,
        seed=args.seed,
        max_retries=max_retries,
        max_concurrency=concurrency,
        tracker=tracker,
    )

    run_id = f"sim_{personas_run_id}_{role}"
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "simulations.jsonl"

    done, stored = load_checkpoint(results_path)
    n_pairs = len(personas) * len(scenarios)
    todo = sum(
        1 for p in personas for s in scenarios
        if pair_key(p.persona_id, s) not in done
    )
    print(
        f"run_id={run_id}  personas={len(personas)}  scenarios={len(scenarios)}  "
        f"pairs={n_pairs}  already_done={len(done)}  todo={todo}",
        file=sys.stderr,
    )

    # incremental checkpoint: append each result as it completes
    new_results: list[SimulationResult] = []
    with results_path.open("a", encoding="utf-8") as f:
        def on_result(result: SimulationResult) -> None:
            new_results.append(result)
            f.write(result.model_dump_json() + "\n")

        if args.sync or isinstance(provider, MockProvider):
            results, batch = simulator.simulate_batch(
                personas, scenarios, done=done, on_result=on_result
            )
        else:
            async def _run():
                try:
                    return await simulator.asimulate_batch(
                        personas, scenarios, done=done, on_result=on_result
                    )
                finally:
                    aclose = getattr(provider, "aclose", None)
                    if aclose is not None:
                        res = aclose()
                        if inspect.isawaitable(res):
                            await res
            results, batch = asyncio.run(_run())

    # full result set (stored + new), re-validated
    all_records = stored + [r.model_dump(mode="json") for r in results]
    all_results = [SimulationResult.model_validate(r) for r in all_records]

    metrics = compute_simulation_metrics(all_results, scenarios, batch, tracker)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        out_dir, stage="simulations", run_id=run_id,
        extra={
            "n": len(all_results),
            "n_personas": len(personas),
            "n_scenarios": len(scenarios),
            "role": role,
            "provider": provider.name,
            "simulator_prompt_version": SIM_PROMPT_VERSION,
            "seed": args.seed,
            "max_retries": max_retries,
            "concurrency": concurrency,
            "personas_run_id": personas_run_id,
            "scenarios": [s.full_id() for s in scenarios],
            "estimated_cost_usd": tracker.estimated_cost_usd,
            "requests": tracker.requests,
            "batch_failed": len(batch.failed),
        },
    )

    print(json.dumps({
        "run_id": run_id,
        "n": len(all_results),
        "new": len(results),
        "failed": len(batch.failed),
        "attempts": batch.attempts,
        "retries": batch.retries,
        "role": role,
        "provider": provider.name,
        "estimated_cost_usd": round(tracker.estimated_cost_usd, 4),
        "out_dir": str(out_dir),
        "metrics": {
            "prob_sum_validation_failure_rate": metrics.get("prob_sum_validation_failure_rate"),
            "confidence": metrics.get("confidence"),
            "probability_entropy": metrics.get("probability_entropy"),
            "option_selection": metrics.get("option_selection"),
        },
    }, indent=2))
    return 0 if not batch.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

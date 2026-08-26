"""Deterministic simulation metrics (code-computed, no LLM).

Computed over accepted ``SimulationResult`` records + batch/cost accounting:

- probability sum validation failure rate (should be ~0: results only enter
  the store after deterministic validation)
- confidence distribution (mean / std / percentiles)
- option selection distribution (per scenario, argmax choice)
- probability entropy (mean Shannon entropy of the distributions)
- persona-level response diversity (choice entropy across scenarios, per
  persona, averaged)
- scenario-level response diversity (choice entropy across personas, per
  scenario, averaged)
- retry / failure rate (from the batch result + cost tracker)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..llm.batch import BatchResult
from ..llm.cost import CostTracker
from ..models.scenario import Scenario
from ..models.simulation import PROB_SUM_TOL, SimulationResult


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _normalized_entropy(counts: Counter, n: int) -> float:
    if n == 0:
        return 0.0
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    k = len(counts)
    return h / math.log2(k) if k > 1 else 0.0


def compute_simulation_metrics(
    results: list[SimulationResult],
    scenarios: list[Scenario],
    batch: BatchResult,
    tracker: CostTracker,
) -> dict[str, Any]:
    n = len(results)
    out: dict[str, Any] = {
        "n": n,
        "n_personas": len({r.persona_id for r in results}),
        "n_scenarios": len({r.scenario_id for r in results}),
    }
    if n == 0:
        out["note"] = "no results"
        return out

    # 1) probability sum validation failure rate (post-acceptance; expect 0)
    bad_sums = sum(
        1 for r in results
        if abs(sum(r.probabilities.values()) - 1.0) > PROB_SUM_TOL
    )
    out["prob_sum_validation_failure_rate"] = round(bad_sums / n, 6)

    # 2) confidence distribution
    confs = sorted(r.confidence for r in results)
    mean_c = sum(confs) / n
    std_c = math.sqrt(sum((c - mean_c) ** 2 for c in confs) / n)
    out["confidence"] = {
        "mean": round(mean_c, 4),
        "std": round(std_c, 4),
        "min": round(confs[0], 4),
        "p50": round(_percentile(confs, 0.50), 4),
        "p95": round(_percentile(confs, 0.95), 4),
        "max": round(confs[-1], 4),
    }

    # 3) option selection distribution (argmax choice), per scenario
    option_selection: dict[str, dict[str, float]] = {}
    for sc in scenarios:
        choices = Counter(
            r.choice for r in results if r.scenario_id == sc.scenario_id
        )
        m = sum(choices.values()) or 1
        option_selection[sc.full_id()] = {
            opt: round(choices.get(opt, 0) / m, 4) for opt in sc.options
        }
    out["option_selection"] = option_selection

    # 4) probability entropy (mean, normalized by log2(n_options))
    entropies = []
    for r in results:
        k = len(r.probabilities)
        h = r.entropy(base=2.0)
        entropies.append(h / math.log2(k) if k > 1 else 0.0)
    out["probability_entropy"] = {
        "mean_normalized": round(sum(entropies) / n, 4),
        "min": round(min(entropies), 4),
        "max": round(max(entropies), 4),
    }

    # 5) persona-level response diversity (choice entropy across scenarios)
    per_persona: dict[str, Counter] = {}
    for r in results:
        per_persona.setdefault(r.persona_id, Counter())[r.choice] += 1
    persona_div = [
        _normalized_entropy(c, sum(c.values())) for c in per_persona.values()
    ]
    out["persona_response_diversity"] = {
        "mean": round(sum(persona_div) / len(persona_div), 4) if persona_div else 0.0,
        "min": round(min(persona_div), 4) if persona_div else 0.0,
        "max": round(max(persona_div), 4) if persona_div else 0.0,
    }

    # 6) scenario-level response diversity (choice entropy across personas)
    per_scenario: dict[str, Counter] = {}
    for r in results:
        per_scenario.setdefault(r.scenario_id, Counter())[r.choice] += 1
    scenario_div = [
        _normalized_entropy(c, sum(c.values())) for c in per_scenario.values()
    ]
    out["scenario_response_diversity"] = {
        "mean": round(sum(scenario_div) / len(scenario_div), 4) if scenario_div else 0.0,
        "min": round(min(scenario_div), 4) if scenario_div else 0.0,
        "max": round(max(scenario_div), 4) if scenario_div else 0.0,
    }

    # 7) retry / failure rate
    total_attempts = batch.attempts
    out["retry_failure"] = {
        "attempts": total_attempts,
        "retries": batch.retries,
        "failures": len(batch.failed),
        "retry_rate": round(batch.retries / total_attempts, 6) if total_attempts else 0.0,
        "failure_rate": round(len(batch.failed) / n, 6) if n else 0.0,
        "tracker": tracker.to_dict(),
    }
    return out

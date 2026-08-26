"""Deterministic distribution validation (no LLM).

Checks generated samples against the target distributions declared in the
PopulationConfig:

- flat `categorical` variables  -> marginal comparison
- `conditional` variables       -> per-parent-value conditional comparison
  (the sampler's contract is to preserve the conditionals, so that is what
  we test; marginals of conditional variables are derived quantities)
- `age_mixture`                 -> binned histogram comparison
- duplicates / missing values   -> record-level checks

Distance metric: Jensen-Shannon divergence (base 2, range [0, 1]).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2 as chi2_dist

from ..models.config import (
    AgeMixtureSpec,
    CategoricalSpec,
    ConditionalSpec,
    PopulationConfig,
)
from ..models.persona import PersonaSkeleton
from ..models.report import ValidationReport, VariableCheck

DEFAULT_THRESHOLDS = {
    "alpha": 0.01,           # chi-squared significance level for pass/fail
    "min_expected_count": 5,  # chi-squared validity condition
    "min_parent_count": 30,   # conditional cells below this are 'insufficient_data'
}


@dataclass
class CheckThresholds:
    alpha: float = DEFAULT_THRESHOLDS["alpha"]
    min_expected_count: float = DEFAULT_THRESHOLDS["min_expected_count"]
    min_parent_count: int = DEFAULT_THRESHOLDS["min_parent_count"]


def _js(p: list[float], q: list[float]) -> float:
    return float(jensenshannon(np.array(p, dtype=float), np.array(q, dtype=float), base=2))


def _compare_table(target: dict[str, float], counts: Counter, n: int,
                   thresholds: CheckThresholds, variable: str, kind: str,
                   detail: str = "") -> VariableCheck:
    """Compare an observed count table against a target distribution.

    Decision: chi-squared goodness-of-fit (sample-size aware).
    - insufficient_data: n < min_parent_count, or any expected cell count
      below min_expected_count (chi-squared approximation invalid).
    - fail: chi-squared p-value < alpha.
    - pass: otherwise.
    """
    target = {str(k): v for k, v in target.items()}
    keys = sorted(target.keys())
    p = [target[k] for k in keys]
    obs = [counts.get(k, 0) for k in keys]
    q = [c / n for c in obs]
    js = _js(p, q)
    mad = max(abs(target[k] - c / n) for k, c in zip(keys, obs))

    expected = [n * tp for tp in p]
    if n < thresholds.min_parent_count or min(expected) < thresholds.min_expected_count:
        status = "insufficient_data"
        chi2_stat, chi2_p = 0.0, 1.0
    else:
        chi2_stat = sum((o - e) ** 2 / e for o, e in zip(obs, expected))
        df = max(len(keys) - 1, 1)
        chi2_p = float(chi2_dist.sf(chi2_stat, df))
        status = "fail" if chi2_p < thresholds.alpha else "pass"

    return VariableCheck(
        variable=variable, kind=kind, n=n,
        target={str(k): round(target[k], 6) for k in keys},
        observed={str(k): round(c / n, 6) for k, c in zip(keys, obs)},
        js_divergence=round(js, 6), max_abs_diff=round(mad, 6),
        chi2_stat=round(chi2_stat, 4), chi2_pvalue=round(chi2_p, 6),
        status=status, detail=detail,
    )


def check_marginal(name: str, spec: CategoricalSpec, rows: list[dict],
                   thresholds: CheckThresholds) -> VariableCheck:
    n = len(rows)
    counts = Counter(str(r[name]) for r in rows)
    target = {c: p for c, p in zip(spec.categories, spec.probabilities)}
    return _compare_table(target, counts, n, thresholds, name, "marginal")


def check_conditional(name: str, spec: ConditionalSpec, rows: list[dict],
                      thresholds: CheckThresholds) -> list[VariableCheck]:
    """One check per (parent combination) that has enough rows."""
    n = len(rows)
    checks: list[VariableCheck] = []
    # Group rows by the parent values referenced in the rules.
    parent_vars = sorted({v for rule in spec.rules for v in rule.when})
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(str(r[v]) for v in parent_vars)
        groups.setdefault(key, []).append(r)

    for key, group in sorted(groups.items()):
        # find the first rule matching this parent combination
        match = None
        for rule in spec.rules:
            if all(str(r) in [str(x) for x in rule.when[v]] for v, r in zip(parent_vars, key) if v in rule.when):
                match = rule
                break
        target = match.distribution if match is not None else spec.default
        counts = Counter(str(r[name]) for r in group)
        detail = " | ".join(f"{v}={k}" for v, k in zip(parent_vars, key))
        checks.append(
            _compare_table(target, counts, len(group), thresholds, name,
                           "conditional", detail=detail)
        )
    return checks


def check_age_mixture(name: str, spec: AgeMixtureSpec, rows: list[dict],
                      thresholds: CheckThresholds) -> VariableCheck:
    n = len(rows)
    target = {f"{b.lo}-{b.hi}": b.weight for b in spec.bands}
    counts = Counter(str(r["age_band"]) for r in rows)
    return _compare_table(target, counts, n, thresholds, name, "age_bins")


def duplicate_and_missing_rates(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    if n == 0:
        return 0.0, 0.0
    keys = [tuple(sorted((k, str(v)) for k, v in r.items() if k != "persona_id")) for r in rows]
    dup_rate = 1.0 - len(set(keys)) / n
    missing = sum(
        1 for r in rows
        for v in r.values()
        if v is None or (isinstance(v, str) and v == "")
    )
    total_cells = sum(len(r) for r in rows)
    return dup_rate, missing / total_cells


def run_distribution_checks(
    config: PopulationConfig,
    skeletons: list[PersonaSkeleton],
    run_id: str,
    thresholds: CheckThresholds | None = None,
) -> ValidationReport:
    th = thresholds or CheckThresholds()
    rows = [s.model_dump(mode="json") for s in skeletons]
    n = len(rows)
    checks: list[VariableCheck] = []

    for var in config.variables:
        spec = var.spec
        if isinstance(spec, CategoricalSpec):
            checks.append(check_marginal(var.name, spec, rows, th))
        elif isinstance(spec, ConditionalSpec):
            checks.extend(check_conditional(var.name, spec, rows, th))
        elif isinstance(spec, AgeMixtureSpec):
            checks.append(check_age_mixture(var.name, spec, rows, th))

    dup_rate, missing_rate = duplicate_and_missing_rates(rows)
    overall = "pass" if not any(c.status == "fail" for c in checks) else "fail"
    return ValidationReport(
        run_id=run_id,
        config_version=config.config_version,
        config_hash=config.config_hash(),
        seed=config.seed,
        n=n,
        checks=checks,
        duplicate_rate=round(dup_rate, 6),
        missing_rate=round(missing_rate, 6),
        overall=overall,
        thresholds={
            "alpha": th.alpha,
            "min_expected_count": th.min_expected_count,
            "min_parent_count": float(th.min_parent_count),
        },
    )

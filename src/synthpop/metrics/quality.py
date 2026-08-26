"""Basic deterministic quality metrics (code-computed, no LLM).

Stage-1 metrics (skeletons):
- per-variable distribution distance (from ValidationReport)
- duplicate / missing rates
- per-variable normalized entropy (diversity indicator)
- age summary (mean / std)

Stage-2+ metrics (latent variance, lexical diversity, disagreement) are added
as those stages land.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..models.config import PopulationConfig
from ..models.persona import PersonaSkeleton
from ..models.report import ValidationReport


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


def compute_skeleton_metrics(
    config: PopulationConfig,
    skeletons: list[PersonaSkeleton],
    report: ValidationReport,
) -> dict[str, Any]:
    n = len(skeletons)
    rows = [s.model_dump(mode="json") for s in skeletons]

    entropy: dict[str, float] = {}
    for var in config.variables:
        name = var.name
        counts = Counter(str(r[name]) for r in rows)
        entropy[name] = round(_normalized_entropy(counts, n), 4)

    ages = [s.age for s in skeletons]
    age_mean = sum(ages) / n
    age_std = math.sqrt(sum((a - age_mean) ** 2 for a in ages) / n)

    return {
        "n": n,
        "run_id": report.run_id,
        "overall": report.overall,
        "duplicate_rate": report.duplicate_rate,
        "missing_rate": report.missing_rate,
        "max_js_divergence": round(max(c.js_divergence for c in report.checks), 6),
        "age": {"mean": round(age_mean, 3), "std": round(age_std, 3)},
        "normalized_entropy": entropy,
        "failed_checks": [
            {"variable": c.variable, "detail": c.detail,
             "js_divergence": c.js_divergence, "max_abs_diff": c.max_abs_diff}
            for c in report.checks if c.status == "fail"
        ],
    }

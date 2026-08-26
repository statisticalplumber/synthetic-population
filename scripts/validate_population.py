#!/usr/bin/env python
"""Stage 1 — deterministic validation of a skeleton run.

Runs distribution checks (target vs generated), duplicate/missing checks and
quality metrics, then writes a report.

Usage:
    python scripts/validate_population.py --run-dir data/generated/skeletons/<run_id>
    python scripts/validate_population.py            # auto-picks latest run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthpop.config import load_population_config, load_yaml_dict  # noqa: E402
from synthpop.metrics import compute_skeleton_metrics               # noqa: E402
from synthpop.storage import load_skeletons, read_parquet_dir       # noqa: E402
from synthpop.validation import CheckThresholds, run_distribution_checks  # noqa: E402


def _latest_run(base: Path) -> Path:
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit(f"no runs found under {base}")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--base", default="data/generated/skeletons")
    ap.add_argument("--gen-config", default="config/generation.yaml")
    ap.add_argument("--pop-config", default=None, help="default: from run manifest")
    ap.add_argument("--reports", default="data/reports")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.base))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pop_cfg_path = args.pop_config or manifest["extra"]["config_path"]
    config = load_population_config(pop_cfg_path)

    gen_cfg = load_yaml_dict(args.gen_config)
    dc = gen_cfg.get("distribution_checks", {})
    thresholds = CheckThresholds(
        alpha=dc.get("alpha", 0.01),
        min_expected_count=dc.get("min_expected_count", 5),
        min_parent_count=dc.get("min_parent_count", 30),
    )

    records = read_parquet_dir(run_dir)
    skeletons = load_skeletons(records)
    report = run_distribution_checks(
        config, skeletons, run_id=manifest["run_id"], thresholds=thresholds
    )
    metrics = compute_skeleton_metrics(config, skeletons, report)

    reports_dir = Path(args.reports) / manifest["run_id"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "validation_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # console summary
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Validation: {report.run_id} (n={report.n})")
    for col in ("variable", "kind", "detail", "n", "JS", "max|Δ|", "status"):
        table.add_column(col)
    for c in report.checks:
        style = {"pass": "green", "fail": "bold red", "insufficient_data": "dim"}[c.status]
        table.add_row(
            c.variable, c.kind, c.detail, str(c.n),
            f"{c.js_divergence:.4f}", f"{c.max_abs_diff:.4f}",
            f"[{style}]{c.status}[/{style}]",
        )
    console.print(table)
    console.print(
        f"overall=[bold]{report.overall}[/bold]  "
        f"duplicate_rate={report.duplicate_rate}  missing_rate={report.missing_rate}  "
        f"reports={reports_dir}"
    )
    return 0 if report.overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

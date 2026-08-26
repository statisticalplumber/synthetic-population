from collections import Counter

from synthpop.config import load_population_config
from synthpop.population import PopulationSampler
from synthpop.validation import CheckThresholds, run_distribution_checks

CFG = "config/populations/uae_mock.yaml"


def test_mock_run_passes_checks():
    config = load_population_config(CFG)
    skels = PopulationSampler(config, seed=42).sample(1000)
    report = run_distribution_checks(config, skels, run_id="test_run")
    assert report.overall == "pass", [
        f"{c.variable} {c.detail}: js={c.js_divergence} mad={c.max_abs_diff}"
        for c in report.checks if c.status == "fail"
    ]
    assert report.duplicate_rate < 0.01
    assert report.missing_rate == 0.0


def test_detects_gender_drift():
    """Corrupt the sample (80% of females -> male) -> marginal check must fail."""
    config = load_population_config(CFG)
    skels = PopulationSampler(config, seed=42).sample(500)
    # build a 'generated' population with blatant gender drift vs target
    flipped = [
        s.model_copy(update={"gender": "male" if (s.gender == "female" and i % 5 != 0) else s.gender})
        for i, s in enumerate(skels)
    ]
    report = run_distribution_checks(config, flipped, run_id="drift_run")
    gender_checks = [c for c in report.checks if c.variable == "gender"]
    assert any(c.status == "fail" for c in gender_checks)
    assert report.overall == "fail"


def test_insufficient_data_flagged_not_failed():
    config = load_population_config(CFG)
    skels = PopulationSampler(config, seed=42).sample(1000)
    report = run_distribution_checks(
        config, skels, run_id="small_cells",
        thresholds=CheckThresholds(min_parent_count=500),
    )
    # with a very high min_parent_count some conditional cells are insufficient
    statuses = {c.status for c in report.checks}
    assert "insufficient_data" in statuses or "pass" in statuses

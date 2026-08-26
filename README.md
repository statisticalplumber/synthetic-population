# synthetic-population

Synthetic population & behavioral simulation platform.

Build statistically controlled synthetic populations (demographic skeletons),
enrich them with latent behavioral attributes, simulate scenario responses,
and export validation + fine-tuning datasets — with full provenance and
quality metrics at every stage.

> **Safety**: everything in `data/` is synthetic. `data_label` is stamped on
> every record (`synthetic_mock` in Stage 1). Personas are statistical
> constructs, never real individuals, and must never be presented as such.

## Stage 1 (this repo, now)

Deterministic, seed-reproducible population sampling from a declarative
population config — **no LLM involved**.

1. **Population config** (`config/populations/*.yaml`) declares marginal and
   conditional distributions (country → emirate → city, age → education,
   education × employment → occupation, …). Configs are validated strictly:
   probabilities must sum to 1, no forward references, required fields present.
2. **Sampler** (`src/synthpop/population/sampler.py`) draws `n` skeletons with
   a seeded numpy RNG, sampling each variable from its (possibly conditional)
   distribution. Output: immutable `PersonaSkeleton` records with provenance.
3. **Validation** (`src/synthpop/validation/`) compares generated vs target
   distributions — chi-squared goodness-of-fit (sample-size aware), JS
   divergence and max-abs-diff as informational metrics — plus duplicate and
   missing-value checks.
4. **Storage** (`src/synthpop/storage/`): Parquet (partitioned by
   country/version/shard), JSONL for interop, run manifests.
5. **Enrichment scaffold** (`src/synthpop/persona/enrich.py` +
   `src/synthpop/llm/`): provider abstraction (mock / OpenAI-compatible),
   batch runner with retries, checkpoint/resume, cost accounting. A
   deterministic `MockProvider` runs offline; real LLM enrichment is a config
   change (`role: luna`), not a code change.

## Layout

```
config/
  generation.yaml          # pipeline settings (sizes, seeds, thresholds)
  models.yaml              # role -> provider/model mapping (env-var based)
  populations/uae_mock.yaml# synthetic population spec (clearly labeled)
schemas/                   # JSON schemas exported from pydantic models
src/synthpop/
  models/                  # pydantic: Persona, Scenario, SimulationResult, ...
  population/sampler.py    # seeded numpy sampler
  validation/              # distribution checks (chi2, JS, dup/missing)
  metrics/                 # quality metrics
  storage/                 # parquet/jsonl/manifest IO
  persona/enrich.py        # LLM enrichment (Stage 2 entry point)
  llm/                     # provider, batch runner, cost tracker
scripts/
  export_schemas.py
  generate_population.py
  validate_population.py
  enrich_personas.py
tests/
```

## Quickstart

Python: `/home/xyane/software/.venv/bin/python` (has pandas/pyarrow/scipy).

```bash
make e2e     # export schemas -> generate 1k skeletons -> validate -> enrich
make test    # pytest
```

Individual stages:

```bash
/home/xyane/software/.venv/bin/python scripts/generate_population.py
/home/xyane/software/.venv/bin/python scripts/validate_population.py
/home/xyane/software/.venv/bin/python scripts/enrich_personas.py
```

Outputs land in `data/generated/...` and `data/reports/<run_id>/`:

- `data/generated/skeletons/<run_id>/` — Parquet + JSONL + manifest
- `data/reports/<run_id>/validation_report.json` — all distribution checks
- `data/reports/<run_id>/metrics.json` — summary metrics
- `data/generated/personas/<run_id>/` — enriched personas (mock provider by default)

`<run_id>` encodes provenance: `ae_uae_mock_v1_n1000_s42`
(country_config_n_seed).

## Key design decisions

- **Source statistics → sampler → LLM enrichment**, never prompt → invented
  population. Target marginals are preserved by construction (conditional
  sampling), and verified by chi-squared checks after generation.
- **Config is the population.** Distributions live in YAML, validated by
  pydantic; the sampler is generic. New country = new YAML.
- **Provenance on every record**: provider, model, prompt_version,
  schema_version, population_config_version, seed, created_at, data_label.
- **Model roles (sol/terra/luna) are configurable** via `config/models.yaml`
  + env vars; no hardcoded endpoints or keys. Secrets come from the
  environment only (see `.env.example`).
- **Control plane vs data plane**: agentic decisions (which config to run,
  whether to scale up) stay outside; the pipeline itself is deterministic
  Python with batch LLM calls.
- **Scaling path**: 1k (now) → 10k → 100k → 1M. Parquet partitioning and
  batch checkpoint/resume are already in place for that path.

## Real LLM smoke test (verified)

Tested against a local model served by llama-server on port 8080
(LM Studio on port 1234 works the same way):

```bash
LUNA_BASE_URL=http://127.0.0.1:8080/v1 \
LUNA_MODEL='<served-model-path>' \
python scripts/test_provider.py --role luna --n 50 --out data/experiments/luna_smoke50/personas.jsonl
```

Result: 50/50 valid, 0 retries, ~3.7 s/req sequential.

**Endpoint capability check (do not assume)**: this llama-server build
accepts only `response_format: {"type": "text"|"json_object"}` —
`json_schema` is silently ignored (its 400 for an invalid type lists the
accepted values). The provider therefore: sends `json_object` (configurable
per role in `config/models.yaml`), embeds the JSON schema in the system
prompt, extracts the JSON object tolerantly (fences/prose), and uses
pydantic validation as the conformance gate (batch retries on failure).

Observed quality (n=50, gemma-4-12b-it Q4): latent values valid and unique
per persona, but compressed toward the middle (std 0.06–0.22 vs ~0.29 for
the mock; e.g. trust_propensity 0.55–0.78). Watch latent-variable variance
as a quality metric; consider prompt tuning or post-hoc rescaling before
using these for simulation.

## Layer 3: behavioral scenario simulation (MVP)

Simulates (persona, scenario) pairs through a configured provider role and
emits probabilistic `SimulationResult` records: a full probability
distribution over the scenario options, a confidence score, and up to 3
declared behavioral factors (known latent label + direction + strength —
**not** chain-of-thought reasoning). Every record carries full provenance
(provider, model, prompt version, scenario id/version, persona version,
inference params, seed, `created_at`, `data_label`).

```bash
# offline mock (deterministic, no API key)
make simulate PERSONAS=data/generated/personas/ae_uae_mock_v1_n1000_s42 ROLE=mock

# real provider (async, bounded concurrency)
ROLE=luna make simulate PERSONAS=data/generated/personas/ae_uae_mock_v1_n1000_s42
```

- `scripts/simulate.py` — CLI: `--personas --scenarios --role --out --limit
  --concurrency --max-retries --seed [--sync]` (concurrency/retries also from
  `config/generation.yaml: simulation:`).
- **Checkpoint/resume**: results are appended to
  `data/generated/simulations/<run_id>/simulations.jsonl` as each pair
  completes; re-running the same command resumes (idempotent by
  `persona_id|scenario_id@scenario_version`).
- **Deterministic validation** (Python, not LLM): option keys must match the
  scenario exactly, probabilities in [0,1] summing to ~1 (tol 0.01),
  confidence in [0,1], factors must be known persona latents with strength
  in [0,1]. Invalid output is retried as a transient failure.
- **Error classification**: timeouts, connect errors, 408/429/5xx and
  malformed output are retryable (exponential backoff + jitter); 400/401/403/
  404 fail immediately (config/auth problems, not worth retrying).
- **Metrics** (`metrics.json`): prob-sum failure rate, confidence
  distribution, option selection per scenario, probability entropy,
  persona/scenario response diversity, retry/failure/cost accounting.
- **Ensemble-ready**: one record = one model's prediction;
  `SimulationResult.ensemble_key()` groups identical (persona, scenario)
  predictions for mean/variance/disagreement aggregation (agreement is not
  ground truth).

### Issues to close before the first real Luna/Terra simulation

1. **Latent variance compression** (observed on gemma-4-12b-it enrichment):
   real-model latents compress toward the middle (std ~0.06–0.22 vs ~0.29
   mock). Simulations driven by such latents may be under-dispersed. Add
   per-variable latent variance checks to the pre-flight; prompt-tune or
   post-hoc rescale before trusting real-model simulation output.
2. **`response_format` capability varies by server** — probe, don't assume.
   The known local llama-server build accepts only `text`/`json_object`
   (`json_schema` silently ignored). Keep roles on `json_object` until the
   target server is probed; the pydantic validation gate is the real
   conformance check either way.
3. **Truncation vs `max_tokens`**: truncated JSON is extracted as a
   retryable failure; with a too-small `max_tokens` that becomes a permanent
   retry loop. Simulation outputs are small (~200–400 tokens); the default
   2048 is ample, but verify per role.
4. **Concurrency vs local-server workers**: llama-server is effectively
   single-flight by default; client-side `concurrency: 8` still holds 8
   in-flight HTTP connections. Tune `simulation.concurrency` to the server's
   actual worker count for real runs.
5. **Resume + metrics semantics**: on a resumed run, `metrics.json` batch/
   cost stats reflect that run's *incremental* batch, while distribution
   metrics cover the full stored set. For the first real run, either run to
   completion in one pass or recompute from the full JSONL.
6. **Costs**: mock token counts are estimates (`len//4`); real runs use the
   server `usage` field. Set `pricing_usd_per_1k` for the role before a
   paid run so `estimated_cost_usd` is meaningful.

## Status / next

- [x] Schemas, sampler, validation, storage, metrics, tests, E2E
- [x] Mock enrichment (deterministic, offline)
- [x] Real LLM enrichment path verified (12B local model, 50-persona smoke)
- [x] Scenario simulation MVP (schemas, prompts, deterministic validation,
      retries/checkpoint, mock path, 10-persona demo, metrics)
- [ ] Full 1,000-persona real-model enrichment run
- [ ] First real Luna/Terra simulation run (see README: issues to close first)
- [ ] Bias/stereotype audit metrics
- [ ] Fine-tuning dataset export (HF format)

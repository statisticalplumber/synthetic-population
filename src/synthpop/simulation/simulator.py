"""Layer 3 — behavioral scenario simulation service.

``BehavioralSimulator`` takes (persona, scenario) pairs and produces
probabilistic ``SimulationResult`` records through the existing
``LLMProvider`` abstraction (model selection via config/models.yaml).

- Real roles (luna/terra/sol): one structured call per (persona, scenario);
  the model returns probabilities + confidence + factor labels (no
  chain-of-thought). Output is deterministically validated before
  acceptance; retryable failures are retried with backoff.
- Mock role: a deterministic offline computation (``mock.py``) stands in
  for the LLM, so mock runs are reproducible and valid by construction.

Every result carries full provenance (provider, model, prompt version,
scenario id/version, persona version, inference params, seed, created_at,
data_label).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Iterable

from ..llm.batch import (
    BatchResult,
    arun_batch,
    run_batch,
    run_batch_async,
)
from ..llm.cost import CostTracker
from ..llm.provider import (
    LLMProvider,
    MockProvider,
    StructuredResponse,
)
from ..models.persona import Persona
from ..models.provenance import SimulationProvenance
from ..models.scenario import Scenario
from ..models.simulation import SimulationOutput, SimulationResult
from .mock import MOCK_MODEL_LABEL, mock_simulation_output
from .prompts import SIM_PROMPT_VERSION, SIM_SYSTEM_PROMPT, build_simulation_prompt
from .validation import (
    SimulationValidationError,
    validate_simulation_output,
    validate_simulation_result,
)


def pair_key(persona_id: str, scenario: Scenario) -> str:
    """Stable idempotency key for one (persona, scenario) simulation."""
    return f"{persona_id}|{scenario.full_id()}"


class BehavioralSimulator:
    """Runs (persona, scenario) simulations through an LLMProvider.

    Model selection is NOT here — the provider (and thus the model) comes
    from config/models.yaml via ``build_provider(role, ...)``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        seed: int = 42,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        jitter: float = 0.5,
        max_concurrency: int = 8,
        tracker: CostTracker | None = None,
    ):
        self.provider = provider
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.jitter = jitter
        self.max_concurrency = max_concurrency
        self.tracker = tracker or CostTracker(provider.name)

    # -- single-pair generation ---------------------------------------------

    def _mock_response(self, persona: Persona, scenario: Scenario) -> StructuredResponse:
        out = mock_simulation_output(persona, scenario, self.seed)
        raw = json.dumps(out.model_dump(mode="json"), sort_keys=True)
        return StructuredResponse(
            data=out,
            raw=raw,
            model=MOCK_MODEL_LABEL,
            prompt_tokens=len(build_simulation_prompt(persona, scenario)) // 4,
            completion_tokens=len(raw) // 4,
        )

    def _call(self, persona: Persona, scenario: Scenario) -> StructuredResponse:
        """One generation (sync). MockProvider path is deterministic/offline."""
        if isinstance(self.provider, MockProvider):
            return self._mock_response(persona, scenario)
        return self.provider.complete_structured(
            system=SIM_SYSTEM_PROMPT,
            user=build_simulation_prompt(persona, scenario),
            schema=SimulationOutput,
        )

    async def _acall(self, persona: Persona, scenario: Scenario) -> StructuredResponse:
        """One generation (async). MockProvider path is deterministic/offline."""
        if isinstance(self.provider, MockProvider):
            return self._mock_response(persona, scenario)
        return await self.provider.acomplete_structured(
            system=SIM_SYSTEM_PROMPT,
            user=build_simulation_prompt(persona, scenario),
            schema=SimulationOutput,
        )

    def _build_result(
        self,
        persona: Persona,
        scenario: Scenario,
        out: SimulationOutput,
        resp: StructuredResponse,
    ) -> SimulationResult:
        choice = max(out.probabilities, key=out.probabilities.get)
        model_params = {
            str(k): v
            for k, v in (getattr(self.provider, "default_params", None) or {}).items()
        }
        provenance = SimulationProvenance(
            provider=self.provider.name,
            model=resp.model,
            simulator_prompt_version=SIM_PROMPT_VERSION,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.scenario_version,
            persona_version=persona.provenance.population_config_version,
            model_params=model_params,
            seed=self.seed,
            data_label=persona.data_label,
        )
        result = SimulationResult.model_validate(
            {
                "persona_id": persona.persona_id,
                "scenario_id": scenario.scenario_id,
                "scenario_version": scenario.scenario_version,
                "choice": choice,
                "probabilities": out.probabilities,
                "confidence": out.confidence,
                "behavioral_factors": [f.model_dump() for f in out.behavioral_factors],
                "provenance": provenance.model_dump(),
            }
        )
        validate_simulation_result(result, persona, scenario)
        return result

    def simulate_one(self, persona: Persona, scenario: Scenario) -> SimulationResult:
        """Simulate one (persona, scenario) pair (sync, no batching)."""
        resp = self._call(persona, scenario)
        out: SimulationOutput = resp.data
        validate_simulation_output(out, persona, scenario)
        return self._build_result(persona, scenario, out, resp)

    async def asimulate_one(
        self, persona: Persona, scenario: Scenario
    ) -> SimulationResult:
        resp = await self._acall(persona, scenario)
        out: SimulationOutput = resp.data
        validate_simulation_output(out, persona, scenario)
        return self._build_result(persona, scenario, out, resp)

    # -- batch ----------------------------------------------------------------

    def _afn_factory(self):
        async def fn(pair: tuple[Persona, Scenario]) -> StructuredResponse:
            persona, scenario = pair
            resp = await self._acall(persona, scenario)
            validate_simulation_output(resp.data, persona, scenario)
            return resp
        return fn

    def _fn_factory(self):
        def fn(pair: tuple[Persona, Scenario]) -> StructuredResponse:
            persona, scenario = pair
            resp = self._call(persona, scenario)
            validate_simulation_output(resp.data, persona, scenario)
            return resp
        return fn

    def _pairs(
        self,
        personas: list[Persona],
        scenarios: list[Scenario],
    ) -> list[tuple[str, tuple[Persona, Scenario]]]:
        return [
            (pair_key(p.persona_id, s), (p, s))
            for p in personas
            for s in scenarios
        ]

    def _incremental_hook(
        self,
        items: list[tuple[str, tuple[Persona, Scenario]]],
        on_result: Callable[[SimulationResult], None] | None,
    ):
        """Batch-level callback: build the result per completed pair.

        Fires DURING the batch (as each pair completes) — this is the
        incremental checkpoint hook. Returns (callback, built_map).
        """
        by_key = {k: (p, s) for k, (p, s) in items}
        built: dict[str, SimulationResult] = {}

        def _cb(key: str, resp: StructuredResponse) -> None:
            persona, scenario = by_key[key]
            built[key] = self._build_result(persona, scenario, resp.data, resp)
            if on_result:
                on_result(built[key])

        return _cb, built

    async def asimulate_batch(
        self,
        personas: list[Persona],
        scenarios: list[Scenario],
        *,
        done: set[str] | None = None,
        on_result: Callable[[SimulationResult], None] | None = None,
    ) -> tuple[list[SimulationResult], BatchResult]:
        """Simulate all (persona, scenario) pairs with bounded concurrency.

        Returns (results, batch_result). ``done`` = keys already completed
        (checkpoint/resume); skipped keys produce no result here — the caller
        merges with previously stored results. ``on_result`` receives each
        built ``SimulationResult`` as it completes (incremental checkpointing).
        """
        items = self._pairs(personas, scenarios)
        fn = self._afn_factory()
        cb, built = self._incremental_hook(items, on_result)
        batch = await arun_batch(
            self.provider,
            items,
            fn,
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_cap_s=self.backoff_cap_s,
            jitter=self.jitter,
            max_concurrency=self.max_concurrency,
            tracker=self.tracker,
            done=done,
            on_result=cb,
        )
        results = [built[k] for k, _ in batch.ok]
        return results, batch

    def simulate_batch(
        self,
        personas: list[Persona],
        scenarios: list[Scenario],
        *,
        done: set[str] | None = None,
        on_result: Callable[[SimulationResult], None] | None = None,
    ) -> tuple[list[SimulationResult], BatchResult]:
        """Simulate all (persona, scenario) pairs (sequential, sync).

        For high volume / real providers prefer ``asimulate_batch`` (bounded
        concurrency via async HTTP); this sync path is fine for the offline
        mock and small runs.
        """
        items = self._pairs(personas, scenarios)
        cb, built = self._incremental_hook(items, on_result)
        batch = run_batch(
            self.provider,
            items,
            self._fn_factory(),
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_cap_s=self.backoff_cap_s,
            jitter=self.jitter,
            tracker=self.tracker,
            done=done,
            on_result=cb,
        )
        results = [built[k] for k, _ in batch.ok]
        return results, batch

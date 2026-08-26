"""Stage 2 — persona enrichment (LLM layer).

Enriches immutable Layer-1 skeletons with Layer-2 latent attributes via a
configurable provider. Demographic fields are NEVER rewritten: the LLM only
produces `LatentAttributes`, which are merged into a new `Persona`.

Prompts are versioned strings; the version is stored in provenance.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.batch import BatchResult, run_batch
from ..llm.cost import CostTracker
from ..llm.provider import LLMProvider
from ..models.config import PopulationConfig
from ..models.persona import LatentAttributes, Persona, PersonaSkeleton
from ..models.provenance import GenerationMetadata

PROMPT_VERSION = "enrich_v1"

ENRICH_SYSTEM_PROMPT = (
    "You infer latent behavioral attributes for a synthetic persona. "
    "Return ONLY the JSON object defined by the schema. Values are floats in [0,1]. "
    "Base inferences on the demographic skeleton, but keep variance high: "
    "demographics are weak predictors of individual behavior. "
    "Do not produce prose."
)

_DEMO_FIELDS = (
    "country", "emirate", "city", "urban_rural", "age", "age_band", "gender",
    "marital_status", "education", "employment_status", "occupation_group",
    "income_band", "household_size", "housing_status",
)


def _enrich_one(skeleton: PersonaSkeleton) -> str:
    payload = {k: skeleton.model_dump()[k] for k in _DEMO_FIELDS}
    return json.dumps(payload, ensure_ascii=False)


def enrich_skeletons(
    skeletons: list[PersonaSkeleton],
    provider: LLMProvider,
    config: PopulationConfig,
    *,
    tracker: CostTracker | None = None,
    max_retries: int = 3,
    done: set[str] | None = None,
) -> tuple[list[Persona], BatchResult]:
    """Enrich all skeletons. Returns (personas, batch_result).

    `done` holds persona_ids already enriched (checkpoint/resume).
    """
    tracker = tracker or CostTracker(provider.name)
    personas: list[Persona] = []
    results: dict[str, Any] = {}

    def fn(skel: PersonaSkeleton):
        resp = provider.complete_structured(
            system=ENRICH_SYSTEM_PROMPT,
            user=_enrich_one(skel),
            schema=LatentAttributes,
        )
        return resp

    batch = run_batch(
        provider,
        [(s.persona_id, s) for s in skeletons],
        fn,
        max_retries=max_retries,
        tracker=tracker,
        done=done,
    )
    results = dict(batch.ok)

    for skel in skeletons:
        if skel.persona_id in results:
            resp = results[skel.persona_id]
            latent: LatentAttributes = resp.data
            skel_dict = skel.model_dump()
            personas.append(
                Persona.model_validate(
                    {
                        **skel_dict,
                        "latent": latent.model_dump(),
                        "enrichment": GenerationMetadata(
                            pipeline_stage="enrichment",
                            provider=provider.name,
                            model=resp.model,
                            prompt_version=PROMPT_VERSION,
                            population_config_version=skel.provenance.population_config_version,
                            seed=skel.provenance.seed,
                            data_label=skel.data_label,
                        ).model_dump(),
                    }
                )
            )
    return personas, batch

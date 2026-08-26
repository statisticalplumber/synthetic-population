"""LLM provider abstraction (data-plane model access).

Model selection is configurable (config/models.yaml), never hardcoded:
roles (sol / terra / luna / mock) map to provider + model + params, and
credentials come exclusively from environment variables.

- ``MockProvider``: deterministic, offline, schema-aware. Used for
  development, tests, and CI. Clearly labeled; never presented as a real LLM.
- ``OpenAICompatibleProvider``: any OpenAI-compatible /v1 endpoint
  (LM Studio, llama-server, vLLM, hosted APIs).
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..models.config import PopulationConfig  # noqa: F401  (re-export convenience)


class StructuredResponse(BaseModel):
    """Result of one structured generation call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: BaseModel  # parsed & validated instance
    raw: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    """Abstract provider. `name` is the role label (sol/terra/luna/mock/...)."""

    name: str = "abstract"

    @abstractmethod
    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        """Generate a JSON object conforming to `schema` and return it parsed."""


class MockProvider(LLMProvider):
    """Deterministic offline provider.

    Synthesizes a schema-conformant instance from the pydantic model's field
    types, seeded by a hash of the prompt. Same input => same output, so
    batch jobs are reproducible and restart-safe.
    """

    name = "mock"

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        seed = int.from_bytes(
            hashlib.sha256(f"{schema.__name__}|{user}".encode()).digest()[:8], "big"
        )
        import numpy as np

        rng = np.random.default_rng(seed)
        instance: dict[str, Any] = {}
        for fname, f in schema.model_fields.items():
            instance[fname] = self._mock_value(f, rng)
        obj = schema.model_validate(instance)  # raises if mock is inconsistent
        raw = json.dumps(obj.model_dump(mode="json"), sort_keys=True)
        return StructuredResponse(
            data=obj, raw=raw, model="mock/deterministic",
            prompt_tokens=len(user) // 4, completion_tokens=len(raw) // 4,
        )

    @staticmethod
    def _mock_value(f, rng) -> Any:
        from typing import Optional as TypingOptional

        ann = f.annotation
        origin = getattr(ann, "__origin__", None)
        if origin is TypingOptional:
            args = getattr(ann, "__args__", (None,))
            inner = args[0] if args and args[0] is not None else float
            # mock fills optionals with a value (simplifies downstream checks)
            ann = inner
        meta = f.metadata
        lo = 0.0
        hi = 1.0
        for m in meta:
            if getattr(m, "ge", None) is not None:
                lo = float(m.ge)
            if getattr(m, "le", None) is not None:
                hi = float(m.le)
        if ann is float or (origin is None and ann in (float,)):
            return round(float(rng.uniform(lo, hi)), 4)
        if ann is int:
            return int(rng.integers(int(lo), int(hi) + 1))
        if ann is bool:
            return bool(rng.integers(0, 2))
        if ann is str:
            return "mock"
        if origin is list:
            return []
        if origin is dict:
            return {}
        return None


class OpenAICompatibleProvider(LLMProvider):
    """Any /v1/chat/completions endpoint (LM Studio, llama-server, vLLM, ...).

    Credentials come from environment variables only — never from config files.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 120.0,
        default_params: dict[str, Any] | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.default_params = default_params or {}

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        api_key = os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model,
            "temperature": (params or self.default_params).get("temperature", 0.2),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        raw = body["choices"][0]["message"]["content"]
        obj = schema.model_validate_json(raw)
        usage = body.get("usage", {})
        return StructuredResponse(
            data=obj,
            raw=raw,
            model=body.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def build_provider(role: str, models_config: dict[str, Any]) -> LLMProvider:
    """Build a provider for a role (sol/terra/luna/mock) from models.yaml data."""
    if role not in models_config.get("roles", {}):
        raise KeyError(f"role {role!r} not defined in models config")
    rc = models_config["roles"][role]
    provider_type = rc.get("provider", "mock")
    if provider_type == "mock":
        return MockProvider()
    if provider_type == "openai_compatible":
        base_url = os.environ.get(rc.get("base_url_env", ""), rc.get("default_base_url", ""))
        model = os.environ.get(rc.get("model_env", ""), rc.get("default_model", ""))
        return OpenAICompatibleProvider(
            name=role,
            base_url=base_url,
            model=model,
            api_key_env=rc.get("api_key_env", "OPENAI_API_KEY"),
            timeout_s=float(rc.get("timeout_s", 120.0)),
            default_params={
                "temperature": rc.get("temperature", 0.2),
                "max_tokens": rc.get("max_tokens", 2048),
            },
        )
    raise ValueError(f"unknown provider type {provider_type!r} for role {role!r}")

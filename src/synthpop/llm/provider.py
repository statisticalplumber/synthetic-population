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


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from model output.

    Tolerates markdown code fences and leading/trailing prose: finds the
    first balanced top-level object. Raises ValueError if none parses.
    """
    text = raw.strip()
    # strip a single markdown fence if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # fallback: first balanced {...} block
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object found in model output: {raw[:200]!r}")


class OpenAICompatibleProvider(LLMProvider):
    """Any /v1/chat/completions endpoint (LM Studio, llama-server, vLLM, ...).

    Credentials come from environment variables only — never from config files.

    `response_format` is configurable because support varies by server:
    - "json_object":  widely supported (llama-server, LM Studio, vLLM); the
      server guarantees valid JSON but NOT schema conformance.
    - "json_schema":  strict structured output (OpenAI, newer vLLM); not
      supported by most llama-server builds (silently ignored or 400).
    - "none":         no response_format; prompt-only.
    In all cases the JSON schema is embedded in the system prompt and the
    output is validated with pydantic — validation is the real gate.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 120.0,
        default_params: dict[str, Any] | None = None,
        response_format: str = "json_object",
    ):
        if response_format not in ("none", "json_object", "json_schema"):
            raise ValueError(f"invalid response_format: {response_format!r}")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.default_params = default_params or {}
        self.response_format = response_format

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

        schema_json = schema.model_json_schema()
        schema_json.pop("$defs", None)  # keep the prompt compact
        system_full = (
            f"{system}\n\nRespond with a single JSON object (no prose, no markdown) "
            f"conforming exactly to this JSON schema:\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": (params or self.default_params).get("temperature", 0.2),
            "messages": [
                {"role": "system", "content": system_full},
                {"role": "user", "content": user},
            ],
        }
        if self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json},
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
        obj = extract_json_object(raw)
        validated = schema.model_validate(obj)  # the real conformance gate
        usage = body.get("usage", {})
        return StructuredResponse(
            data=validated,
            raw=raw,
            model=body.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def probe_response_format(
    base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 30.0
) -> str:
    """Detect which response_format types a /v1 endpoint accepts.

    Returns "json_schema", "json_object", or "none". Uses a deliberately
    invalid type and reads the server's error message (OpenAI-compatible
    servers list the accepted types), falling back to direct probes.
    """
    api_key = os.environ.get(api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "response_format": {"type": "__probe__"},
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        r = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions", json=payload,
            headers=headers, timeout=timeout_s,
        )
    except httpx.HTTPError:
        return "none"
    if r.status_code == 400:
        msg = r.text.lower()
        if "json_schema" in msg:
            return "json_schema"
        if "json_object" in msg:
            return "json_object"
        return "none"
    return "json_object"  # server accepted everything; assume at least json_object


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
            response_format=rc.get("response_format", "json_object"),
        )
    raise ValueError(f"unknown provider type {provider_type!r} for role {role!r}")

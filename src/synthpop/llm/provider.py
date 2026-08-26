"""LLM provider abstraction (data-plane model access).

Model selection is configurable (config/models.yaml), never hardcoded:
roles (sol / terra / luna / mock) map to provider + model + params, and
credentials come exclusively from environment variables.

- ``MockProvider``: deterministic, offline, schema-aware. Used for
  development, tests, and CI. Clearly labeled; never presented as a real LLM.
- ``OpenAICompatibleProvider``: any OpenAI-compatible /v1 endpoint
  (LM Studio, llama-server, vLLM, hosted APIs). Sync (httpx) and async
  (httpx.AsyncClient) paths; all merged inference params (temperature,
  max_tokens, top_p, ...) are forwarded to the endpoint.

Error taxonomy (for retry classification in the batch runner):
- ``RetryableLLMError``: timeouts, connection failures, 408/429/5xx,
  transient malformed output. Safe to retry with backoff.
- ``NonRetryableLLMError``: 401/403 (auth), 404 (endpoint), 400/other 4xx
  (invalid schema/config, unsupported model). Retrying is pointless.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..models.config import PopulationConfig  # noqa: F401  (re-export convenience)


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


class LLMError(ValueError):
    """Base class for provider errors (ValueError for back-compat)."""


class RetryableLLMError(LLMError):
    """Transient failure: safe to retry with backoff."""


class NonRetryableLLMError(LLMError):
    """Permanent failure: retrying will not help (auth, bad endpoint, ...)."""


def classify_http_status(status_code: int) -> type[LLMError]:
    """Map an HTTP status code to an error class (retryability)."""
    if status_code in (408, 429) or 500 <= status_code < 600:
        return RetryableLLMError
    return NonRetryableLLMError


def classify_transport_error() -> type[LLMError]:
    """Transport-level failures (timeout, connect error) are retryable."""
    return RetryableLLMError


# ---------------------------------------------------------------------------
# structured response
# ---------------------------------------------------------------------------


class StructuredResponse(BaseModel):
    """Result of one structured generation call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: BaseModel  # parsed & validated instance
    raw: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# JSON schema preparation (nested-schema safe)
# ---------------------------------------------------------------------------


def inline_local_refs(schema: dict) -> dict:
    """Replace local ``$ref`` pointers (``#/$defs/...``) with the referenced
    definition, inlining recursively.

    This is the safe alternative to dropping ``$defs``: a schema with nested
    pydantic models contains ``$ref``s that would dangle if ``$defs`` were
    removed. Inlining keeps the schema valid and self-contained.
    """
    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                target = schema
                for part in ref[2:].split("/"):
                    target = target[part]
                # resolve the definition, but keep any local overrides
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def prepare_prompt_schema(schema: Type[BaseModel]) -> dict:
    """JSON schema for prompt embedding / json_schema response_format.

    Inlines local ``$ref``s and drops the now-unnecessary ``$defs`` so no
    dangling references remain.
    """
    schema_json = schema.model_json_schema()
    schema_json = inline_local_refs(schema_json)
    schema_json.pop("$defs", None)
    return schema_json


# ---------------------------------------------------------------------------
# provider interface
# ---------------------------------------------------------------------------


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

    async def acomplete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        """Async generation. Default: run the sync path in a worker thread."""
        return await asyncio.to_thread(
            self.complete_structured,
            system=system, user=user, schema=schema, params=params,
        )

    async def aclose(self) -> None:
        """Release async resources (async HTTP client). Default: no-op."""


# ---------------------------------------------------------------------------
# mock provider
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from model output.

    Tolerates markdown code fences and leading/trailing prose: finds the
    first balanced top-level object. Raises RetryableLLMError if none parses
    (transient malformed output — a retry may fix it).
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
    raise RetryableLLMError(f"no JSON object found in model output: {raw[:200]!r}")


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


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

    All merged inference params (``default_params`` overridden by per-call
    ``params``) are forwarded to the endpoint — including ``max_tokens``.
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
        transport: httpx.BaseTransport | None = None,
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
        self._transport = transport
        self._async_client: httpx.AsyncClient | None = None

    # -- request building (shared by sync/async) -----------------------------

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _merged_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.default_params)
        if params:
            merged.update(params)
        return merged

    def _build_payload(
        self,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        schema_json = prepare_prompt_schema(schema)
        system_full = (
            f"{system}\n\nRespond with a single JSON object (no prose, no markdown) "
            f"conforming exactly to this JSON schema:\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        )

        merged = self._merged_params(params)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_full},
                {"role": "user", "content": user},
            ],
        }
        # forward ALL inference params (temperature, max_tokens, top_p, ...)
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty",
                    "presence_penalty", "top_k", "repetition_penalty"):
            if key in merged:
                payload[key] = merged[key]

        if self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json},
            }
        return payload

    def _parse_response(
        self, body: dict[str, Any], schema: Type[BaseModel]
    ) -> StructuredResponse:
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

    # -- sync path ------------------------------------------------------------

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        payload = self._build_payload(system, user, schema, params)
        try:
            with httpx.Client(
                base_url=self.base_url,
                transport=self._transport,  # None => real network
                timeout=self.timeout_s,
            ) as client:
                resp = client.post(
                    "/chat/completions", json=payload, headers=self._headers()
                )
        except httpx.TimeoutException as e:
            raise RetryableLLMError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise classify_transport_error()(f"transport error: {e}") from e

        if resp.status_code != 200:
            raise classify_http_status(resp.status_code)(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            body = resp.json()
            return self._parse_response(body, schema)
        except (KeyError, TypeError, RetryableLLMError) as e:
            raise RetryableLLMError(f"malformed response body: {e}") from e

    # -- async path -----------------------------------------------------------

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                transport=self._transport,
                timeout=self.timeout_s,
            )
        return self._async_client

    async def acomplete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[BaseModel],
        params: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        payload = self._build_payload(system, user, schema, params)
        client = self._get_async_client()
        try:
            resp = await client.post(
                "/chat/completions", json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as e:
            raise RetryableLLMError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise classify_transport_error()(f"transport error: {e}") from e

        if resp.status_code != 200:
            raise classify_http_status(resp.status_code)(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            body = resp.json()
            return self._parse_response(body, schema)
        except (KeyError, TypeError, RetryableLLMError) as e:
            raise RetryableLLMError(f"malformed response body: {e}") from e

    async def aclose(self) -> None:
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
        self._async_client = None


# ---------------------------------------------------------------------------
# response_format probing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


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

"""Batch execution with classified retries, backoff, and bounded concurrency.

Design:
- ``run_batch`` (sync) and ``arun_batch`` (async) iterate items, call
  ``fn(item)``, and retry *retryable* failures with exponential backoff +
  jitter. ``NonRetryableLLMError`` (auth, bad endpoint, unsupported model,
  invalid local config) fails immediately — no retry.
- Idempotent + restartable: the caller passes a ``done`` set of already-
  completed keys; batch resumes where it left off.
- Every attempt (success or failure) is recorded in a CostTracker.
- Async path: bounded in-flight requests (semaphore), input-order results,
  clean shutdown (cancels in-flight tasks on cancellation).

``retries`` counts *performed* retries (failed attempts that were followed by
another attempt for the same item). ``attempts`` counts all attempts.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from .cost import CostTracker
from .provider import LLMProvider, NonRetryableLLMError, StructuredResponse

log = logging.getLogger("synthpop.batch")


@dataclass
class BatchResult:
    ok: list[tuple[str, Any]] = field(default_factory=list)
    failed: list[tuple[str, Exception]] = field(default_factory=list)
    attempts: int = 0
    retries: int = 0


def backoff_delay(
    attempt: int,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 60.0,
    jitter: float = 0.5,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with jitter.

    delay = min(cap, base * 2**attempt) * uniform(1 - jitter, 1.0)
    ``jitter=0`` gives deterministic (test-friendly) delays.
    """
    delay = min(backoff_cap_s, backoff_base_s * (2 ** attempt))
    r = rng or random
    return delay * r.uniform(1.0 - jitter, 1.0)


def _attempt(
    key: str,
    item: Any,
    fn: Callable[[Any], Any],
    *,
    max_retries: int,
    backoff_base_s: float,
    backoff_cap_s: float,
    jitter: float,
    tracker: CostTracker,
    result: BatchResult,
    on_result: Callable[[str, StructuredResponse], None] | None,
) -> tuple[str, Any, Exception | None]:
    """Run one item with retries (sync). Returns (key, resp|None, err|None)."""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        result.attempts += 1
        try:
            resp = fn(item)
            tracker.record_success(resp.prompt_tokens, resp.completion_tokens)
            if on_result:
                on_result(key, resp)
            return key, resp, None
        except NonRetryableLLMError as e:
            tracker.record_failure()
            log.error(
                "batch item %s non-retryable failure (%s: %s); not retrying",
                key, type(e).__name__, e,
            )
            return key, None, e
        except Exception as e:  # noqa: BLE001 - classified retryable by default
            tracker.record_failure()
            if attempt < max_retries:
                result.retries += 1
                sleep_s = backoff_delay(attempt, backoff_base_s, backoff_cap_s, jitter)
                log.warning(
                    "batch item %s attempt %d failed (%s: %s); retrying in %.2fs",
                    key, attempt + 1, type(e).__name__, e, sleep_s,
                )
                time.sleep(sleep_s)
            else:
                log.error(
                    "batch item %s permanently failed: %s: %s",
                    key, type(e).__name__, e,
                )
            last_err = e
    return key, None, last_err


def run_batch(
    provider: LLMProvider,
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], StructuredResponse],
    *,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 60.0,
    jitter: float = 0.5,
    tracker: CostTracker | None = None,
    done: set[str] | None = None,
    on_result: Callable[[str, StructuredResponse], None] | None = None,
) -> BatchResult:
    """Run ``fn(item)`` for each (key, item), with classified retries.

    ``fn`` should raise on invalid/unparseable output so retries can kick in.
    ``done`` = keys already completed (checkpoint/resume support).
    """
    tracker = tracker or CostTracker(provider.name)
    done = done or set()
    result = BatchResult()

    for key, item in items:
        if key in done:
            continue
        _, resp, err = _attempt(
            key, item, fn,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
            jitter=jitter,
            tracker=tracker,
            result=result,
            on_result=on_result,
        )
        if resp is not None:
            result.ok.append((key, resp))
        if err is not None:
            result.failed.append((key, err))
    return result


async def arun_batch(
    provider: LLMProvider,
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], Awaitable[StructuredResponse]],
    *,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 60.0,
    jitter: float = 0.5,
    max_concurrency: int = 8,
    tracker: CostTracker | None = None,
    done: set[str] | None = None,
    on_result: Callable[[str, StructuredResponse], None] | None = None,
) -> BatchResult:
    """Async batch runner with bounded concurrency.

    - ``fn(item)`` must be a coroutine (or awaitable) returning
      ``StructuredResponse``; raise on failure so retries kick in.
    - ``max_concurrency`` bounds in-flight requests (semaphore).
    - Results are returned in input order (deterministic despite concurrency).
    - Clean shutdown: on cancellation, in-flight tasks are cancelled and
      awaited before the ``CancelledError`` propagates.
    """
    items = list(items)
    tracker = tracker or CostTracker(getattr(provider, "name", "unknown"))
    done = done or set()
    result = BatchResult()
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def worker(key: str, item: Any):
        async with sem:
            last_err: Exception | None = None
            resp = None
            for attempt in range(max_retries + 1):
                result.attempts += 1
                try:
                    resp = await fn(item)
                    tracker.record_success(resp.prompt_tokens, resp.completion_tokens)
                    if on_result:
                        on_result(key, resp)
                    return key, resp, None
                except NonRetryableLLMError as e:
                    tracker.record_failure()
                    log.error(
                        "batch item %s non-retryable failure (%s: %s); not retrying",
                        key, type(e).__name__, e,
                    )
                    return key, None, e
                except Exception as e:  # noqa: BLE001 - classified retryable by default
                    tracker.record_failure()
                    if attempt < max_retries:
                        result.retries += 1
                        delay = backoff_delay(attempt, backoff_base_s, backoff_cap_s, jitter)
                        log.warning(
                            "batch item %s attempt %d failed (%s: %s); retrying in %.2fs",
                            key, attempt + 1, type(e).__name__, e, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        log.error(
                            "batch item %s permanently failed: %s: %s",
                            key, type(e).__name__, e,
                        )
                    last_err = e
            return key, None, last_err

    todo = [(k, it) for k, it in items if k not in done]
    tasks = [asyncio.create_task(worker(k, it)) for k, it in todo]
    try:
        outcomes = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # Clean shutdown: cancel in-flight work, let it settle, then propagate.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    by_key = {o[0]: o for o in outcomes}
    for key, _ in items:
        if key in done:
            continue
        _, resp, err = by_key[key]
        if resp is not None:
            result.ok.append((key, resp))
        if err is not None:
            result.failed.append((key, err))
    return result


def run_batch_async(
    provider: LLMProvider,
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], Awaitable[StructuredResponse]],
    **kwargs: Any,
) -> BatchResult:
    """Sync wrapper around ``arun_batch`` (runs its own event loop).

    Closes the provider's async resources on the way out (clean shutdown).
    """
    async def _run() -> BatchResult:
        try:
            return await arun_batch(provider, items, fn, **kwargs)
        finally:
            aclose = getattr(provider, "aclose", None)
            if aclose is not None:
                res = aclose()
                if inspect.isawaitable(res):
                    await res

    return asyncio.run(_run())

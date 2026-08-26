"""Batch execution with retries and structured-output validation.

Design:
- `run_batch` iterates items, calls `fn(item)`, and retries on
  validation/transport errors with exponential backoff.
- Idempotent + restartable: the caller passes a `done` set of already-
  completed keys; batch resumes where it left off.
- Every attempt (success or failure) is recorded in a CostTracker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .cost import CostTracker
from .provider import LLMProvider, StructuredResponse

log = logging.getLogger("synthpop.batch")


@dataclass
class BatchResult:
    ok: list[tuple[str, Any]] = field(default_factory=list)
    failed: list[tuple[str, Exception]] = field(default_factory=list)
    attempts: int = 0
    retries: int = 0


def run_batch(
    provider: LLMProvider,
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], StructuredResponse],
    *,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    tracker: CostTracker | None = None,
    done: set[str] | None = None,
    on_result: Callable[[str, StructuredResponse], None] | None = None,
) -> BatchResult:
    """Run `fn(item)` for each (key, item), with retries.

    fn should raise on invalid/unparseable output so retries can kick in.
    `done` = keys already completed (checkpoint/resume support).
    """
    tracker = tracker or CostTracker(provider.name)
    done = done or set()
    result = BatchResult()

    for key, item in items:
        if key in done:
            continue
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            result.attempts += 1
            try:
                resp = fn(item)
                tracker.record_success(resp.prompt_tokens, resp.completion_tokens)
                result.ok.append((key, resp))
                if on_result:
                    on_result(key, resp)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 - retry any failure, log detail
                last_err = e
                tracker.record_failure()
                result.retries += 1
                if attempt < max_retries:
                    sleep_s = backoff_base_s * (2 ** attempt)
                    log.warning(
                        "batch item %s attempt %d failed (%s: %s); retrying in %.1fs",
                        key, attempt + 1, type(e).__name__, e, sleep_s,
                    )
                    time.sleep(sleep_s)
        if last_err is not None:
            log.error("batch item %s permanently failed: %s: %s", key, type(last_err).__name__, last_err)
            result.failed.append((key, last_err))
    return result

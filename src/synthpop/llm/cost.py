"""Token & cost accounting.

Per-provider prices (USD per 1k tokens) come from config/models.yaml so cost
is observable without hardcoding. Failed-request cost is tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    provider: str
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    requests: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failed_prompt_tokens: int = 0
    failed_completion_tokens: int = 0

    def record_success(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def record_failure(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.requests += 1
        self.failures += 1
        self.failed_prompt_tokens += prompt_tokens
        self.failed_completion_tokens += completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        ok = (
            self.prompt_tokens / 1000 * self.price_per_1k_prompt
            + self.completion_tokens / 1000 * self.price_per_1k_completion
        )
        failed = (
            self.failed_prompt_tokens / 1000 * self.price_per_1k_prompt
            + self.failed_completion_tokens / 1000 * self.price_per_1k_completion
        )
        return ok + failed

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "requests": self.requests,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failed_prompt_tokens": self.failed_prompt_tokens,
            "failed_completion_tokens": self.failed_completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def cost_per_record(tracker: CostTracker, n_records: int) -> float:
    if n_records <= 0:
        return 0.0
    return tracker.estimated_cost_usd / n_records

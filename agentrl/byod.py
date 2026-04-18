"""Public bring-your-own-data task helpers for AgentRL."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

from agentrl.core.base import BaseEnvironment, BaseVerifier


RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class BYODRecord:
    """One user-owned training or evaluation record."""

    input: str
    reference_answer: str
    supervised_target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

PromptFormatter = Callable[[RecordT, Any | None], str]
RewardFn = Callable[[str, dict[str, Any]], float]
SupervisedTargetFn = Callable[[RecordT], str | None]
StateBuilder = Callable[[RecordT], dict[str, Any]]


@dataclass(slots=True)
class BYODTask:
    """Thin adapter that exposes the environment/verifier pair."""

    environment: BaseEnvironment
    verifier: BaseVerifier
    _supervised_samples_fn: Callable[[Any | None], list[tuple[str, str]]]

    def supervised_samples(self, tokenizer: Any | None = None) -> list[tuple[str, str]]:
        return self._supervised_samples_fn(tokenizer)


class _SingleTurnEnvironment(BaseEnvironment):
    def __init__(
        self,
        records: Sequence[RecordT],
        prompt_formatter: PromptFormatter,
        state_builder: StateBuilder,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("BYOD tasks require at least one record.")
        self._records = list(records)
        self._prompt_formatter = prompt_formatter
        self._state_builder = state_builder
        self._rng = random.Random(seed)
        self._current_record: RecordT | None = None

    def reset(self) -> str:
        self._current_record = self._rng.choice(self._records)
        return self._prompt_formatter(self._current_record, None)

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, Any]:
        if self._current_record is None:
            raise RuntimeError("reset() must be called before state().")
        return self._state_builder(self._current_record)


class _HookVerifier(BaseVerifier):
    def __init__(self, reward_fn: RewardFn) -> None:
        self._reward_fn = reward_fn

    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        reward = float(self._reward_fn(response, env_state))
        return min(1.0, max(0.0, reward))


def make_single_turn_task(
    *,
    records: Sequence[RecordT],
    prompt_formatter: PromptFormatter[RecordT],
    reward_fn: RewardFn,
    state_builder: StateBuilder[RecordT] | None = None,
    supervised_target_fn: SupervisedTargetFn[RecordT] | None = None,
    seed: int = 0,
) -> BYODTask:
    """Build a single-turn task from user-owned records."""

    def default_state_builder(record: RecordT) -> dict[str, Any]:
        if not isinstance(record, BYODRecord):
            raise ValueError("state_builder is required for custom record types.")
        return {
            "input": record.input,
            "reference_answer": record.reference_answer,
            "metadata": dict(record.metadata),
        }

    if state_builder is None:
        for record in records:
            if not isinstance(record, BYODRecord):
                raise ValueError("state_builder is required for custom record types.")

    environment = _SingleTurnEnvironment(
        records=records,
        prompt_formatter=prompt_formatter,
        state_builder=state_builder or default_state_builder,
        seed=seed,
    )
    verifier = _HookVerifier(reward_fn)

    def build_samples(tokenizer: Any | None) -> list[tuple[str, str]]:
        if supervised_target_fn is None:
            raise ValueError("No supervised target hook configured for this task.")
        rows: list[tuple[str, str]] = []
        for record in records:
            target = supervised_target_fn(record)
            if target is None:
                continue
            rows.append((prompt_formatter(record, tokenizer), target))
        if not rows:
            raise ValueError("No supervised targets found for this BYOD task.")
        return rows

    return BYODTask(
        environment=environment,
        verifier=verifier,
        _supervised_samples_fn=build_samples,
    )

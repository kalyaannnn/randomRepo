"""Bring-your-own-dataset task example for AgentRL.

This module is intentionally small. It demonstrates the minimum task-side
pattern for users who want to plug their own dataset into AgentRL without
adding a framework-level dataset system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentrl import BaseEnvironment, BaseVerifier
from agentrl.byod import BYODRecord, BYODTask, make_single_turn_task


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One user-provided task example.

    Attributes:
        prompt: User-facing prompt or observation.
        expected_answer: Deterministic verifier target for the final response.
        target: Optional supervised target used for SFT bootstrap.
        metadata: Arbitrary structured metadata kept for verifier state/debugging.
    """

    prompt: str
    expected_answer: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = (
    "You are solving a user-defined task.\n"
    "Respond in the format expected by the verifier."
)


def _render_prompt(prompt: str, tokenizer: Any | None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:\n"


def _normalized_text(text: str, ignore_case: bool = False) -> str:
    normalized = str(text).strip()
    return normalized.lower() if ignore_case else normalized


def _exact_match_reward(response: str, expected_answer: str, ignore_case: bool = False) -> float:
    return 1.0 if _normalized_text(response, ignore_case) == _normalized_text(expected_answer, ignore_case) else 0.0


def _reward_fn(response: str, state: dict[str, Any]) -> float:
    return _exact_match_reward(response, state["reference_answer"])


def build_demo_task(records: list[TaskRecord], seed: int = 0) -> BYODTask:
    adapted_records = [
        BYODRecord(
            input=record.prompt,
            reference_answer=record.expected_answer,
            supervised_target=record.target,
            metadata=dict(record.metadata),
        )
        for record in records
    ]
    return make_single_turn_task(
        records=adapted_records,
        prompt_formatter=lambda record, tokenizer: _render_prompt(record.input, tokenizer),
        reward_fn=_reward_fn,
        supervised_target_fn=lambda record: record.supervised_target,
        seed=seed,
    )


class BYODEnvironment(BaseEnvironment):
    """Minimal single-turn environment backed by user-provided records.

    Records can come from in-memory ``TaskRecord`` objects or from a local JSONL
    file. The environment owns prompt rendering and bootstrap sample creation,
    while the framework continues to own runtime, training, and telemetry.
    """

    def __init__(
        self,
        records: list[TaskRecord] | None = None,
        jsonl_path: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if records is None and jsonl_path is None:
            raise ValueError("Provide either records or jsonl_path.")
        if records is not None and jsonl_path is not None:
            raise ValueError("Provide records or jsonl_path, not both.")

        self._records = records if records is not None else self._load_jsonl(jsonl_path)
        if not self._records:
            raise ValueError("Task dataset must contain at least one record.")
        self._task = build_demo_task(records=self._records, seed=seed)

    def reset(self) -> str:
        """Return the next task prompt."""

        return self._task.environment.reset()

    def step(self, action: str) -> tuple[str, bool]:
        """Mark the example as complete.

        This example is single-turn by default. Multi-turn tasks can follow the
        same pattern with task-specific state transitions in ``step``.
        """

        return self._task.environment.step(action)

    def state(self) -> dict[str, Any]:
        """Expose verifier-facing state for the current record."""

        return self._task.environment.state()

    def supervised_samples(self, tokenizer: Any | None = None) -> list[tuple[str, str]]:
        """Return prompt/target pairs for SFT bootstrap.

        Only records with ``target`` populated are returned. This keeps dataset
        choice task-owned while matching ``SFTBootstrapTrainer``'s public input
        shape.
        """

        return self._task.supervised_samples(tokenizer=tokenizer)

    def render_prompt(self, tokenizer: Any | None, prompt: str) -> str:
        """Render one record through an optional chat template."""

        return _render_prompt(prompt, tokenizer)

    @staticmethod
    def _load_jsonl(path: str | Path | None) -> list[TaskRecord]:
        if path is None:
            raise ValueError("jsonl_path must be provided when records is None.")
        rows: list[TaskRecord] = []
        path_obj = Path(path).expanduser()
        with path_obj.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                try:
                    prompt = str(payload["prompt"])
                    expected_answer = str(payload["expected_answer"])
                except KeyError as exc:
                    raise ValueError(
                        f"Missing required key {exc.args[0]!r} in {path_obj} line {line_number}."
                    ) from exc
                target = payload.get("target")
                metadata = payload.get("metadata", {})
                rows.append(
                    TaskRecord(
                        prompt=prompt,
                        expected_answer=expected_answer,
                        target=str(target) if target is not None else None,
                        metadata=dict(metadata),
                    )
                )
        return rows


class ExactMatchVerifier(BaseVerifier):
    """Simple verifier for BYOD examples.

    The reward is ``1.0`` when the stripped final response matches the
    expected answer from the environment state exactly, and ``0.0``
    otherwise.
    """

    def __init__(self, ignore_case: bool = False) -> None:
        self.ignore_case = ignore_case

    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        if "expected_answer" in env_state:
            expected = env_state["expected_answer"]
        elif "reference_answer" in env_state:
            expected = env_state["reference_answer"]
        else:
            raise KeyError("expected_answer")
        return _exact_match_reward(response, expected, ignore_case=self.ignore_case)

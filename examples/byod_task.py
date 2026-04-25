"""Bring-your-own-dataset task example for AgentRL.

This module is intentionally small. It demonstrates the minimum task-side
pattern for users who want to plug their own dataset into AgentRL without
adding a framework-level dataset system.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import textwrap
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


def build_mbpp_comparison_records(limit: int, seed: int = 0) -> list[BYODRecord]:
    """Build the frozen MBPP-style BYOD records used for AgentRL-vs-TRL demos."""

    rows = _load_mbpp_rows(limit=limit, seed=seed)
    records: list[BYODRecord] = []
    for row in rows:
        task_id = int(row["task_id"])
        prompt = _render_mbpp_prompt(
            problem=str(row.get("prompt", row.get("text", ""))),
            tests=[str(test) for test in row.get("test_list", [])],
        )
        records.append(
            BYODRecord(
                input=prompt,
                reference_answer=f"task::{task_id}",
                supervised_target=str(row.get("code", "")),
                metadata={
                    "task_id": task_id,
                    "test_setup_code": str(row.get("test_setup_code", "")),
                    "test_list": [str(test) for test in row.get("test_list", [])],
                },
            )
        )
    return records


def build_mbpp_comparison_task(limit: int, seed: int = 0) -> BYODTask:
    """Build the single-turn BYOD task shared by the AgentRL and TRL demos."""

    records = build_mbpp_comparison_records(limit=limit, seed=seed)
    return make_single_turn_task(
        records=records,
        prompt_formatter=_mbpp_prompt_formatter,
        reward_fn=_mbpp_reward_fn,
        supervised_target_fn=lambda record: record.supervised_target,
        seed=seed,
    )


def build_mbpp_comparison_dataset(limit: int, seed: int = 0) -> dict[str, Any]:
    """Build lightweight TRL-facing SFT/RL datasets plus the shared reward function."""

    records = build_mbpp_comparison_records(limit=limit, seed=seed)
    sft_rows = [
        {"prompt": _mbpp_prompt_formatter(record, None), "completion": record.supervised_target or ""}
        for record in records
        if record.supervised_target
    ]
    rl_rows = [
        {
            "prompt": _mbpp_prompt_formatter(record, None),
            "task_id": record.metadata["task_id"],
            "test_setup_code": record.metadata.get("test_setup_code", ""),
            "test_list": list(record.metadata.get("test_list", [])),
        }
        for record in records
    ]

    try:
        from datasets import Dataset
    except ImportError:
        sft_dataset = sft_rows
        rl_dataset = rl_rows
    else:
        sft_dataset = Dataset.from_list(sft_rows)
        rl_dataset = Dataset.from_list(rl_rows)

    def reward_fn(completions: list[str] | str, **kwargs: Any) -> list[float] | float:
        single = isinstance(completions, str)
        completion_rows = [completions] if single else list(completions)
        setup_rows = _broadcast_reward_kwarg(kwargs.get("test_setup_code", ""), len(completion_rows))
        test_rows = _broadcast_reward_kwarg(kwargs.get("test_list", []), len(completion_rows))
        rewards = [
            _strict_code_reward(str(completion), str(setup_code), list(tests))
            for completion, setup_code, tests in zip(completion_rows, setup_rows, test_rows, strict=True)
        ]
        return rewards[0] if single else rewards

    return {"sft": sft_dataset, "rl": rl_dataset, "reward_fn": reward_fn}


def _load_mbpp_rows(limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be > 0.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("MBPP comparison records require `datasets`. Install with `pip install datasets`.") from exc

    dataset = load_dataset("mbpp", "sanitized", split="test")
    rows = [dict(row) for row in dataset]
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def _render_mbpp_prompt(problem: str, tests: list[str]) -> str:
    tests_preview = "\n".join(tests[:3])
    return textwrap.dedent(
        f"""
        Write Python code that solves the problem below.

        Problem:
        {problem}

        Requirements:
        - Return only Python code.
        - Define any functions needed by the tests.
        - Do not include Markdown fences.

        Public tests:
        {tests_preview}
        """
    ).strip()


def _mbpp_prompt_formatter(record: BYODRecord, tokenizer: Any | None) -> str:
    messages = [
        {
            "role": "system",
            "content": "You write correct Python code that passes tests. Return code only.",
        },
        {"role": "user", "content": record.input},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System:\n{messages[0]['content']}\n\nUser:\n{record.input}\n\nAssistant:\n"


def _mbpp_reward_fn(response: str, state: dict[str, Any]) -> float:
    metadata = dict(state.get("metadata", {}))
    return _strict_code_reward(
        response,
        str(metadata.get("test_setup_code", "")),
        list(metadata.get("test_list", [])),
    )


def _broadcast_reward_kwarg(value: Any, count: int) -> list[Any]:
    if isinstance(value, list) and len(value) == count:
        return value
    return [value for _ in range(count)]


def _strict_code_reward(response: str, test_setup_code: str, test_list: list[Any]) -> float:
    if not test_list:
        return 0.0
    code = _strip_markdown_fences(response)
    with tempfile.TemporaryDirectory() as temp_dir:
        candidate = Path(temp_dir) / "candidate.py"
        test_code = "\n".join(str(test) for test in test_list)
        candidate.write_text(
            "\n\n".join(part for part in [test_setup_code, code, test_code] if part.strip()),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if completed.returncode == 0 else 0.0


def _strip_markdown_fences(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


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

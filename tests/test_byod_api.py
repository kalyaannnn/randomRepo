from __future__ import annotations

import pytest

import agentrl
from agentrl.byod import BYODRecord, BYODTask, make_single_turn_task


def test_make_single_turn_task_builds_environment_verifier_and_samples() -> None:
    calls: list[object | None] = []
    records = [
        BYODRecord(
            input="Solve 2 + 2.",
            reference_answer="Final answer: 4",
            supervised_target="2 + 2 = 4\n\nFinal answer: 4",
            metadata={"difficulty": "easy"},
        )
    ]

    def prompt_formatter(record: BYODRecord, tokenizer: object | None) -> str:
        calls.append(tokenizer)
        return f"Prompt: {record.input}"

    task = make_single_turn_task(
        records=records,
        prompt_formatter=prompt_formatter,
        reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
        supervised_target_fn=lambda record: record.supervised_target,
    )

    assert isinstance(task, BYODTask)

    prompt = task.environment.reset()
    state = task.environment.state()
    reward = task.verifier.verify("Final answer: 4", state)
    samples = task.supervised_samples(tokenizer="tokenizer")

    assert prompt == "Prompt: Solve 2 + 2."
    assert state == {
        "input": "Solve 2 + 2.",
        "reference_answer": "Final answer: 4",
        "metadata": {"difficulty": "easy"},
    }
    assert reward == 1.0
    assert samples == [("Prompt: Solve 2 + 2.", "2 + 2 = 4\n\nFinal answer: 4")]
    assert calls == [None, "tokenizer"]


def test_make_single_turn_task_requires_supervised_hook_for_samples() -> None:
    task = make_single_turn_task(
        records=[
            BYODRecord(
                input="Solve 1 + 1.",
                reference_answer="Final answer: 2",
            )
        ],
        prompt_formatter=lambda record, tokenizer: record.input,
        reward_fn=lambda response, state: 0.0,
    )

    with pytest.raises(ValueError, match="No supervised target hook configured for this task."):
        task.supervised_samples()


def test_package_root_exports_byod_api() -> None:
    assert agentrl.BYODRecord is BYODRecord
    assert agentrl.BYODTask is BYODTask
    assert agentrl.make_single_turn_task is make_single_turn_task


def test_make_single_turn_task_requires_state_builder_for_custom_records() -> None:
    class CustomRecord:
        def __init__(self, text: str) -> None:
            self.text = text

    with pytest.raises(ValueError, match="state_builder is required for custom record types."):
        make_single_turn_task(
            records=[CustomRecord("Solve 3 + 3.")],
            prompt_formatter=lambda record, tokenizer: record.text,
            reward_fn=lambda response, state: 0.0,
            supervised_target_fn=lambda record: "unused",
        )


def test_make_single_turn_task_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        make_single_turn_task(
            records=[],
            prompt_formatter=lambda record, tokenizer: record.input,
            reward_fn=lambda response, state: 0.0,
        )

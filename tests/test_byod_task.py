from __future__ import annotations

import json

import pytest

from examples.byod_task import BYODEnvironment, ExactMatchVerifier, TaskRecord, build_demo_task


def test_byod_environment_and_verifier_roundtrip() -> None:
    env = BYODEnvironment(
        records=[
            TaskRecord(
                prompt="Reply with exactly one line: Final answer: 4",
                expected_answer="Final answer: 4",
                target="Reasoning...\n\nFinal answer: 4",
            )
        ],
        seed=123,
    )
    verifier = ExactMatchVerifier()

    prompt = env.reset()
    _, done = env.step("Final answer: 4")
    state = env.state()
    reward = verifier.verify("Final answer: 4", state)

    assert "Final answer: 4" in prompt
    assert state["input"] == "Reply with exactly one line: Final answer: 4"
    assert state["reference_answer"] == "Final answer: 4"
    assert state["metadata"] == {}
    assert done is True
    assert reward == 1.0


def test_byod_environment_builds_supervised_samples() -> None:
    class TemplateTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            rendered = []
            for message in messages:
                rendered.append(f"{message['role']}::{message['content']}")
            if add_generation_prompt:
                rendered.append("assistant::")
            return "\n".join(rendered)

    env = BYODEnvironment(
        records=[
            TaskRecord(
                prompt="Solve 3 + 4.",
                expected_answer="Final answer: 7",
                target="3 + 4 = 7\n\nFinal answer: 7",
            )
        ]
    )

    samples = env.supervised_samples(tokenizer=TemplateTokenizer())

    assert len(samples) == 1
    prompt, target = samples[0]
    assert "system::You are solving a user-defined task." in prompt
    assert "user::Solve 3 + 4." in prompt
    assert target == "3 + 4 = 7\n\nFinal answer: 7"


def test_byod_environment_loads_jsonl_records(tmp_path) -> None:
    payload = {
        "prompt": "Say exactly hello",
        "expected_answer": "hello",
        "target": "hello",
        "metadata": {"split": "train"},
    }
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    env = BYODEnvironment(jsonl_path=jsonl_path, seed=0)

    prompt = env.reset()
    state = env.state()

    assert prompt == (
        "System:\n"
        "You are solving a user-defined task.\n"
        "Respond in the format expected by the verifier.\n\n"
        "User:\nSay exactly hello\n\nAssistant:\n"
    )
    assert state["input"] == "Say exactly hello"
    assert state["reference_answer"] == "hello"
    assert state["metadata"] == {"split": "train"}


def test_byod_environment_normalizes_non_empty_metadata(tmp_path) -> None:
    payload = {
        "prompt": "Normalize metadata",
        "expected_answer": "ok",
        "target": "ok",
        "metadata": [["split", "train"], ["source", "generated"]],
    }
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    env = BYODEnvironment(jsonl_path=jsonl_path)

    env.reset()
    state = env.state()

    assert state["metadata"] == {"split": "train", "source": "generated"}


def test_byod_environment_skips_records_without_supervised_targets(tmp_path) -> None:
    records = [
        {"prompt": "Unsupervised record", "expected_answer": "a"},
        {"prompt": "Supervised record", "expected_answer": "b", "target": "b"},
    ]
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    env = BYODEnvironment(jsonl_path=jsonl_path)

    assert env.supervised_samples() == [(
        "System:\n"
        "You are solving a user-defined task.\n"
        "Respond in the format expected by the verifier.\n\n"
        "User:\nSupervised record\n\nAssistant:\n",
        "b",
    )]


def test_byod_environment_rejects_malformed_jsonl_record(tmp_path) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        json.dumps({"prompt": "Missing answer"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required key 'expected_answer'"):
        BYODEnvironment(jsonl_path=jsonl_path)


def test_byod_environment_requires_supervised_targets_for_bootstrap() -> None:
    env = BYODEnvironment(records=[TaskRecord(prompt="p", expected_answer="a")])

    with pytest.raises(ValueError, match="No supervised targets found"):
        env.supervised_samples()


def test_exact_match_verifier_can_ignore_case() -> None:
    verifier = ExactMatchVerifier(ignore_case=True)

    reward = verifier.verify("final answer: 4", {"expected_answer": "Final Answer: 4"})

    assert reward == 1.0


def test_build_demo_task_wraps_example_records_with_official_api() -> None:
    records = [
        TaskRecord(
            prompt="Return exactly: ok",
            expected_answer="ok",
            target="ok",
        )
    ]

    task = build_demo_task(records=records)
    prompt = task.environment.reset()
    reward = task.verifier.verify("ok", task.environment.state())
    samples = task.supervised_samples(tokenizer=None)

    assert "Return exactly: ok" in prompt
    assert reward == 1.0
    assert samples == [(prompt, "ok")]

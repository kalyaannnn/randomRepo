from __future__ import annotations

import sys

from examples.math_env import MathEnvironment, MathProblem, MathVerifier
from examples import train_math


def test_math_environment_and_verifier_roundtrip() -> None:
    env = MathEnvironment(
        split="train",
        problems=[MathProblem("2 + 2", 4)],
        seed=123,
    )
    verifier = MathVerifier()

    prompt = env.reset()
    _, done = env.step("Final answer: 4")
    reward = verifier.verify("Reasoning...\nFinal answer: 4", env.state())

    assert "2 + 2" in prompt
    assert done is True
    assert reward == 1.0


def test_train_math_uses_public_api_shape(monkeypatch) -> None:
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self) -> None:
            captured["trained"] = True

    monkeypatch.setattr(train_math, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_math.py",
            "--model",
            "fake/model",
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--group-size",
            "4",
            "--max-new-tokens",
            "32",
            "--output-dir",
            "./artifacts",
            "--split",
            "eval",
        ],
    )

    train_math.main()

    assert captured["config"].model_name == "fake/model"
    assert captured["config"].steps == 3
    assert captured["config"].batch_size == 2
    assert captured["config"].group_size == 4
    assert captured["config"].max_new_tokens == 32
    assert captured["config"].output_dir == "./artifacts"
    assert captured["environment"].split == "eval"
    assert captured["verifier"].__class__.__name__ == "MathVerifier"
    assert captured["trained"] is True


def test_smoke_split_uses_easy_builtin_problems() -> None:
    env = MathEnvironment(split="smoke", seed=0)

    prompt = env.reset()
    answer = int(env.state()["answer"])

    assert "Reply with exactly one line and nothing else" in prompt
    assert answer in {2, 3, 4, 5, 6}


def test_smoke_verifier_requires_exact_one_line_format() -> None:
    verifier = MathVerifier()

    strict_state = {"answer": 6, "split": "smoke"}

    assert verifier.verify("Final answer: 6", strict_state) == 1.0
    assert verifier.verify("6\nThe answer is 6.", strict_state) == 0.0
    assert verifier.verify("6Human: Solve the following equation", strict_state) == 0.0

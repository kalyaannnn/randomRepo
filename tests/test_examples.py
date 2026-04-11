from __future__ import annotations

import sys

from examples.benchmark_gsm8k_subset import main as benchmark_gsm8k_main
from examples import benchmark_gsm8k_subset, train_math
from examples.gsm8k_subset import (
    GSM8KProblem,
    GSM8KSubsetEnvironment,
    GSM8KSubsetVerifier,
)
from examples.math_env import MathEnvironment, MathProblem, MathVerifier


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


def test_easy_split_uses_harder_synthetic_problems_and_strict_format() -> None:
    env = MathEnvironment(split="easy", seed=0)
    verifier = MathVerifier()

    prompt = env.reset()
    strict_state = {"answer": 5, "split": "easy"}

    assert "Reply with exactly one line and nothing else" in prompt
    assert verifier.verify("Final answer: 5", strict_state) == 1.0
    assert verifier.verify("5\nextra text", strict_state) == 0.0


def test_gsm8k_subset_environment_and_verifier_roundtrip() -> None:
    env = GSM8KSubsetEnvironment(
        split="train",
        problems=[
            GSM8KProblem(
                "A jar has 10 candies and 3 more are added. How many candies are in the jar?",
                13,
                "10 + 3 = 13\n#### 13",
            )
        ],
        seed=123,
    )
    verifier = GSM8KSubsetVerifier()

    prompt = env.reset()
    _, done = env.step("Final answer: 13")
    reward = verifier.verify("Final answer: 13", env.state())

    assert "GSM8K math word problem" in prompt
    assert "Final answer: <integer>" in prompt
    assert done is True
    assert reward == 1.0
    assert env.state()["curriculum"] == "easy"


def test_gsm8k_subset_verifier_requires_exact_one_line_format() -> None:
    verifier = GSM8KSubsetVerifier()
    state = {"answer": 18, "split": "train", "dataset": "gsm8k"}

    assert verifier.verify("Final answer: 18", state) == 1.0
    assert verifier.verify("18", state) == 0.0
    assert verifier.verify("Final answer: 18\nextra", state) == 0.0


def test_gsm8k_subset_extracts_answer_and_filters_examples(monkeypatch) -> None:
    rows = [
        {"question": "Short question one?", "answer": "work\n#### 7"},
        {"question": "This question is intentionally much too long " * 20, "answer": "work\n#### 99"},
        {"question": "Short question two?", "answer": "more work\n#### 11"},
    ]

    def fake_load_dataset_split(self):
        assert self.dataset_name == "gsm8k"
        assert self.dataset_config_name == "main"
        assert self.split == "train"
        return rows

    monkeypatch.setattr(GSM8KSubsetEnvironment, "_load_dataset_split", fake_load_dataset_split)

    env = GSM8KSubsetEnvironment(
        split="train",
        subset_size=2,
        max_question_words=6,
    )

    prompt = env.reset()
    state = env.state()

    assert len(env._problems) == 2
    assert "Problem:" in prompt
    assert state["answer"] in {7, 11}


def test_gsm8k_subset_easy_curriculum_prefers_simpler_examples(monkeypatch) -> None:
    rows = [
        {
            "question": (
                "A club raises 18 dollars, then 24 dollars, and later spends 7 dollars on snacks. "
                "After buying 3 pens for 2 dollars each, how much money remains?"
            ),
            "answer": "18 + 24 - 7 - 6 = 29\n#### 29",
        },
        {
            "question": "A cart has 2 apples and gets 3 more. How many apples are there in total?",
            "answer": "2 + 3 = 5\n#### 5",
        },
    ]

    def fake_load_dataset_split(self):
        return rows

    monkeypatch.setattr(GSM8KSubsetEnvironment, "_load_dataset_split", fake_load_dataset_split)

    easy_env = GSM8KSubsetEnvironment(
        split="train",
        subset_size=1,
        max_question_words=60,
        curriculum="easy",
    )
    standard_env = GSM8KSubsetEnvironment(
        split="train",
        subset_size=1,
        max_question_words=60,
        curriculum="standard",
    )

    assert easy_env._problems[0].answer == 5
    assert standard_env._problems[0].answer == 29


def test_benchmark_gsm8k_subset_uses_public_api_shape(monkeypatch) -> None:
    captured = {}

    class StubEnvironment:
        def __init__(self, split, subset_size, max_question_words, curriculum):
            self.split = split
            self.subset_size = subset_size
            self.max_question_words = max_question_words
            self.curriculum = curriculum

    class StubVerifier:
        pass

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self) -> None:
            captured["trained"] = True

    monkeypatch.setattr(benchmark_gsm8k_subset, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(benchmark_gsm8k_subset, "GSM8KSubsetEnvironment", StubEnvironment)
    monkeypatch.setattr(benchmark_gsm8k_subset, "GSM8KSubsetVerifier", StubVerifier)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_gsm8k_subset.py",
            "--model",
            "fake/model",
            "--steps",
            "7",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "48",
            "--output-dir",
            "./bench",
            "--split",
            "test",
            "--subset-size",
            "16",
            "--max-question-words",
            "50",
            "--curriculum",
            "standard",
            "--replay-every",
            "1",
        ],
    )

    benchmark_gsm8k_main()

    assert captured["config"].model_name == "fake/model"
    assert captured["config"].steps == 7
    assert captured["config"].batch_size == 1
    assert captured["config"].group_size == 4
    assert captured["config"].max_new_tokens == 48
    assert captured["config"].output_dir == "./bench"
    assert captured["config"].replay_every == 1
    assert captured["environment"].split == "test"
    assert captured["environment"].subset_size == 16
    assert captured["environment"].max_question_words == 50
    assert captured["environment"].curriculum == "standard"
    assert captured["verifier"].__class__.__name__ == "StubVerifier"
    assert captured["trained"] is True

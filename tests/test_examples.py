from __future__ import annotations

import sys
import re
from pathlib import Path
from types import SimpleNamespace

from examples.bootstrap_gsm8k_subset import main as bootstrap_gsm8k_main
from examples.benchmark_gsm8k_subset import main as benchmark_gsm8k_main
from examples.benchmark_systems import main as benchmark_systems_main
from examples.eval_gsm8k_subset import main as eval_gsm8k_main
from examples import (
    benchmark_gsm8k_subset,
    benchmark_systems,
    bootstrap_gsm8k_subset,
    eval_gsm8k_subset,
    train_math,
)
from examples.gsm8k_subset import (
    GSM8KProblem,
    GSM8KSubsetEnvironment,
    GSM8KSubsetVerifier,
)
from examples.math_env import MathEnvironment, MathProblem, MathVerifier
from examples.tool_use_env import ToolUseEnvironment, ToolUseVerifier


def test_readme_mentions_official_byod_api() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)

    assert "BYODRecord" in readme
    assert "make_single_turn_task" in readme
    assert "TRL-compatible clipped GRPO path" in readme
    assert "codeDemo.ipynb" in readme

    for link in links:
        if link.startswith("http"):
            continue
        assert Path(link).is_file(), f"README link does not resolve to a file: {link}"


def test_agentrl_single_turn_baseline_main_runs(monkeypatch, tmp_path, capsys) -> None:
    from examples import agentrl_single_turn_baseline

    captured = {}

    class StubTask:
        environment = object()
        verifier = object()

        def supervised_samples(self, tokenizer=None):
            del tokenizer
            return [("prompt", "target")]

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self):
            return [{"mean_reward": 0.5, "total_step_time_ms": 10.0, "peak_vram_mb": 64.0}]

    monkeypatch.setattr(agentrl_single_turn_baseline, "build_mbpp_comparison_task", lambda **kwargs: StubTask())
    monkeypatch.setattr(agentrl_single_turn_baseline, "run_bootstrap", lambda **kwargs: str(tmp_path / "adapter"))
    monkeypatch.setattr(agentrl_single_turn_baseline, "GRPOTrainer", StubTrainer)

    metrics = agentrl_single_turn_baseline.main(
        [
            "--model",
            "fake/model",
            "--limit",
            "4",
            "--sft-epochs",
            "1",
            "--steps",
            "3",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "64",
            "--output-dir",
            str(tmp_path),
        ],
        return_metrics=True,
    )

    out = capsys.readouterr().out
    assert '"framework": "agentrl"' in out
    assert metrics["framework"] == "agentrl"
    assert metrics["sft_epochs"] == 1
    assert captured["config"].init_adapter_path == str(tmp_path / "adapter")


def test_trl_single_turn_baseline_main_runs(monkeypatch, tmp_path, capsys) -> None:
    from examples import trl_single_turn_baseline

    class StubTrainer:
        def __init__(self, *args, **kwargs):
            self.model = "trained-model"

        def train(self):
            return SimpleNamespace(metrics={"train_runtime": 1.25})

    monkeypatch.setattr(
        trl_single_turn_baseline,
        "build_mbpp_comparison_dataset",
        lambda **kwargs: {
            "sft": [{"prompt": "p", "completion": "c"}],
            "rl": [{"prompt": "p"}],
            "reward_fn": lambda *args, **kwargs: [1.0],
        },
    )
    monkeypatch.setattr(trl_single_turn_baseline, "SFTTrainer", StubTrainer)
    monkeypatch.setattr(trl_single_turn_baseline, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(trl_single_turn_baseline, "SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(trl_single_turn_baseline, "GRPOConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    metrics = trl_single_turn_baseline.main(
        [
            "--model",
            "fake/model",
            "--limit",
            "4",
            "--sft-epochs",
            "1",
            "--steps",
            "3",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "64",
            "--output-dir",
            str(tmp_path),
        ],
        return_metrics=True,
    )

    out = capsys.readouterr().out
    assert '"framework": "trl"' in out
    assert metrics["framework"] == "trl"
    assert metrics["quality_metric"] == "mean_reward"


def test_compare_single_turn_baselines_writes_summary(monkeypatch, tmp_path) -> None:
    from examples import compare_single_turn_baselines

    monkeypatch.setattr(
        compare_single_turn_baselines,
        "run_agentrl",
        lambda args: {"framework": "agentrl", "quality_metric": "mean_reward", "mean_reward": 0.6},
    )
    monkeypatch.setattr(
        compare_single_turn_baselines,
        "run_trl",
        lambda args: {"framework": "trl", "quality_metric": "mean_reward", "mean_reward": 0.55},
    )

    comparison = compare_single_turn_baselines.main(
        ["--model", "fake/model", "--output-dir", str(tmp_path)],
        return_metrics=True,
    )

    summary = (tmp_path / "comparison.json").read_text(encoding="utf-8")
    assert '"framework": "agentrl"' in summary
    assert '"framework": "trl"' in summary
    assert comparison["agentrl"]["mean_reward"] == 0.6


def test_build_colab_single_turn_demo_writes_notebook(tmp_path) -> None:
    from examples.build_colab_single_turn_demo import main

    notebook_path = tmp_path / "agentrl_trl_15b_t4_demo.ipynb"

    main(["--output", str(notebook_path)])

    text = notebook_path.read_text(encoding="utf-8")
    assert "Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "compare_single_turn_baselines" in text
    assert "SFT bootstrap" in text


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
    assert captured["config"].use_continuous_batching is True
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
    verifier = GSM8KSubsetVerifier(reward_mode="strict")

    prompt = env.reset()
    _, done = env.step("Final answer: 13")
    reward = verifier.verify("Final answer: 13", env.state())

    assert "candies" in prompt
    assert done is True
    assert reward == 1.0
    assert env.state()["curriculum"] == "easy"


def test_gsm8k_subset_verifier_requires_exact_one_line_format() -> None:
    verifier = GSM8KSubsetVerifier(reward_mode="strict")
    state = {"answer": 18, "split": "train", "dataset": "gsm8k"}

    assert verifier.verify("Final answer: 18", state) == 1.0
    assert verifier.verify("18", state) == 0.0
    assert verifier.verify("Final answer: 18\nextra", state) == 0.0
    assert verifier.verify("Final answer: 17", state) == 0.0
    assert verifier.verify("The answer is 18.", state) == 0.0


def test_gsm8k_subset_binary_alias_matches_strict_behavior() -> None:
    verifier = GSM8KSubsetVerifier(reward_mode="binary")
    state = {"answer": 18, "split": "train", "dataset": "gsm8k"}

    assert verifier.verify("Final answer: 18", state) == 1.0
    assert verifier.verify("Final answer: 18\nextra", state) == 0.0
    assert verifier.verify("The answer is 18.", state) == 0.0
    assert verifier.verify("Final answer: 17", state) == 0.0
    assert verifier.verify("I have no idea.", state) == 0.0


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
    assert "Short question" in prompt
    assert state["answer"] in {7, 11}


def test_gsm8k_subset_builds_rationale_supervised_samples() -> None:
    class TemplateTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            rendered = []
            for message in messages:
                rendered.append(f"{message['role']}::{message['content']}")
            if add_generation_prompt:
                rendered.append("assistant::")
            return "\n".join(rendered)

    env = GSM8KSubsetEnvironment(
        split="train",
        problems=[
            GSM8KProblem(
                "A jar has 10 candies and 3 more are added. How many candies are in the jar?",
                13,
                "10 + 3 = 13\n#### 13",
            )
        ],
    )

    samples = env.supervised_samples(tokenizer=TemplateTokenizer())

    assert len(samples) == 1
    prompt, target = samples[0]
    assert "system::You are solving grade-school math word problems." in prompt
    assert "user::A jar has 10 candies and 3 more are added. How many candies are in the jar?" in prompt
    assert target == "10 + 3 = 13\n\nFinal answer: 13"


def test_gsm8k_subset_postprocesses_junk_continuations() -> None:
    env = GSM8KSubsetEnvironment(
        split="train",
        problems=[GSM8KProblem("What is 2 + 2?", 4, "2 + 2 = 4\n#### 4")],
    )

    cleaned = env.postprocess_response("Final answer: 4\nHuman: keep going")

    assert cleaned == "Final answer: 4"


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
            self.STOP_STRINGS = ("\nHuman:",)

    class StubVerifier:
        def __init__(self, reward_mode):
            self.reward_mode = reward_mode

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
            "--reward-mode",
            "strict",
            "--init-adapter-path",
            "./adapter",
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
    assert captured["config"].init_adapter_path == "./adapter"
    assert captured["config"].temperature == 0.8
    assert captured["config"].top_p == 0.95
    assert captured["config"].use_continuous_batching is True
    assert captured["config"].stop_strings == ("\nHuman:",)
    assert captured["environment"].split == "test"
    assert captured["environment"].subset_size == 16
    assert captured["environment"].max_question_words == 50
    assert captured["environment"].curriculum == "standard"
    assert captured["verifier"].reward_mode == "strict"
    assert captured["trained"] is True


def test_bootstrap_gsm8k_subset_uses_public_api_shape(monkeypatch, tmp_path) -> None:
    captured = {}

    class StubEnvironment:
        def __init__(self, split, subset_size, max_question_words, curriculum):
            self.split = split
            self.subset_size = subset_size
            self.max_question_words = max_question_words
            self.curriculum = curriculum

        def supervised_samples(self, tokenizer=None):
            captured["bootstrap_tokenizer"] = tokenizer
            return [("Problem: 2 + 2", "Final answer: 4")]

    class StubLayout:
        def __init__(self, **kwargs):
            captured["layout_kwargs"] = kwargs

    class StubBootstrapTrainer:
        def __init__(self, config, tokenizer, layout):
            captured["config"] = config
            captured["tokenizer"] = tokenizer
            captured["layout"] = layout

        def train(self, samples, epochs):
            captured["samples"] = samples
            captured["epochs"] = epochs
            return [{"loss": 0.25}]

        def save_adapter(self, output_dir):
            captured["adapter_dir"] = output_dir
            return Path(output_dir)

    monkeypatch.setattr(bootstrap_gsm8k_subset, "GSM8KSubsetEnvironment", StubEnvironment)
    monkeypatch.setattr(bootstrap_gsm8k_subset, "SharedWeightLayout", StubLayout)
    monkeypatch.setattr(bootstrap_gsm8k_subset, "SFTBootstrapTrainer", StubBootstrapTrainer)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    pad_token_id=0,
                    eos_token="<eos>",
                )
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap_gsm8k_subset.py",
            "--model",
            "fake/model",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--max-seq-length",
            "128",
            "--lr",
            "1e-4",
            "--adapter-dir",
            str(tmp_path / "adapter"),
            "--split",
            "test",
            "--subset-size",
            "16",
            "--max-question-words",
            "50",
            "--curriculum",
            "standard",
        ],
    )

    bootstrap_gsm8k_main()

    assert captured["config"].model_name == "fake/model"
    assert captured["config"].batch_size == 3
    assert captured["config"].max_prompt_tokens == 128
    assert captured["config"].lr == 1e-4
    assert captured["config"].output_dir == str(tmp_path / "adapter")
    assert captured["layout_kwargs"]["device"] == "cpu"
    assert captured["samples"] == [("Problem: 2 + 2", "Final answer: 4")]
    assert captured["epochs"] == 2
    assert captured["adapter_dir"] == str(tmp_path / "adapter")
    assert captured["bootstrap_tokenizer"] is captured["tokenizer"]


def test_benchmark_systems_writes_summary(monkeypatch, tmp_path) -> None:
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self):
            return [
                {
                    "total_step_time_ms": 100.0,
                    "generation_time_ms": 60.0,
                    "training_time_ms": 40.0,
                    "tokens_per_second": 20.0,
                    "prefill_tokens_per_second": 30.0,
                    "decode_tokens_per_second": 15.0,
                    "padding_ratio": 0.1,
                    "generation_padding_ratio": 0.2,
                    "sequence_padding_ratio": 0.05,
                    "cache_reuse_effectiveness": 0.7,
                    "peak_vram_mb": 123.0,
                    "rollout_peak_vram_mb": 111.0,
                    "rollout_runtime_headroom_mb": 456.0,
                    "runtime_adjustments": 1.0,
                    "runtime_low_headroom": 0.0,
                    "dominant_runtime_bottleneck": "padding",
                    "runtime_recommendation": "Padding waste is high; reduce chunk_size or group together similar prompt lengths.",
                    "last_runtime_adjustment_reason": "high_padding_chunk_size",
                }
            ]

    monkeypatch.setattr(benchmark_systems, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "16",
            "--split",
            "easy",
            "--output-dir",
            str(tmp_path / "systems"),
        ],
    )

    benchmark_systems_main()

    summary = (tmp_path / "systems" / "summary.json").read_text(encoding="utf-8")
    assert '"mean_step_time_ms": 100.0' in summary
    assert '"mode_name": "continuous batching"' in summary
    assert '"task_name": "math"' in summary
    assert '"mean_generation_fraction": 0.6' in summary
    assert '"mean_cache_reuse_effectiveness": 0.7' in summary
    assert '"dominant_runtime_bottleneck": "padding"' in summary
    assert '"efficiency_diagnosis": "padding-limited"' in summary
    assert '"steps_with_runtime_adjustment": 1' in summary
    assert '"benchmark_verdict": "continuous batching was padding-limited, dominated by padding, and needed 1 runtime adjustment.' in summary


def test_benchmark_systems_supports_tool_use_task(monkeypatch, tmp_path) -> None:
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self):
            return [
                {
                    "mean_reward": 0.5,
                    "reward_std": 0.25,
                    "total_step_time_ms": 120.0,
                    "generation_time_ms": 72.0,
                    "training_time_ms": 48.0,
                    "tokens_per_second": 18.0,
                    "prefill_tokens_per_second": 28.0,
                    "decode_tokens_per_second": 14.0,
                    "padding_ratio": 0.12,
                    "generation_padding_ratio": 0.18,
                    "sequence_padding_ratio": 0.09,
                    "cache_reuse_effectiveness": 0.75,
                    "peak_vram_mb": 130.0,
                    "rollout_peak_vram_mb": 118.0,
                    "rollout_runtime_headroom_mb": 430.0,
                    "runtime_adjustments": 0.0,
                    "runtime_low_headroom": 0.0,
                    "dominant_runtime_bottleneck": "decode",
                    "runtime_recommendation": "Decode dominates; continuous modes should help on longer-lived episodes.",
                    "last_runtime_adjustment_reason": "none",
                }
            ]

    monkeypatch.setattr(benchmark_systems, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--group-size",
            "2",
            "--max-new-tokens",
            "24",
            "--max-episode-steps",
            "4",
            "--task",
            "tool-use",
            "--split",
            "easy",
            "--output-dir",
            str(tmp_path / "systems_tool_use"),
        ],
    )

    benchmark_systems_main()

    assert isinstance(captured["environment"], ToolUseEnvironment)
    assert isinstance(captured["verifier"], ToolUseVerifier)
    assert captured["config"].max_episode_steps == 4

    summary = (tmp_path / "systems_tool_use" / "summary.json").read_text(encoding="utf-8")
    assert '"task_name": "tool-use"' in summary
    assert '"mean_reward": 0.5' in summary
    assert '"mode_name": "continuous batching"' in summary


def test_benchmark_systems_reports_paged_kv_diagnosis(monkeypatch, tmp_path) -> None:
    class StubTrainer:
        def __init__(self, config, environment, verifier):
            del environment, verifier
            self.config = config

        def train(self):
            assert self.config.use_paged_kv_continuous is True
            return [
                {
                    "total_step_time_ms": 90.0,
                    "generation_time_ms": 40.0,
                    "training_time_ms": 50.0,
                    "tokens_per_second": 24.0,
                    "prefill_tokens_per_second": 18.0,
                    "decode_tokens_per_second": 16.0,
                    "padding_ratio": 0.1,
                    "generation_padding_ratio": 0.1,
                    "sequence_padding_ratio": 0.05,
                    "cache_reuse_effectiveness": 0.85,
                    "scheduler_prefill_kv_pressure": 0.2,
                    "scheduler_decode_kv_pressure": 0.3,
                    "paged_kv_allocator_pressure": 0.95,
                    "peak_vram_mb": 125.0,
                    "rollout_peak_vram_mb": 115.0,
                    "rollout_runtime_headroom_mb": 450.0,
                    "runtime_adjustments": 0.0,
                    "runtime_low_headroom": 0.0,
                    "dominant_runtime_bottleneck": "paged_kv",
                    "runtime_recommendation": "Paged-KV allocator pressure is high and live sequences are being deferred; reduce chunk_size, max_new_tokens, or prompt length before scaling concurrency.",
                    "last_runtime_adjustment_reason": "none",
                }
            ]

    monkeypatch.setattr(benchmark_systems, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "16",
            "--split",
            "easy",
            "--output-dir",
            str(tmp_path / "systems_paged"),
        ],
    )

    original_run_one = benchmark_systems._run_one

    def stubbed_run_one(args, use_continuous_batching, output_dir, *, use_paged_kv_continuous=False):
        return original_run_one(
            args,
            use_continuous_batching=use_continuous_batching,
            output_dir=output_dir,
            use_paged_kv_continuous=True,
        )

    monkeypatch.setattr(benchmark_systems, "_run_one", stubbed_run_one)

    benchmark_systems_main()

    summary = (tmp_path / "systems_paged" / "summary.json").read_text(encoding="utf-8")
    assert '"mode_name": "paged-kv continuous batching"' in summary
    assert '"mean_scheduler_kv_pressure": 0.95' in summary
    assert '"dominant_runtime_bottleneck": "paged_kv"' in summary
    assert '"efficiency_diagnosis": "paged-kv-limited"' in summary


def test_benchmark_systems_writes_comparison_verdict(monkeypatch, tmp_path) -> None:
    call_count = {"value": 0}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            del environment, verifier
            self.config = config

        def train(self):
            call_count["value"] += 1
            if self.config.use_paged_kv_continuous:
                return [
                    {
                        "total_step_time_ms": 70.0,
                        "generation_time_ms": 42.0,
                        "training_time_ms": 28.0,
                        "tokens_per_second": 35.0,
                        "prefill_tokens_per_second": 27.0,
                        "decode_tokens_per_second": 22.0,
                        "padding_ratio": 0.15,
                        "generation_padding_ratio": 0.15,
                        "sequence_padding_ratio": 0.08,
                        "cache_reuse_effectiveness": 0.9,
                        "paged_kv_allocator_pressure": 0.9,
                        "peak_vram_mb": 119.0,
                        "rollout_peak_vram_mb": 109.0,
                        "rollout_runtime_headroom_mb": 510.0,
                        "runtime_adjustments": 0.0,
                        "runtime_low_headroom": 0.0,
                        "dominant_runtime_bottleneck": "paged_kv",
                        "runtime_recommendation": "Paged-KV allocator pressure is high and live sequences are being deferred; reduce chunk_size, max_new_tokens, or prompt length before scaling concurrency.",
                        "last_runtime_adjustment_reason": "none",
                    }
                ]
            if self.config.use_continuous_batching:
                return [
                    {
                        "total_step_time_ms": 80.0,
                        "generation_time_ms": 50.0,
                        "training_time_ms": 30.0,
                        "tokens_per_second": 30.0,
                        "prefill_tokens_per_second": 25.0,
                        "decode_tokens_per_second": 18.0,
                        "padding_ratio": 0.2,
                        "generation_padding_ratio": 0.2,
                        "sequence_padding_ratio": 0.1,
                        "cache_reuse_effectiveness": 0.8,
                        "peak_vram_mb": 120.0,
                        "rollout_peak_vram_mb": 110.0,
                        "rollout_runtime_headroom_mb": 500.0,
                        "runtime_adjustments": 1.0,
                        "runtime_low_headroom": 0.0,
                        "dominant_runtime_bottleneck": "padding",
                        "runtime_recommendation": "Padding waste is high; reduce chunk_size or group together similar prompt lengths.",
                        "last_runtime_adjustment_reason": "high_padding_chunk_size",
                    }
                ]
            return [
                {
                    "total_step_time_ms": 100.0,
                    "generation_time_ms": 60.0,
                    "training_time_ms": 40.0,
                    "tokens_per_second": 20.0,
                    "prefill_tokens_per_second": 20.0,
                    "decode_tokens_per_second": 12.0,
                    "padding_ratio": 0.1,
                    "generation_padding_ratio": 0.1,
                    "sequence_padding_ratio": 0.05,
                    "cache_reuse_effectiveness": 0.4,
                    "peak_vram_mb": 118.0,
                    "rollout_peak_vram_mb": 108.0,
                    "rollout_runtime_headroom_mb": 520.0,
                    "runtime_adjustments": 0.0,
                    "runtime_low_headroom": 0.0,
                    "dominant_runtime_bottleneck": "balanced",
                    "runtime_recommendation": "Runtime phases look balanced for the current workload.",
                    "last_runtime_adjustment_reason": "none",
                }
            ]

    monkeypatch.setattr(benchmark_systems, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "16",
            "--split",
            "easy",
            "--output-dir",
            str(tmp_path / "systems_compare"),
            "--compare-runtime-modes",
        ],
    )

    benchmark_systems_main()

    comparison = (tmp_path / "systems_compare" / "comparison.json").read_text(encoding="utf-8")
    assert '"mode_name": "standard rollout"' in comparison
    assert '"mode_name": "continuous batching"' in comparison
    assert '"mode_name": "paged-kv continuous batching"' in comparison
    assert '"comparison_verdict": "Fastest overall: paged-kv continuous batching (70.00 ms mean step time, paged-kv-limited). Slowest overall: standard rollout (100.00 ms). Paged-KV continuous batching was 10.00 ms faster than legacy continuous batching (70.00 ms vs 80.00 ms)."' in comparison


def test_benchmark_systems_can_include_speculative_mode(monkeypatch, tmp_path) -> None:
    captured_configs = []

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            del environment, verifier
            self.config = config
            captured_configs.append(config)

        def train(self):
            return [
                {
                    "total_step_time_ms": 75.0 if self.config.use_speculative_decoding else 100.0,
                    "generation_time_ms": 45.0,
                    "training_time_ms": 30.0,
                    "tokens_per_second": 32.0,
                    "prefill_tokens_per_second": 24.0,
                    "decode_tokens_per_second": 20.0,
                    "padding_ratio": 0.1,
                    "generation_padding_ratio": 0.1,
                    "sequence_padding_ratio": 0.05,
                    "cache_reuse_effectiveness": 0.6,
                    "peak_vram_mb": 118.0,
                    "rollout_peak_vram_mb": 108.0,
                    "rollout_runtime_headroom_mb": 520.0,
                    "runtime_adjustments": 0.0,
                    "runtime_low_headroom": 0.0,
                    "dominant_runtime_bottleneck": "decode",
                    "runtime_recommendation": "Decode dominates.",
                    "last_runtime_adjustment_reason": "none",
                }
            ]

    monkeypatch.setattr(benchmark_systems, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--draft-model",
            "fake/draft",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "16",
            "--output-dir",
            str(tmp_path / "systems_speculative"),
            "--compare-runtime-modes",
            "--include-speculative",
        ],
    )

    benchmark_systems_main()

    comparison = (tmp_path / "systems_speculative" / "comparison.json").read_text(encoding="utf-8")
    assert '"mode_name": "speculative decoding"' in comparison
    assert any(config.use_speculative_decoding for config in captured_configs)
    assert any(config.draft_model_name == "fake/draft" for config in captured_configs)


def test_eval_gsm8k_subset_writes_summary(monkeypatch, tmp_path) -> None:
    class StubEnvironment:
        dataset_name = "gsm8k"
        dataset_config_name = "main"

        def __init__(self, split, subset_size, max_question_words, curriculum):
            self.split = split
            self.subset_size = subset_size
            self.max_question_words = max_question_words
            self.curriculum = curriculum

        def problems(self):
            return [GSM8KProblem("What is 2 + 2?", 4, "2 + 2 = 4\n#### 4")]

        @staticmethod
        def render_prompt(tokenizer, question):
            del tokenizer
            return f"Prompt: {question}"

        @staticmethod
        def postprocess_response(response):
            return response.split("\nHuman:", 1)[0]

    class StubVerifier:
        def __init__(self, reward_mode):
            self.reward_mode = reward_mode

        def verify(self, response, env_state):
            del env_state
            return 1.0 if response.strip() == "Final answer: 4" else 0.0

        @staticmethod
        def extract_terminal_final_answer(response):
            if response.strip() == "Final answer: 4":
                return 4
            return None

    class StubTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        eos_token = "<eos>"

        def __call__(self, prompt, return_tensors, add_special_tokens):
            del prompt, return_tensors, add_special_tokens
            import torch

            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

        def decode(self, response_ids, skip_special_tokens):
            del response_ids, skip_special_tokens
            return "Final answer: 4\nHuman: keep going"

    class StubModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            return None

        def generate(self, **kwargs):
            del kwargs
            import torch

            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    class StubLayout:
        def __init__(self, **kwargs):
            self.device = "cpu"
            self.model = StubModel()

    monkeypatch.setattr(eval_gsm8k_subset, "GSM8KSubsetEnvironment", StubEnvironment)
    monkeypatch.setattr(eval_gsm8k_subset, "GSM8KSubsetVerifier", StubVerifier)
    monkeypatch.setattr(eval_gsm8k_subset, "SharedWeightLayout", StubLayout)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: StubTokenizer()
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_gsm8k_subset.py",
            "--model",
            "fake/model",
            "--init-adapter-path",
            "./adapter",
            "--output-dir",
            str(tmp_path / "eval"),
            "--split",
            "test",
            "--subset-size",
            "16",
            "--max-question-words",
            "50",
            "--curriculum",
            "standard",
            "--max-new-tokens",
            "16",
        ],
    )

    eval_gsm8k_main()

    summary = (tmp_path / "eval" / "summary.json").read_text(encoding="utf-8")
    predictions = (tmp_path / "eval" / "predictions.jsonl").read_text(encoding="utf-8")

    assert '"exact_match_rate": 1.0' in summary
    assert '"pass_at_1": 1.0' in summary
    assert '"subset_size": 1' in summary
    assert '"reward": 1.0' in predictions
    assert '"sampled_responses": ["Final answer: 4"]' in predictions


def test_eval_gsm8k_subset_reports_pass_at_k(monkeypatch, tmp_path) -> None:
    class StubEnvironment:
        dataset_name = "gsm8k"
        dataset_config_name = "main"

        def __init__(self, split, subset_size, max_question_words, curriculum):
            self.split = split
            self.subset_size = subset_size
            self.max_question_words = max_question_words
            self.curriculum = curriculum

        def problems(self):
            return [GSM8KProblem("What is 2 + 2?", 4, "2 + 2 = 4\n#### 4")]

        @staticmethod
        def render_prompt(tokenizer, question):
            del tokenizer
            return f"Prompt: {question}"

        @staticmethod
        def postprocess_response(response):
            return response

    class StubVerifier:
        def __init__(self, reward_mode):
            self.reward_mode = reward_mode

        def verify(self, response, env_state):
            del env_state
            return 1.0 if response.strip() == "Final answer: 4" else 0.0

        @staticmethod
        def extract_terminal_final_answer(response):
            if response.strip() == "Final answer: 4":
                return 4
            return None

    class StubTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        eos_token = "<eos>"

        def __call__(self, prompt, return_tensors, add_special_tokens):
            del prompt, return_tensors, add_special_tokens
            import torch

            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

        def decode(self, response_ids, skip_special_tokens):
            del skip_special_tokens
            mapping = {
                (3,): "wrong",
                (4,): "Final answer: 4",
                (5,): "still wrong",
            }
            return mapping[tuple(int(token) for token in response_ids.tolist())]

    class StubModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            return None

        def generate(self, **kwargs):
            import torch

            assert kwargs["num_return_sequences"] == 3
            assert kwargs["do_sample"] is True
            return torch.tensor(
                [
                    [1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5],
                ],
                dtype=torch.long,
            )

    class StubLayout:
        def __init__(self, **kwargs):
            self.device = "cpu"
            self.model = StubModel()

    monkeypatch.setattr(eval_gsm8k_subset, "GSM8KSubsetEnvironment", StubEnvironment)
    monkeypatch.setattr(eval_gsm8k_subset, "GSM8KSubsetVerifier", StubVerifier)
    monkeypatch.setattr(eval_gsm8k_subset, "SharedWeightLayout", StubLayout)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: StubTokenizer()
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_gsm8k_subset.py",
            "--model",
            "fake/model",
            "--init-adapter-path",
            "./adapter",
            "--output-dir",
            str(tmp_path / "eval"),
            "--split",
            "train",
            "--subset-size",
            "16",
            "--max-question-words",
            "50",
            "--curriculum",
            "easy",
            "--max-new-tokens",
            "96",
            "--num-samples",
            "3",
            "--pass-k",
            "2",
        ],
    )

    eval_gsm8k_main()

    summary = (tmp_path / "eval" / "summary.json").read_text(encoding="utf-8")
    predictions = (tmp_path / "eval" / "predictions.jsonl").read_text(encoding="utf-8")

    assert '"pass_at_1": 0.0' in summary
    assert '"pass_at_k": 1.0' in summary
    assert '"fraction_with_any_correct": 1.0' in summary
    assert '"mean_reward": 0.3333333333333333' in summary
    assert '"parsed_predictions": [null, 4, null]' in predictions
    assert '"rewards": [0.0, 1.0, 0.0]' in predictions

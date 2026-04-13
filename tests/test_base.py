from __future__ import annotations

import pytest

from agentrl import BaseEnvironment, BaseVerifier, ConfigurationError, GRPOConfig


class ToyEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self._done = False

    def reset(self) -> str:
        self._done = False
        return "solve 2 + 2"

    def step(self, action: str) -> tuple[str, bool]:
        self._done = True
        return (f"received:{action}", self._done)

    def state(self) -> dict[str, int]:
        return {"answer": 4}


class ToyVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, int]) -> float:
        return 1.0 if response.strip() == str(env_state["answer"]) else 0.0


def test_base_contracts_are_subclassable_and_behave_as_expected() -> None:
    env = ToyEnvironment()
    verifier = ToyVerifier()

    initial = env.reset()
    next_observation, done = env.step("4")
    reward = verifier.verify("4", env.state())

    assert initial == "solve 2 + 2"
    assert next_observation == "received:4"
    assert done is True
    assert reward == 1.0


def test_config_defaults_match_prompt_surface() -> None:
    config = GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    assert config.group_size == 8
    assert config.batch_size == 4
    assert config.max_new_tokens == 512
    assert config.beta == 0.01
    assert config.lr == 1e-5
    assert config.steps == 500
    assert config.use_lora is True
    assert config.use_gradient_checkpointing is False
    assert config.use_continuous_batching is True
    assert config.use_speculative_decoding is False
    assert config.max_episode_steps == 8
    assert config.output_path.name == "checkpoints"
    assert config.top_p == 1.0
    assert config.stop_strings == ()


def test_config_requires_draft_model_for_speculative_decoding() -> None:
    with pytest.raises(ConfigurationError, match="draft_model_name"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_speculative_decoding=True,
        )


def test_config_rejects_invalid_group_size() -> None:
    with pytest.raises(ConfigurationError, match="group_size"):
        GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", group_size=1)


def test_config_rejects_wandb_without_project() -> None:
    with pytest.raises(ConfigurationError, match="wandb_project"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            log_to_wandb=True,
        )


def test_config_rejects_invalid_max_episode_steps() -> None:
    with pytest.raises(ConfigurationError, match="max_episode_steps"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_episode_steps=0,
        )


def test_config_rejects_use_lora_false_until_supported() -> None:
    with pytest.raises(ConfigurationError, match="use_lora=False"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_lora=False,
        )


def test_config_replay_generation_is_deterministic() -> None:
    config = GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    assert config.rollout_generation_kwargs() == {"temperature": 1.0, "top_p": 1.0, "do_sample": True}
    assert config.replay_generation_kwargs() == {"temperature": 0.0, "top_p": 1.0, "do_sample": False}

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
    assert config.beta == 0.0
    assert config.epsilon == 0.2
    assert config.lr == 1e-5
    assert config.lr_scheduler == "constant"
    assert config.warmup_steps == 0
    assert config.min_lr_ratio == 0.0
    assert config.steps == 500
    assert config.use_lora is True
    assert config.use_gradient_checkpointing is False
    assert config.use_continuous_batching is True
    assert config.use_paged_kv_continuous is False
    assert config.use_speculative_decoding is False
    assert config.max_episode_steps == 8
    assert config.output_path.name == "checkpoints"
    assert config.profile_steps is None
    assert config.profile_path.name == "profiles"
    assert config.use_adaptive_kl is False
    assert config.num_iterations == 1
    assert config.grpo_mode == "trl"
    assert config.use_async_rollout_workers is False
    assert config.use_async_trajectory_copy is False
    assert config.experimental_vllm_rollout is False
    assert config.top_p == 1.0
    assert config.stop_strings == ()


def test_config_requires_draft_model_for_speculative_decoding() -> None:
    with pytest.raises(ConfigurationError, match="draft_model_name"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_speculative_decoding=True,
        )


def test_config_requires_continuous_batching_for_paged_kv_mode() -> None:
    with pytest.raises(ConfigurationError, match="use_paged_kv_continuous"):
        GRPOConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_continuous_batching=False,
            use_paged_kv_continuous=True,
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


def test_config_rejects_invalid_lr_scheduler() -> None:
    with pytest.raises(ConfigurationError, match="lr_scheduler"):
        GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", lr_scheduler="linear")


def test_config_rejects_num_iterations_other_than_one() -> None:
    with pytest.raises(ConfigurationError, match="num_iterations"):
        GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", num_iterations=2)


def test_config_rejects_non_trl_grpo_mode() -> None:
    with pytest.raises(ConfigurationError, match="grpo_mode"):
        GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", grpo_mode="legacy")


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"use_adaptive_kl": True}, "use_adaptive_kl"),
        ({"kl_target": 0.1}, "kl_target"),
    ],
)
def test_config_rejects_adaptive_kl_options_in_trl_mode(
    kwargs: dict[str, bool | float],
    expected_message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=expected_message):
        GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", **kwargs)


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

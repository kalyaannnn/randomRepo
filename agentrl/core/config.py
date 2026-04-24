"""Configuration objects and validation helpers for AgentRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when an AgentRL configuration is internally inconsistent."""


@dataclass(slots=True)
class GRPOConfig:
    """Training and runtime configuration for GRPO post-training.

    The defaults follow the user-facing API contract in the project prompt so a
    minimal training script can omit most optional knobs.
    """

    model_name: str
    group_size: int = 8
    batch_size: int = 4
    max_new_tokens: int = 512
    beta: float = 0.0
    epsilon: float = 0.2
    num_iterations: int = 1
    grpo_mode: str = "trl"
    lr: float = 1e-5
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    steps: int = 500
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_gradient_checkpointing: bool = False
    use_continuous_batching: bool = True
    use_paged_kv_continuous: bool = False
    use_speculative_decoding: bool = False
    draft_model_name: str | None = None
    speculative_k: int = 4
    prefill_chunk_size: int = 512
    chunk_size: int | None = None
    auto_tune_chunk_size: bool = True
    execution_policy: str = "balanced"
    min_runtime_headroom_mb: float = 1024.0
    auto_reduce_on_oom: bool = True
    oom_retry_budget: int = 1
    max_prompt_tokens: int | None = None
    max_episode_steps: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    debug_temperature: float = 0.0
    do_sample: bool = True
    stop_strings: tuple[str, ...] = ()
    clip_range: float = 10.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    dtype: str = "float16"
    device: str | None = None
    trust_remote_code: bool = False
    log_to_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    reward_histogram_every: int = 10
    profile_steps: int | None = None
    profile_dir: str = "./profiles"
    use_adaptive_kl: bool = False
    kl_target: float | None = None
    kl_beta_multiplier: float = 1.5
    min_beta: float = 1e-4
    max_beta: float = 1.0
    replay_every: int = 50
    save_every: int = 100
    output_dir: str = "./checkpoints"
    jsonl_metrics_name: str = "metrics.jsonl"
    checkpoint_prefix: str = "checkpoint"
    init_adapter_path: str | None = None
    torch_compile: bool = False
    sdpa_backend: str = "auto"
    optimizer_name: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    pad_to_multiple_of: int | None = 8
    trajectory_buffer_max_batches: int = 8
    use_async_rollout_workers: bool = False
    async_rollout_num_workers: int = 2
    async_rollout_queue_size: int = 4
    use_async_trajectory_copy: bool = False
    async_trajectory_max_pending_batches: int = 2
    experimental_vllm_rollout: bool = False
    _output_path: Path = field(init=False, repr=False)
    _profile_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config invariants that later components depend on."""

        self._validate_positive_int("group_size", self.group_size, minimum=2)
        self._validate_positive_int("batch_size", self.batch_size)
        self._validate_positive_int("max_new_tokens", self.max_new_tokens)
        self._validate_positive_int("num_iterations", self.num_iterations)
        self._validate_positive_int("steps", self.steps)
        self._validate_positive_int("lora_r", self.lora_r)
        self._validate_positive_int("lora_alpha", self.lora_alpha)
        self._validate_positive_int("speculative_k", self.speculative_k)
        self._validate_positive_int("prefill_chunk_size", self.prefill_chunk_size)
        self._validate_positive_int("max_episode_steps", self.max_episode_steps)
        self._validate_positive_int("gradient_accumulation_steps", self.gradient_accumulation_steps)
        self._validate_positive_int("reward_histogram_every", self.reward_histogram_every)
        self._validate_positive_int("replay_every", self.replay_every)
        self._validate_positive_int("save_every", self.save_every)
        self._validate_positive_int("trajectory_buffer_max_batches", self.trajectory_buffer_max_batches)
        self._validate_positive_int("async_rollout_num_workers", self.async_rollout_num_workers)
        self._validate_positive_int("async_rollout_queue_size", self.async_rollout_queue_size)
        self._validate_positive_int("async_trajectory_max_pending_batches", self.async_trajectory_max_pending_batches)

        self._validate_probability("beta", self.beta, allow_one=True)
        self._validate_probability("epsilon", self.epsilon, allow_one=True)
        self._validate_probability("lora_dropout", self.lora_dropout, allow_one=True)
        self._validate_probability("adam_beta1", self.adam_beta1, allow_one=False)
        self._validate_probability("adam_beta2", self.adam_beta2, allow_one=False)

        if self.lr <= 0:
            raise ConfigurationError("lr must be > 0.")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ConfigurationError("lr_scheduler must be one of: constant, cosine.")
        if self.warmup_steps < 0:
            raise ConfigurationError("warmup_steps must be >= 0.")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ConfigurationError("min_lr_ratio must satisfy 0.0 <= min_lr_ratio <= 1.0.")
        if self.clip_range <= 0:
            raise ConfigurationError("clip_range must be > 0.")
        if self.temperature < 0:
            raise ConfigurationError("temperature must be >= 0.")
        if not (0.0 < self.top_p <= 1.0):
            raise ConfigurationError("top_p must satisfy 0.0 < top_p <= 1.0.")
        if self.debug_temperature != 0.0:
            raise ConfigurationError("debug_temperature must be 0.0 for deterministic replay.")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens <= 0:
            raise ConfigurationError("max_prompt_tokens must be > 0 when provided.")
        if self.execution_policy not in {"safe", "balanced", "max_throughput"}:
            raise ConfigurationError(
                "execution_policy must be one of: safe, balanced, max_throughput."
            )
        if self.min_runtime_headroom_mb < 0:
            raise ConfigurationError("min_runtime_headroom_mb must be >= 0.")
        if self.oom_retry_budget < 0:
            raise ConfigurationError("oom_retry_budget must be >= 0.")
        if self.init_adapter_path is not None and not self.init_adapter_path.strip():
            raise ConfigurationError("init_adapter_path must be a non-empty string when provided.")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be > 0 when provided.")
        if self.max_grad_norm <= 0:
            raise ConfigurationError("max_grad_norm must be > 0.")
        if self.weight_decay < 0:
            raise ConfigurationError("weight_decay must be >= 0.")
        if self.adam_eps <= 0:
            raise ConfigurationError("adam_eps must be > 0.")
        if self.profile_steps is not None and self.profile_steps < 0:
            raise ConfigurationError("profile_steps must be >= 0 when provided.")
        if self.kl_target is not None and self.kl_target <= 0:
            raise ConfigurationError("kl_target must be > 0 when provided.")
        if self.kl_beta_multiplier <= 1.0:
            raise ConfigurationError("kl_beta_multiplier must be > 1.0.")
        if self.min_beta <= 0 or self.max_beta <= 0:
            raise ConfigurationError("min_beta and max_beta must be > 0.")
        if self.min_beta > self.max_beta:
            raise ConfigurationError("min_beta must be <= max_beta.")
        if not self.lora_target_modules:
            raise ConfigurationError("lora_target_modules must contain at least one module name.")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ConfigurationError("dtype must be one of: float16, bfloat16, float32.")
        if self.device not in {None, "auto", "cpu", "cuda", "mps"}:
            raise ConfigurationError("device must be one of: None, auto, cpu, cuda, mps.")
        if self.sdpa_backend not in {"auto", "flash_attention", "math"}:
            raise ConfigurationError("sdpa_backend must be one of: auto, flash_attention, math.")
        if not self.use_lora:
            raise ConfigurationError(
                "use_lora=False is not supported yet. AgentRL currently depends on the "
                "shared-weight LoRA/reference layout."
            )
        if self.num_iterations != 1:
            raise ConfigurationError(
                "num_iterations must be 1 for the TRL-compatible GRPO redesign."
            )
        if self.grpo_mode != "trl":
            raise ConfigurationError(
                "grpo_mode must be 'trl' for the TRL-compatible GRPO redesign."
            )
        if self.use_adaptive_kl:
            raise ConfigurationError(
                "use_adaptive_kl=True is not supported in the TRL-compatible GRPO redesign."
            )
        if self.kl_target is not None:
            raise ConfigurationError(
                "kl_target is not supported in the TRL-compatible GRPO redesign."
            )

        if self.use_speculative_decoding and not self.draft_model_name:
            raise ConfigurationError(
                "use_speculative_decoding=True requires draft_model_name to be set."
            )
        if self.use_paged_kv_continuous and not self.use_continuous_batching:
            raise ConfigurationError(
                "use_paged_kv_continuous=True requires use_continuous_batching=True."
            )
        if not self.use_speculative_decoding and self.draft_model_name is not None:
            raise ConfigurationError(
                "draft_model_name is set but use_speculative_decoding=False. "
                "Enable speculative decoding or clear draft_model_name."
            )
        if self.log_to_wandb and not self.wandb_project:
            raise ConfigurationError("log_to_wandb=True requires wandb_project.")
        self._output_path = Path(self.output_dir).expanduser()
        self._profile_path = Path(self.profile_dir).expanduser()

    @property
    def output_path(self) -> Path:
        """Return the normalized output directory path."""

        return self._output_path

    @property
    def profile_path(self) -> Path:
        """Return the normalized profile output directory path."""

        return self._profile_path

    def rollout_generation_kwargs(self) -> dict[str, float | bool]:
        """Return generation kwargs for exploratory rollout sampling."""

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

    def replay_generation_kwargs(self) -> dict[str, float | bool]:
        """Return generation kwargs for deterministic replay."""

        return {
            "temperature": self.debug_temperature,
            "top_p": 1.0,
            "do_sample": False,
        }

    def _validate_positive_int(self, name: str, value: int, minimum: int = 1) -> None:
        if value < minimum:
            comparator = f">= {minimum}" if minimum != 1 else "> 0"
            raise ConfigurationError(f"{name} must be {comparator}.")

    def _validate_probability(self, name: str, value: float, allow_one: bool) -> None:
        upper_ok = value <= 1.0 if allow_one else value < 1.0
        if value < 0.0 or not upper_ok:
            operator = "<=" if allow_one else "<"
            raise ConfigurationError(f"{name} must satisfy 0.0 <= {name} {operator} 1.0.")

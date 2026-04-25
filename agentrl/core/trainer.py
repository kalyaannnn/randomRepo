"""GRPO trainer for AgentRL."""

from __future__ import annotations

import logging
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutBatch, RolloutOrchestrator
from agentrl.generation.continuous import ContinuousBatchingOrchestrator
from agentrl.generation.scheduler import compute_safe_chunk_size
from agentrl.generation.speculative import SpeculativeRolloutOrchestrator
from agentrl.memory.buffer import TrajectoryBuffer
from agentrl.memory.layout import SharedWeightLayout
from agentrl.observability.debugger import AgentRLDebugger
from agentrl.observability.logger import MetricsLogger
from agentrl.observability.profiler import SystemsProfiler
from agentrl.runtime.controller import ExecutionController


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StepMetrics:
    """Structured scalar metrics for one GRPO update."""

    mean_reward: float
    reward_std: float
    policy_loss: float
    kl_loss: float
    total_loss: float
    advantage_mean: float
    advantage_std: float
    unique_response_ratio: float


@dataclass(slots=True)
class GRPOObjectiveStats:
    """Structured outputs from the sampled-token GRPO objective."""

    policy_loss_tensor: torch.Tensor
    kl_loss_tensor: torch.Tensor
    total_loss_tensor: torch.Tensor
    mean_ratio: float
    mean_token_kl: float
    clip_ratio_region_mean: float
    clip_ratio_low_mean: float
    clip_ratio_high_mean: float

    @property
    def policy_loss(self) -> float:
        return float(self.policy_loss_tensor.detach().item())

    @property
    def kl_loss(self) -> float:
        return float(self.kl_loss_tensor.detach().item())

    @property
    def total_loss(self) -> float:
        return float(self.total_loss_tensor.detach().item())


def _gather_sampled_token_logprobs(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Gather logprobs for the sampled next-token actions."""

    token_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return token_logprobs.gather(dim=-1, index=targets).squeeze(-1)


def _compute_logprob_ratio(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Compute a clipped importance ratio from current-vs-old logprobs."""

    logprob_delta = (current_logprobs - old_logprobs).clamp(min=-clip_range, max=clip_range)
    return logprob_delta.exp()


def _compute_sampled_token_kl(
    current_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Approximate sampled-token KL with the TRL-style per-token estimator."""

    logprob_delta = ref_logprobs - current_logprobs
    return logprob_delta.exp() - logprob_delta - 1.0


def _masked_token_mean(values: torch.Tensor, sampled_token_mask: torch.Tensor) -> torch.Tensor:
    """Average values across active sampled completion tokens only."""

    mask = sampled_token_mask.to(dtype=values.dtype)
    token_count = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / token_count


def _compute_clipped_grpo_objective(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    sampled_token_mask: torch.Tensor,
    epsilon: float,
    beta: float,
    ref_logprobs: torch.Tensor | None,
    clip_range: float,
) -> GRPOObjectiveStats:
    """Compute the sampled-token clipped GRPO objective and parity metrics."""

    broadcast_advantages = advantages.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    ).unsqueeze(-1).expand_as(current_logprobs)
    ratios = _compute_logprob_ratio(current_logprobs, old_logprobs, clip_range=clip_range)
    clipped_ratios = ratios.clamp(min=1.0 - epsilon, max=1.0 + epsilon)

    unclipped_surrogate = ratios * broadcast_advantages
    clipped_surrogate = clipped_ratios * broadcast_advantages
    policy_terms = -torch.minimum(unclipped_surrogate, clipped_surrogate)
    policy_loss = _masked_token_mean(policy_terms, sampled_token_mask)

    if beta > 0.0:
        if ref_logprobs is None:
            raise ValueError("ref_logprobs is required when beta > 0.0.")
        token_kl = _compute_sampled_token_kl(current_logprobs, ref_logprobs)
        mean_token_kl_tensor = _masked_token_mean(token_kl, sampled_token_mask)
        kl_loss = policy_loss.new_tensor(beta) * mean_token_kl_tensor
    else:
        token_kl = torch.zeros_like(current_logprobs)
        mean_token_kl_tensor = policy_loss.new_zeros(())
        kl_loss = policy_loss.new_zeros(())

    total_loss = policy_loss + kl_loss
    low_clip = ratios < (1.0 - epsilon)
    high_clip = ratios > (1.0 + epsilon)
    clip_region = low_clip | high_clip

    return GRPOObjectiveStats(
        policy_loss_tensor=policy_loss,
        kl_loss_tensor=kl_loss,
        total_loss_tensor=total_loss,
        mean_ratio=float(_masked_token_mean(ratios, sampled_token_mask).detach().item()),
        mean_token_kl=float(mean_token_kl_tensor.detach().item()),
        clip_ratio_region_mean=float(
            _masked_token_mean(clip_region.to(dtype=current_logprobs.dtype), sampled_token_mask).detach().item()
        ),
        clip_ratio_low_mean=float(
            _masked_token_mean(low_clip.to(dtype=current_logprobs.dtype), sampled_token_mask).detach().item()
        ),
        clip_ratio_high_mean=float(
            _masked_token_mean(high_clip.to(dtype=current_logprobs.dtype), sampled_token_mask).detach().item()
        ),
    )


class GRPOTrainer:
    """Main entrypoint for rollout collection and GRPO optimization."""

    def __init__(
        self,
        config: GRPOConfig,
        environment: BaseEnvironment,
        verifier: BaseVerifier,
        tokenizer: Any | None = None,
        layout: SharedWeightLayout | Any | None = None,
        rollout_orchestrator: RolloutOrchestrator | None = None,
        draft_model: Any | None = None,
        metrics_logger: MetricsLogger | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
        profiler: SystemsProfiler | None = None,
        debugger: AgentRLDebugger | None = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.verifier = verifier
        self._set_seed()
        self._validate_experimental_flags()
        self.tokenizer = tokenizer or self._build_tokenizer()
        self.layout = layout or self._build_layout()
        self.device = self._resolve_device()
        self.rng = self._build_torch_generator()
        self._maybe_compile_model()
        self._maybe_autoconfigure_chunk_size()
        self.runtime_controller = ExecutionController(config=self.config, device=self.device)
        self.gradient_checkpointing_enabled = self._maybe_enable_gradient_checkpointing()

        self.rollout = rollout_orchestrator or self._build_rollout_orchestrator(draft_model=draft_model)
        self.metrics_logger = metrics_logger or MetricsLogger(
            output_dir=self.config.output_dir,
            jsonl_name=self.config.jsonl_metrics_name,
            log_to_wandb=self.config.log_to_wandb,
            wandb_project=self.config.wandb_project,
            wandb_run_name=self.config.wandb_run_name,
        )
        self.trajectory_buffer = trajectory_buffer or TrajectoryBuffer(
            output_dir=self.config.output_dir,
            max_batches=self.config.trajectory_buffer_max_batches,
        )
        self.profiler = profiler or SystemsProfiler()
        self.debugger = debugger
        self.optimizer = torch.optim.AdamW(
            self.layout.trainable_parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._build_lr_scheduler()
        self.current_beta = float(self.config.beta)
        self.startup_report = self._build_startup_report()
        self._log_startup_report()

    def _validate_experimental_flags(self) -> None:
        """Reject experimental runtime flags that are not implemented yet."""

        if self.config.use_async_rollout_workers:
            raise NotImplementedError(
                "use_async_rollout_workers is reserved for a future CPU worker path and is not implemented yet."
            )
        if self.config.use_async_trajectory_copy:
            raise NotImplementedError(
                "use_async_trajectory_copy is reserved for a future pinned-memory copy path and is not implemented yet."
            )
        if self.config.experimental_vllm_rollout:
            raise NotImplementedError(
                "experimental_vllm_rollout is reserved for a future optional vLLM rollout path and is not implemented yet."
            )

    def train(self) -> list[dict[str, float]]:
        """Run the configured GRPO training loop."""

        history: list[dict[str, float]] = []
        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        try:
            for step in range(self.config.steps):
                should_step = (
                    ((step + 1) % self.config.gradient_accumulation_steps == 0)
                    or step == self.config.steps - 1
                )
                with self.profiler as prof:
                    self._run_profiled_step(step=step, profiler=prof, should_step=should_step)
                    batch = self._last_profiled_batch
                    metrics = self._last_profiled_metrics

                self.trajectory_buffer.add(batch, step=step)
                if step % self.config.replay_every == 0:
                    self.trajectory_buffer.save(step)

                system_metrics = prof.metrics()
                effective_batch_tokens = float(batch.completion_mask.sum().item())
                total_ms = max(system_metrics.get("total_step_time_ms", 0.0), 1e-6)
                system_metrics["effective_batch_tokens"] = effective_batch_tokens
                system_metrics["tokens_per_second"] = effective_batch_tokens / (total_ms / 1000.0)
                system_metrics["padding_ratio"] = float(batch.metadata.get("padding_ratio", 0.0))
                system_metrics["padding_waste_tokens"] = float(batch.metadata.get("padding_waste_tokens", 0.0))
                system_metrics["generation_padding_ratio"] = float(
                    batch.metadata.get("generation_padding_ratio", 0.0)
                )
                system_metrics["generation_padding_waste_tokens"] = float(
                    batch.metadata.get("generation_padding_waste_tokens", 0.0)
                )
                system_metrics["sequence_padding_ratio"] = float(
                    batch.metadata.get("sequence_padding_ratio", 0.0)
                )
                system_metrics["sequence_padding_waste_tokens"] = float(
                    batch.metadata.get("sequence_padding_waste_tokens", 0.0)
                )
                system_metrics["prefill_time_ms"] = float(batch.metadata.get("prefill_time_ms", 0.0))
                system_metrics["decode_time_ms"] = float(batch.metadata.get("decode_time_ms", 0.0))
                system_metrics["prefill_tokens"] = float(batch.metadata.get("prefill_tokens", 0.0))
                system_metrics["decode_tokens"] = float(batch.metadata.get("decode_tokens", 0.0))
                system_metrics["prefill_tokens_per_second"] = float(
                    batch.metadata.get("prefill_tokens_per_second", 0.0)
                )
                system_metrics["decode_tokens_per_second"] = float(
                    batch.metadata.get("decode_tokens_per_second", 0.0)
                )
                system_metrics["cache_reuse_tokens"] = float(batch.metadata.get("cache_reuse_tokens", 0.0))
                system_metrics["cache_reuse_effectiveness"] = float(
                    batch.metadata.get("cache_reuse_effectiveness", 0.0)
                )
                system_metrics["paged_kv_block_size_tokens"] = float(
                    batch.metadata.get("paged_kv_block_size_tokens", 0.0)
                )
                system_metrics["paged_kv_free_block_count"] = float(
                    batch.metadata.get("paged_kv_free_block_count", 0.0)
                )
                system_metrics["paged_kv_used_block_count"] = float(
                    batch.metadata.get("paged_kv_used_block_count", 0.0)
                )
                system_metrics["paged_kv_allocator_occupancy"] = float(
                    batch.metadata.get("paged_kv_allocator_occupancy", 0.0)
                )
                system_metrics["paged_kv_block_reuse_count"] = float(
                    batch.metadata.get("paged_kv_block_reuse_count", 0.0)
                )
                system_metrics["paged_kv_allocator_pressure"] = float(
                    batch.metadata.get("paged_kv_allocator_pressure", 0.0)
                )
                system_metrics["paged_kv_max_blocks_in_use"] = float(
                    batch.metadata.get("paged_kv_max_blocks_in_use", 0.0)
                )
                system_metrics["paged_kv_resident_sequences"] = float(
                    batch.metadata.get("paged_kv_resident_sequences", 0.0)
                )
                system_metrics["paged_kv_preempted_sequences"] = float(
                    batch.metadata.get("paged_kv_preempted_sequences", 0.0)
                )
                system_metrics["paged_kv_max_preempted_sequences"] = float(
                    batch.metadata.get("paged_kv_max_preempted_sequences", 0.0)
                )
                system_metrics["scheduler_prefill_block_budget"] = float(
                    batch.metadata.get("scheduler_prefill_block_budget", 0.0)
                )
                system_metrics["scheduler_decode_block_budget"] = float(
                    batch.metadata.get("scheduler_decode_block_budget", 0.0)
                )
                system_metrics["scheduler_prefill_admitted_blocks"] = float(
                    batch.metadata.get("scheduler_prefill_admitted_blocks", 0.0)
                )
                system_metrics["scheduler_decode_admitted_blocks"] = float(
                    batch.metadata.get("scheduler_decode_admitted_blocks", 0.0)
                )
                system_metrics["scheduler_prefill_token_budget"] = float(
                    batch.metadata.get("scheduler_prefill_token_budget", 0.0)
                )
                system_metrics["scheduler_decode_token_budget"] = float(
                    batch.metadata.get("scheduler_decode_token_budget", 0.0)
                )
                system_metrics["scheduler_prefill_passes"] = float(
                    batch.metadata.get("scheduler_prefill_passes", 0.0)
                )
                system_metrics["scheduler_decode_passes"] = float(
                    batch.metadata.get("scheduler_decode_passes", 0.0)
                )
                system_metrics["scheduler_prefill_admitted_sequences"] = float(
                    batch.metadata.get("scheduler_prefill_admitted_sequences", 0.0)
                )
                system_metrics["scheduler_decode_admitted_sequences"] = float(
                    batch.metadata.get("scheduler_decode_admitted_sequences", 0.0)
                )
                system_metrics["scheduler_prefill_kv_budget_mb"] = float(
                    batch.metadata.get("scheduler_prefill_kv_budget_mb", 0.0)
                )
                system_metrics["scheduler_decode_kv_budget_mb"] = float(
                    batch.metadata.get("scheduler_decode_kv_budget_mb", 0.0)
                )
                system_metrics["scheduler_prefill_admitted_kv_mb"] = float(
                    batch.metadata.get("scheduler_prefill_admitted_kv_mb", 0.0)
                )
                system_metrics["scheduler_decode_admitted_kv_mb"] = float(
                    batch.metadata.get("scheduler_decode_admitted_kv_mb", 0.0)
                )
                system_metrics["scheduler_prefill_kv_pressure"] = float(
                    batch.metadata.get("scheduler_prefill_kv_pressure", 0.0)
                )
                system_metrics["scheduler_decode_kv_pressure"] = float(
                    batch.metadata.get("scheduler_decode_kv_pressure", 0.0)
                )
                system_metrics["scheduler_length_sort_passes"] = float(
                    batch.metadata.get("scheduler_length_sort_passes", 0.0)
                )
                system_metrics["scheduler_length_sorted_sequences"] = float(
                    batch.metadata.get("scheduler_length_sorted_sequences", 0.0)
                )
                system_metrics["scheduler_deferred_sequences"] = float(
                    batch.metadata.get("scheduler_deferred_sequences", 0.0)
                )
                system_metrics["scheduler_max_concurrent_sequences"] = float(
                    batch.metadata.get("scheduler_max_concurrent_sequences", 0.0)
                )
                system_metrics["rollout_peak_vram_mb"] = float(system_metrics.get("generation_peak_vram_mb", 0.0))
                system_metrics["rollout_runtime_headroom_mb"] = float(
                    system_metrics.get("generation_runtime_headroom_mb", 0.0)
                )
                system_metrics.update(self.runtime_controller.observe(system_metrics))
                merged_metrics = {**metrics, **system_metrics}
                self.metrics_logger.log(step, merged_metrics)
                if self.debugger is not None:
                    self.debugger.capture(step, batch, merged_metrics)
                metrics_with_step = {"step": float(step), **merged_metrics}
                history.append(metrics_with_step)
                if (step + 1) % self.config.save_every == 0:
                    self._save_adapter_checkpoint(step + 1)
        finally:
            self._save_adapter_checkpoint(self.config.steps, suffix="final")
            self.metrics_logger.close()
        return history

    def step(
        self,
        batch: RolloutBatch,
        perform_optimizer_step: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one GRPO update from a collected rollout batch."""

        self.layout.model.train()
        if perform_optimizer_step or self.config.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad(set_to_none=True)

        flat_input_ids = batch.input_ids.view(-1, batch.input_ids.shape[-1])
        flat_attention_mask = batch.attention_mask.view(-1, batch.attention_mask.shape[-1])
        flat_completion_mask = batch.completion_mask.view(-1, batch.completion_mask.shape[-1])
        flat_advantages = batch.advantages.reshape(-1)

        autocast_context = self._autocast_context()
        current_lr = self._current_learning_rate()
        with autocast_context:
            policy_logits = self.layout.policy_forward(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )
            current_logprobs = _gather_sampled_token_logprobs(flat_input_ids, policy_logits)
            old_logprobs = batch.old_policy_logprobs.view(-1, batch.old_policy_logprobs.shape[-1])[:, 1:].to(
                device=current_logprobs.device,
                dtype=current_logprobs.dtype,
            )
            sampled_token_mask = flat_completion_mask[:, 1:]

            ref_logprobs = None
            if self.current_beta > 0.0:
                with torch.no_grad():
                    ref_logits = self.layout.reference_forward(
                        input_ids=flat_input_ids,
                        attention_mask=flat_attention_mask,
                    )
                ref_logprobs = _gather_sampled_token_logprobs(flat_input_ids, ref_logits)

            objective = _compute_clipped_grpo_objective(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                advantages=flat_advantages,
                sampled_token_mask=sampled_token_mask,
                epsilon=self.config.epsilon,
                beta=self.current_beta,
                ref_logprobs=ref_logprobs,
                clip_range=self.config.clip_range,
            )

        scaled_loss = objective.total_loss_tensor / float(self.config.gradient_accumulation_steps)
        scaled_loss.backward()

        if perform_optimizer_step:
            torch.nn.utils.clip_grad_norm_(list(self.layout.trainable_parameters()), self.config.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        metrics = self._build_metrics(batch, objective)
        metrics["learning_rate"] = current_lr
        metrics["beta"] = self.current_beta
        self._log_degenerate_batch_warnings(batch, metrics)
        return objective.total_loss_tensor.detach(), metrics

    def _build_metrics(
        self,
        batch: RolloutBatch,
        objective: GRPOObjectiveStats,
    ) -> dict[str, float]:
        """Collect scalar metrics required by the GRPO contract."""

        return {
            "mean_reward": float(batch.rewards.mean().item()),
            "reward_std": float(batch.rewards.std(unbiased=False).item()),
            "policy_loss": objective.policy_loss,
            "kl_loss": objective.kl_loss,
            "mean_token_kl": objective.mean_token_kl,
            "total_loss": objective.total_loss,
            "advantage_mean": float(batch.advantages.mean().item()),
            "advantage_std": float(batch.advantages.std(unbiased=False).item()),
            "unique_response_ratio": float(batch.metadata.get("unique_response_ratio", 0.0)),
            "clip_ratio/region_mean": objective.clip_ratio_region_mean,
            "clip_ratio/low_mean": objective.clip_ratio_low_mean,
            "clip_ratio/high_mean": objective.clip_ratio_high_mean,
            "mean_ratio": objective.mean_ratio,
        }

    def _build_lr_scheduler(self):
        """Construct the configured learning-rate scheduler."""

        if self.config.lr_scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=self._constant_lr_lambda,
            )
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._cosine_lr_lambda,
        )

    def _constant_lr_lambda(self, step: int) -> float:
        """Return the LR multiplier for constant-with-warmup scheduling."""

        if step <= 0:
            return 1.0
        if self.config.warmup_steps <= 0:
            return 1.0
        if step < self.config.warmup_steps:
            return float(step) / float(self.config.warmup_steps)
        return 1.0

    def _cosine_lr_lambda(self, step: int) -> float:
        """Return the LR multiplier for cosine decay with optional warmup."""

        if step <= 0:
            return 1.0
        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return float(step) / float(self.config.warmup_steps)

        decay_steps = max(self.config.steps - self.config.warmup_steps, 1)
        progress = min(max(step - self.config.warmup_steps, 0), decay_steps) / float(decay_steps)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def _current_learning_rate(self) -> float:
        """Return the active optimizer learning rate."""

        return float(self.optimizer.param_groups[0]["lr"])

    def _run_profiled_step(self, step: int, profiler: SystemsProfiler, should_step: bool) -> None:
        """Run one train step, optionally exporting a torch profiler trace."""

        self._last_profiled_batch = None
        self._last_profiled_metrics = None
        if self.config.profile_steps is None or step >= self.config.profile_steps:
            batch, metrics = self._execute_step_with_recovery(profiler, should_step)
            self._last_profiled_batch = batch
            self._last_profiled_metrics = metrics
            return

        try:
            from torch.profiler import ProfilerActivity, profile
        except ImportError:
            LOGGER.warning("torch.profiler is unavailable; skipping profile export for step %s", step)
            batch, metrics = self._execute_step_with_recovery(profiler, should_step)
            self._last_profiled_batch = batch
            self._last_profiled_metrics = metrics
            return

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.config.profile_path.mkdir(parents=True, exist_ok=True)
        trace_path = self.config.profile_path / f"step_{step:06d}_chrome_trace.json"
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as torch_profiler:
            batch, metrics = self._execute_step_with_recovery(profiler, should_step)
        torch_profiler.export_chrome_trace(str(trace_path))
        metrics["profile_trace_path"] = str(trace_path)
        self._last_profiled_batch = batch
        self._last_profiled_metrics = metrics

    def _execute_step_with_recovery(
        self,
        profiler: SystemsProfiler,
        should_step: bool,
    ) -> tuple[RolloutBatch, dict[str, float]]:
        """Collect and train once, retrying with safer settings after OOM."""

        while True:
            try:
                with profiler.phase("generation"):
                    batch = self.rollout.collect()
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise
                if not self.runtime_controller.handle_oom(stage="generation"):
                    raise
                self._clear_runtime_oom_state()
                continue

            try:
                with profiler.phase("training"):
                    _, metrics = self.step(batch, perform_optimizer_step=should_step)
                return batch, metrics
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise
                if not self.runtime_controller.handle_oom(stage="training"):
                    raise
                self._clear_runtime_oom_state()

    def _is_cuda_oom(self, exc: RuntimeError) -> bool:
        """Return True when the exception looks like a device OOM."""

        return "out of memory" in str(exc).lower()

    def _clear_runtime_oom_state(self) -> None:
        """Clear accumulated gradients and CUDA caches before retrying."""

        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_degenerate_batch_warnings(
        self,
        batch: RolloutBatch,
        metrics: dict[str, float],
    ) -> None:
        """Emit explicit warnings for collapsed reward and exploration patterns."""

        if metrics["reward_std"] == 0.0:
            LOGGER.warning(
                "reward_std == 0 for this batch. The verifier may be too coarse or the model has collapsed."
            )
        if metrics["unique_response_ratio"] < 0.3:
            repeated = None
            responses = batch.metadata.get("responses", [])
            for group in responses:
                if len(set(group)) < len(group):
                    counts: dict[str, int] = {}
                    for response in group:
                        counts[response] = counts.get(response, 0) + 1
                    repeated = max(counts, key=counts.get)
                    break
            LOGGER.warning(
                "unique_response_ratio < 0.3 (%.3f). Repeated response: %r",
                metrics["unique_response_ratio"],
                repeated,
            )

    def _build_tokenizer(self) -> Any:
        """Load a tokenizer for the configured model."""

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("GRPOTrainer requires `transformers` to load a tokenizer.") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _build_layout(self) -> SharedWeightLayout:
        """Build the shared base/reference layout from config."""

        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise ImportError("GRPOTrainer requires `peft` to construct the LoRA adapter.") from exc

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        return SharedWeightLayout(
            model_name=self.config.model_name,
            lora_config=lora_config,
            dtype=self.config.dtype,
            device=self._resolve_device(),
            trust_remote_code=self.config.trust_remote_code,
            sdpa_backend=self.config.sdpa_backend,
            adapter_path=self.config.init_adapter_path,
        )

    def _build_rollout_orchestrator(self, draft_model: Any | None = None) -> RolloutOrchestrator:
        """Select the configured rollout orchestrator implementation."""

        common_kwargs = {
            "config": self.config,
            "environment": self.environment,
            "verifier": self.verifier,
            "tokenizer": self.tokenizer,
            "layout": self.layout,
            "device": self.device,
            "rng": self.rng,
        }
        if self.config.use_speculative_decoding:
            return SpeculativeRolloutOrchestrator(draft_model=draft_model, **common_kwargs)
        if self.config.use_continuous_batching:
            return ContinuousBatchingOrchestrator(**common_kwargs)
        return RolloutOrchestrator(**common_kwargs)

    def _resolve_device(self) -> torch.device:
        """Resolve the single-device execution policy for the trainer."""

        requested = self.config.device or "auto"
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def _autocast_context(self):
        """Return an autocast context aligned with the configured dtype."""

        if self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        if self.config.dtype not in {"float16", "bfloat16"}:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.config.dtype))

    def _set_seed(self) -> None:
        """Apply the configured global seed across Python, NumPy, and Torch."""

        random.seed(self.config.seed)
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _build_torch_generator(self) -> torch.Generator:
        """Build a dedicated RNG for rollout-time sampling."""

        generator = torch.Generator(device=self.device.type if self.device.type == "cuda" else "cpu")
        generator.manual_seed(self.config.seed)
        return generator

    def _maybe_compile_model(self) -> None:
        """Optionally torch.compile the hot model path."""

        if not self.config.torch_compile or not hasattr(torch, "compile"):
            return
        self.layout.model = torch.compile(self.layout.model)

    def _maybe_autoconfigure_chunk_size(self) -> None:
        """Infer a safe continuous-batching chunk size when possible."""

        if self.config.chunk_size is not None:
            return
        if not self.config.auto_tune_chunk_size:
            return
        if not self.config.use_continuous_batching:
            return
        if self.device.type != "cuda":
            return

        model_config = getattr(self.layout.model, "config", None)
        if model_config is None:
            return
        try:
            self.config.chunk_size = compute_safe_chunk_size(self.config, model_config)
        except (AttributeError, ValueError):
            return

    def _save_adapter_checkpoint(self, step: int, suffix: str | None = None) -> Path | None:
        """Persist the current LoRA adapter state when the layout supports it."""

        save_adapter = getattr(self.layout, "save_adapter", None)
        if save_adapter is None:
            LOGGER.warning("Layout does not expose save_adapter(); skipping adapter checkpoint save.")
            return None

        checkpoint_name = (
            f"{self.config.checkpoint_prefix}_{suffix}"
            if suffix is not None
            else f"{self.config.checkpoint_prefix}_{step:06d}"
        )
        output_path = self.config.output_path / checkpoint_name
        saved_path = save_adapter(output_path)
        LOGGER.info("Saved adapter checkpoint: %s", saved_path)
        return Path(saved_path)

    def _build_startup_report(self) -> dict[str, float | str | None]:
        """Collect one startup snapshot of parameter VRAM and live device headroom."""

        report: dict[str, float | str | None] = {
            "device": self.device.type,
            "device_name": str(self.device),
            "gradient_checkpointing_requested": self.config.use_gradient_checkpointing,
            "gradient_checkpointing_enabled": self.gradient_checkpointing_enabled,
        }

        vram_report = getattr(self.layout, "vram_report", None)
        if vram_report is not None:
            report.update({
                f"parameter_{key}": float(value)
                for key, value in vram_report().items()
            })

        if self.device.type != "cuda" or not torch.cuda.is_available():
            report.update(
                {
                    "device_total_mb": None,
                    "device_free_mb": None,
                    "device_allocated_mb": None,
                    "device_reserved_mb": None,
                    "runtime_headroom_mb": None,
                }
            )
            model_config = getattr(self.layout.model, "config", None)
            report.update(self.runtime_controller.build_preflight_report(report, model_config))
            return report

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        report.update(
            {
                "device_total_mb": total_bytes / (1024 * 1024),
                "device_free_mb": free_bytes / (1024 * 1024),
                "device_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 * 1024),
                "device_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 * 1024),
                "runtime_headroom_mb": free_bytes / (1024 * 1024),
            }
        )
        model_config = getattr(self.layout.model, "config", None)
        report.update(self.runtime_controller.build_preflight_report(report, model_config))
        return report

    def _maybe_enable_gradient_checkpointing(self) -> bool:
        """Enable gradient checkpointing and input-gradient hooks when supported."""

        if not self.config.use_gradient_checkpointing:
            return False

        enabled = False
        input_grads_enabled = False
        seen: set[int] = set()
        model_candidates = [
            getattr(self.layout, "model", None),
            getattr(getattr(self.layout, "model", None), "model", None),
            getattr(getattr(self.layout, "model", None), "base_model", None),
        ]

        get_base_model = getattr(getattr(self.layout, "model", None), "get_base_model", None)
        if callable(get_base_model):
            try:
                model_candidates.append(get_base_model())
            except Exception:  # pragma: no cover - defensive around HF/PEFT wrappers
                LOGGER.debug("get_base_model() failed while enabling gradient checkpointing.", exc_info=True)

        for candidate in model_candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))

            enable_input_require_grads = getattr(candidate, "enable_input_require_grads", None)
            if callable(enable_input_require_grads):
                enable_input_require_grads()
                input_grads_enabled = True

            gradient_checkpointing_enable = getattr(candidate, "gradient_checkpointing_enable", None)
            if callable(gradient_checkpointing_enable):
                gradient_checkpointing_enable()
                enabled = True

        if self.config.use_gradient_checkpointing and not enabled:
            LOGGER.warning(
                "use_gradient_checkpointing=True but the loaded model does not expose "
                "gradient_checkpointing_enable()."
            )
        elif enabled:
            LOGGER.info(
                "gradient checkpointing enabled%s",
                " with input gradient hooks" if input_grads_enabled else "",
            )
        return enabled

    def _log_startup_report(self) -> None:
        """Emit a concise startup device and memory summary."""

        report = self.startup_report
        LOGGER.info(
            "startup device=%s | parameter_total_mb=%s | free_mb=%s | reserved_mb=%s | allocated_mb=%s | "
            "chunk_size=%s | preflight_risk=%s | recommendation=%s",
            report.get("device_name"),
            self._format_optional_metric(report.get("parameter_total_mb")),
            self._format_optional_metric(report.get("device_free_mb")),
            self._format_optional_metric(report.get("device_reserved_mb")),
            self._format_optional_metric(report.get("device_allocated_mb")),
            self._format_optional_metric(report.get("current_chunk_size")),
            self._format_optional_metric(report.get("preflight_risk")),
            self._format_optional_metric(report.get("execution_recommendation")),
        )

    def _format_optional_metric(self, value: float | str | None) -> str:
        """Format optional numeric metrics for startup logging."""

        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

"""GRPO trainer for AgentRL."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from contextlib import nullcontext
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
        self.tokenizer = tokenizer or self._build_tokenizer()
        self.layout = layout or self._build_layout()
        self.device = self._resolve_device()
        self.rng = self._build_torch_generator()
        self._maybe_compile_model()
        self._maybe_autoconfigure_chunk_size()

        if self.config.use_gradient_checkpointing:
            gradient_checkpointing_enable = getattr(self.layout.model, "gradient_checkpointing_enable", None)
            if gradient_checkpointing_enable is not None:
                gradient_checkpointing_enable()

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

    def train(self) -> list[dict[str, float]]:
        """Run the configured GRPO training loop."""

        history: list[dict[str, float]] = []
        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        try:
            for step in range(self.config.steps):
                with self.profiler as prof:
                    with prof.phase("generation"):
                        batch = self.rollout.collect()
                    with prof.phase("training"):
                        should_step = (
                            ((step + 1) % self.config.gradient_accumulation_steps == 0)
                            or step == self.config.steps - 1
                        )
                        _, metrics = self.step(batch, perform_optimizer_step=should_step)

                self.trajectory_buffer.add(batch, step=step)
                if step % self.config.replay_every == 0:
                    self.trajectory_buffer.save(step)

                system_metrics = prof.metrics()
                effective_batch_tokens = float(batch.action_mask.sum().item())
                total_ms = max(system_metrics.get("total_step_time_ms", 0.0), 1e-6)
                system_metrics["effective_batch_tokens"] = effective_batch_tokens
                system_metrics["tokens_per_second"] = effective_batch_tokens / (total_ms / 1000.0)
                system_metrics["padding_ratio"] = float(batch.metadata.get("padding_ratio", 0.0))
                merged_metrics = {**metrics, **system_metrics}
                self.metrics_logger.log(step, merged_metrics)
                if self.debugger is not None:
                    self.debugger.capture(step, batch, merged_metrics)
                metrics_with_step = {"step": float(step), **merged_metrics}
                history.append(metrics_with_step)
        finally:
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
        flat_action_mask = batch.action_mask.view(-1, batch.action_mask.shape[-1])
        flat_advantages = batch.advantages.reshape(-1)

        autocast_context = self._autocast_context()
        with autocast_context:
            policy_logits = self.layout.policy_forward(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )
            with torch.no_grad():
                ref_logits = self.layout.reference_forward(
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,
                )

            token_logprobs, token_ref_logprobs, token_kl = self._token_statistics(
                flat_input_ids,
                policy_logits,
                ref_logits,
            )
            masked_action = flat_action_mask[:, 1:].to(dtype=policy_logits.dtype)
            sequence_delta = ((token_logprobs - token_ref_logprobs) * masked_action).sum(dim=-1)

            policy_loss = -(flat_advantages * sequence_delta).mean()
            action_token_count = masked_action.sum().clamp(min=1.0)
            kl_loss = self.config.beta * (token_kl * masked_action).sum() / action_token_count
            total_loss = policy_loss + kl_loss

        scaled_loss = total_loss / float(self.config.gradient_accumulation_steps)
        scaled_loss.backward()

        if perform_optimizer_step:
            torch.nn.utils.clip_grad_norm_(list(self.layout.trainable_parameters()), self.config.max_grad_norm)
            self.optimizer.step()

        metrics = self._build_metrics(batch, policy_loss, kl_loss, total_loss)
        self._log_degenerate_batch_warnings(batch, metrics)
        return total_loss.detach(), metrics

    def _token_statistics(
        self,
        input_ids: torch.Tensor,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute token logprobs and analytical per-token KL."""

        policy_logprobs = torch.log_softmax(policy_logits[:, :-1, :], dim=-1)
        ref_logprobs = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        gathered_policy = policy_logprobs.gather(dim=-1, index=targets).squeeze(-1)
        gathered_ref = ref_logprobs.gather(dim=-1, index=targets).squeeze(-1)

        policy_probs = policy_logprobs.exp()
        token_kl = (policy_probs * (policy_logprobs - ref_logprobs)).sum(dim=-1)
        return gathered_policy, gathered_ref, token_kl

    def _build_metrics(
        self,
        batch: RolloutBatch,
        policy_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> dict[str, float]:
        """Collect scalar metrics required by the GRPO contract."""

        return {
            "mean_reward": float(batch.rewards.mean().item()),
            "reward_std": float(batch.rewards.std(unbiased=False).item()),
            "policy_loss": float(policy_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "advantage_mean": float(batch.advantages.mean().item()),
            "advantage_std": float(batch.advantages.std(unbiased=False).item()),
            "unique_response_ratio": float(batch.metadata.get("unique_response_ratio", 0.0)),
        }

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

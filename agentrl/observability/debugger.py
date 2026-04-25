"""Step-level debugging helpers for AgentRL."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import torch

from agentrl.core.rollout import RolloutBatch


@dataclass(slots=True)
class _DebugSnapshot:
    step: int
    batch: RolloutBatch
    metrics: dict[str, Any] = field(default_factory=dict)
    exception_text: str | None = None
    gpu_state: dict[str, Any] = field(default_factory=dict)


class AgentRLDebugger:
    """Capture low-reward or failing batches for later inspection."""

    def __init__(self, reward_threshold: float = 0.1) -> None:
        self.reward_threshold = reward_threshold
        self._snapshots: dict[int, _DebugSnapshot] = {}

    def __enter__(self) -> "AgentRLDebugger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        if exc is not None:
            exception_text = "".join(traceback.format_exception(exc_type, exc, tb))
            self.capture_exception(step=-1, exception_text=exception_text)
        return False

    def capture(self, step: int, batch: RolloutBatch, metrics: dict[str, Any]) -> None:
        """Capture a batch when reward falls below the configured threshold."""

        if float(batch.rewards.max().item()) >= self.reward_threshold:
            return
        self._snapshots[step] = _DebugSnapshot(
            step=step,
            batch=self._detach_batch(batch),
            metrics=dict(metrics),
            gpu_state=self._gpu_state(),
        )

    def capture_exception(
        self,
        step: int,
        batch: RolloutBatch | None = None,
        metrics: dict[str, Any] | None = None,
        exception_text: str | None = None,
    ) -> None:
        """Capture a crashing batch plus traceback text and GPU state."""

        captured_batch = self._detach_batch(batch) if batch is not None else self._empty_batch()
        self._snapshots[step] = _DebugSnapshot(
            step=step,
            batch=captured_batch,
            metrics=dict(metrics or {}),
            exception_text=exception_text,
            gpu_state=self._gpu_state(),
        )

    def debug_episode(self, step: int) -> str:
        """Render token-level policy/reference comparisons for one captured step."""

        snapshot = self._snapshots.get(step)
        if snapshot is None:
            raise KeyError(f"No debug snapshot captured for step {step}.")

        batch = snapshot.batch
        responses = batch.metadata.get("responses", [])
        prompts = batch.metadata.get("prompts", [])
        lines: list[str] = [f"=== Debug Step {step} ==="]
        if snapshot.exception_text:
            lines.append("Exception:")
            lines.append(snapshot.exception_text.rstrip())

        if prompts:
            lines.append(f'Prompt: "{prompts[0]}"')
        if responses:
            lines.append(f'Final responses: {responses[0]!r}')

        flat_old_policy = batch.old_policy_logprobs.view(-1, batch.old_policy_logprobs.shape[-1])
        flat_completion = batch.completion_mask.view(-1, batch.completion_mask.shape[-1])
        flat_ids = batch.input_ids.view(-1, batch.input_ids.shape[-1])
        flat_rewards = batch.rewards.reshape(-1)
        flat_advantages = batch.advantages.reshape(-1)

        for sequence_index in range(flat_ids.shape[0]):
            lines.append(
                f"[Sequence {sequence_index + 1}] reward={float(flat_rewards[sequence_index].item()):.2f} "
                f"| advantage={float(flat_advantages[sequence_index].item()):+.2f}"
            )
            for token_position in range(flat_ids.shape[-1]):
                if not bool(flat_completion[sequence_index, token_position].item()):
                    continue
                token_id = int(flat_ids[sequence_index, token_position].item())
                old_policy_lp = float(flat_old_policy[sequence_index, token_position].item())
                marker = " *" if abs(old_policy_lp) > 1.0 else ""
                lines.append(
                    f"  token={token_id} | log_prob(old_policy)={old_policy_lp:.4f}{marker}"
                )

        if snapshot.metrics:
            lines.append(f"Metrics: {snapshot.metrics}")
        if snapshot.gpu_state:
            lines.append(f"GPU state: {snapshot.gpu_state}")
        return "\n".join(lines)

    def _detach_batch(self, batch: RolloutBatch) -> RolloutBatch:
        return RolloutBatch(
            input_ids=batch.input_ids.detach().cpu(),
            attention_mask=batch.attention_mask.detach().cpu(),
            completion_mask=batch.completion_mask.detach().cpu(),
            old_policy_logprobs=batch.old_policy_logprobs.detach().cpu(),
            rewards=batch.rewards.detach().cpu(),
            advantages=batch.advantages.detach().cpu(),
            metadata=batch.metadata,
        )

    def _empty_batch(self) -> RolloutBatch:
        empty_long = torch.zeros((1, 1, 1), dtype=torch.long)
        empty_float = torch.zeros((1, 1, 1), dtype=torch.float32)
        empty_reward = torch.zeros((1, 1), dtype=torch.float32)
        return RolloutBatch(
            input_ids=empty_long,
            attention_mask=empty_long.clone(),
            completion_mask=empty_long.to(dtype=torch.bool),
            old_policy_logprobs=empty_float,
            rewards=empty_reward,
            advantages=empty_reward.clone(),
            metadata={},
        )

    def _gpu_state(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        }

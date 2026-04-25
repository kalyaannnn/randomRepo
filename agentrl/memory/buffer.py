"""Trajectory buffer and serialization helpers for AgentRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from agentrl.core.rollout import RolloutBatch


class TrajectoryBuffer:
    """Keep rollout batches on-device and serialize them compactly for replay."""

    def __init__(self, output_dir: str = "./checkpoints", max_batches: int = 8) -> None:
        """Create an empty buffer rooted at the configured output directory.

        Args:
            output_dir: Base directory where trajectory files are written.
        """

        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_batches = max_batches
        self._memory_batches: dict[int, RolloutBatch] = {}

    def add(self, batch: RolloutBatch, step: int) -> None:
        """Store a rollout batch in memory without moving tensors off device.

        Args:
            batch: Batch to retain.
            step: Training step associated with this batch.
        """

        self._memory_batches[step] = batch
        self._evict_if_needed()

    def save(self, step: int) -> Path:
        """Serialize one retained batch to disk using compact tensor dtypes.

        Args:
            step: Step whose retained batch should be written.

        Returns:
            Path to the written trajectory file.
        """

        batch = self._memory_batches.get(step)
        if batch is None:
            raise KeyError(f"No in-memory rollout batch found for step {step}.")

        path = self.output_dir / f"trajectory_{step:06d}.pt"
        payload = self._serialize_batch(batch, step=step)
        torch.save(payload, path, pickle_protocol=5)
        return path

    def load(self, step: int, device: torch.device | str | None = None) -> RolloutBatch:
        """Load a serialized rollout batch from disk or memory.

        Args:
            step: Step identifier to load.
            device: Optional device to move tensors onto after deserialization.

        Returns:
            The reconstructed rollout batch.
        """

        if step in self._memory_batches and device is None:
            return self._memory_batches[step]

        path = self.output_dir / f"trajectory_{step:06d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No serialized trajectory exists for step {step}: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        batch = self._deserialize_batch(payload)
        if device is not None:
            batch = self._move_batch(batch, torch.device(device))
        return batch

    def filter(self, min_reward: float = 0.5) -> list[RolloutBatch]:
        """Return saved or retained batches containing at least one good trajectory.

        Args:
            min_reward: Minimum per-trajectory reward threshold.

        Returns:
            All batches where `max(reward) >= min_reward`.
        """

        matching: list[RolloutBatch] = []
        seen_steps: set[int] = set()

        for step, batch in self._memory_batches.items():
            if float(batch.rewards.max().item()) >= min_reward:
                matching.append(batch)
            seen_steps.add(step)

        for path in sorted(self.output_dir.glob("trajectory_*.pt")):
            step = int(path.stem.split("_")[-1])
            if step in seen_steps:
                continue
            batch = self.load(step)
            if float(batch.rewards.max().item()) >= min_reward:
                matching.append(batch)

        return matching

    def size_bytes(self) -> int:
        """Return total disk usage of serialized trajectory files."""

        return sum(path.stat().st_size for path in self.output_dir.glob("trajectory_*.pt"))

    def _serialize_batch(self, batch: RolloutBatch, step: int) -> dict[str, Any]:
        """Convert a rollout batch into a compact CPU payload for `torch.save()`."""

        vocab_upper = int(batch.input_ids.max().item()) if batch.input_ids.numel() else 0
        token_dtype = torch.int16 if vocab_upper < 65536 else torch.int32

        return {
            "step": step,
            "input_ids": batch.input_ids.detach().to(device="cpu", dtype=token_dtype),
            "attention_mask": batch.attention_mask.detach().to(device="cpu", dtype=torch.uint8),
            "completion_mask": batch.completion_mask.detach().to(device="cpu", dtype=torch.bool),
            "old_policy_logprobs": batch.old_policy_logprobs.detach().to(device="cpu", dtype=torch.float16),
            "rewards": batch.rewards.detach().to(device="cpu", dtype=torch.float32),
            "advantages": batch.advantages.detach().to(device="cpu", dtype=torch.float32),
            "metadata": batch.metadata,
        }

    def _deserialize_batch(self, payload: dict[str, Any]) -> RolloutBatch:
        """Reconstruct a rollout batch from a serialized CPU payload."""

        completion_mask = payload["completion_mask"] if "completion_mask" in payload else payload["action_mask"]
        old_policy_logprobs = (
            payload["old_policy_logprobs"] if "old_policy_logprobs" in payload else payload["policy_logprobs"]
        )
        return RolloutBatch(
            input_ids=payload["input_ids"].to(dtype=torch.long),
            attention_mask=payload["attention_mask"].to(dtype=torch.long),
            completion_mask=completion_mask.to(dtype=torch.bool),
            old_policy_logprobs=old_policy_logprobs.to(dtype=torch.float32),
            rewards=payload["rewards"].to(dtype=torch.float32),
            advantages=payload["advantages"].to(dtype=torch.float32),
            metadata=payload["metadata"],
        )

    def _move_batch(self, batch: RolloutBatch, device: torch.device) -> RolloutBatch:
        """Move all tensor fields in a rollout batch to the requested device."""

        return RolloutBatch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            completion_mask=batch.completion_mask.to(device),
            old_policy_logprobs=batch.old_policy_logprobs.to(device),
            rewards=batch.rewards.to(device),
            advantages=batch.advantages.to(device),
            metadata=batch.metadata,
        )

    def _evict_if_needed(self) -> None:
        """Evict oldest in-memory batches once the cap is exceeded."""

        while len(self._memory_batches) > self.max_batches:
            oldest_step = min(self._memory_batches)
            del self._memory_batches[oldest_step]

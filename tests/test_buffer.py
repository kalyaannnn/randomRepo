from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agentrl.core.rollout import RolloutBatch
from agentrl.memory.buffer import TrajectoryBuffer


def make_batch() -> RolloutBatch:
    return RolloutBatch(
        input_ids=torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.long),
        attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
        completion_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
        old_policy_logprobs=torch.tensor([[[0.0, -0.1, -0.2], [0.0, -0.3, -0.4]]], dtype=torch.float32),
        rewards=torch.tensor([[1.0, 0.25]], dtype=torch.float32),
        advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        metadata={"responses": [["good", "bad"]]},
    )


def test_buffer_add_keeps_same_in_memory_batch(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path))
    batch = make_batch()

    buffer.add(batch, step=7)

    assert buffer.load(7) is batch


def test_buffer_save_uses_compact_dtypes_and_roundtrips(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path))
    batch = make_batch()
    buffer.add(batch, step=12)

    path = buffer.save(12)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    loaded = buffer.load(12, device="cpu")

    assert path.name == "trajectory_000012.pt"
    assert payload["input_ids"].dtype == torch.int16
    assert payload["old_policy_logprobs"].dtype == torch.float16
    assert loaded.input_ids.dtype == torch.long
    assert loaded.old_policy_logprobs.dtype == torch.float32
    assert loaded.metadata == batch.metadata
    assert torch.equal(loaded.rewards, batch.rewards)


def test_buffer_uses_int32_when_token_ids_exceed_int16_range(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path))
    batch = make_batch()
    batch = RolloutBatch(
        input_ids=batch.input_ids + 70000,
        attention_mask=batch.attention_mask,
        completion_mask=batch.completion_mask,
        old_policy_logprobs=batch.old_policy_logprobs,
        rewards=batch.rewards,
        advantages=batch.advantages,
        metadata=batch.metadata,
    )
    buffer.add(batch, step=3)

    path = buffer.save(3)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["input_ids"].dtype == torch.int32


def test_buffer_filter_and_size_bytes_include_saved_trajectories(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path))
    good_batch = make_batch()
    bad_batch = RolloutBatch(
        input_ids=good_batch.input_ids.clone(),
        attention_mask=good_batch.attention_mask.clone(),
        completion_mask=good_batch.completion_mask.clone(),
        old_policy_logprobs=good_batch.old_policy_logprobs.clone(),
        rewards=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        advantages=good_batch.advantages.clone(),
        metadata={"responses": [["low", "lower"]]},
    )

    buffer.add(good_batch, step=1)
    buffer.add(bad_batch, step=2)
    buffer.save(1)
    buffer.save(2)

    filtered = buffer.filter(min_reward=0.5)

    assert filtered == [good_batch]
    assert buffer.size_bytes() > 0


def test_buffer_evicts_oldest_in_memory_batches_when_capped(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path), max_batches=2)

    buffer.add(make_batch(), step=1)
    buffer.add(make_batch(), step=2)
    buffer.add(make_batch(), step=3)

    with pytest.raises(FileNotFoundError):
        buffer.load(1)
    assert buffer.load(2)
    assert buffer.load(3)

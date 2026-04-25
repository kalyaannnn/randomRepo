from __future__ import annotations

from pathlib import Path

import torch

from agentrl.core.rollout import RolloutBatch
from agentrl.memory.buffer import TrajectoryBuffer
from agentrl.observability.debugger import AgentRLDebugger
from agentrl.observability.logger import MetricsLogger
from agentrl.observability.profiler import SystemsProfiler
from agentrl.observability.replay import ReplayBuffer, TrajectoryStore


def make_batch() -> RolloutBatch:
    return RolloutBatch(
        input_ids=torch.tensor([[[10, 11, 12], [20, 21, 22]]], dtype=torch.long),
        attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
        completion_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
        old_policy_logprobs=torch.tensor([[[0.0, -1.6, -0.2], [0.0, -0.3, -0.4]]], dtype=torch.float32),
        rewards=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        metadata={
            "prompts": ["Solve: 3x + 7 = 22"],
            "responses": [["x = 5", "x = 3"]],
        },
    )


def test_metrics_logger_writes_jsonl_and_returns_stdout_line(tmp_path: Path) -> None:
    logger = MetricsLogger(output_dir=str(tmp_path))

    rendered = logger.log(3, {"mean_reward": 0.5, "policy_loss": 1.25})
    jsonl = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")

    assert "step=3" in rendered
    assert "mean_reward=0.5000" in rendered
    assert '"step": 3' in jsonl
    assert '"mean_reward": 0.5' in jsonl


def test_replay_buffer_show_and_compare_render_saved_trajectories(tmp_path: Path) -> None:
    store = TrajectoryBuffer(output_dir=str(tmp_path))
    batch_a = make_batch()
    batch_b = make_batch()
    batch_b = RolloutBatch(
        input_ids=batch_b.input_ids,
        attention_mask=batch_b.attention_mask,
        completion_mask=batch_b.completion_mask,
        old_policy_logprobs=batch_b.old_policy_logprobs,
        rewards=batch_b.rewards,
        advantages=batch_b.advantages,
        metadata={"prompts": ["Solve: 3x + 7 = 22"], "responses": [["x = 5", "x = 4"]]},
    )
    store.add(batch_a, 10)
    store.add(batch_b, 20)
    store.save(10)
    store.save(20)

    replay = ReplayBuffer(output_dir=str(tmp_path))
    shown = replay.show(10)
    compared = replay.compare(10, 20)

    assert "Prompt:" in shown
    assert 'x = 5' in shown
    assert "Compare step 10 vs 20" in compared
    assert "x = 4" in compared


def test_trajectory_store_lists_saved_steps(tmp_path: Path) -> None:
    store = TrajectoryBuffer(output_dir=str(tmp_path))
    store.add(make_batch(), 1)
    store.add(make_batch(), 7)
    store.save(1)
    store.save(7)

    trajectory_store = TrajectoryStore(output_dir=str(tmp_path))

    assert trajectory_store.list_steps() == [1, 7]


def test_debugger_captures_low_reward_and_renders_token_diffs() -> None:
    debugger = AgentRLDebugger(reward_threshold=0.5)
    batch = make_batch()
    low_reward_batch = RolloutBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        completion_mask=batch.completion_mask,
        old_policy_logprobs=batch.old_policy_logprobs,
        rewards=torch.tensor([[0.1, 0.0]], dtype=torch.float32),
        advantages=batch.advantages,
        metadata=batch.metadata,
    )

    debugger.capture(step=4, batch=low_reward_batch, metrics={"total_loss": 1.0})
    debug_text = debugger.debug_episode(4)

    assert "Debug Step 4" in debug_text
    assert "reward=0.10" in debug_text
    assert "log_prob(old_policy)=" in debug_text
    assert "*" in debug_text


def test_profiler_reports_phase_table_and_metrics() -> None:
    profiler = SystemsProfiler()
    with profiler:
        with profiler.phase("generation"):
            _ = sum(range(1000))
        with profiler.phase("training"):
            _ = sum(range(2000))

    report = profiler.report()
    metrics = profiler.metrics()

    assert "Phase          Time (ms)" in report
    assert "generation_time_ms" in metrics
    assert "generation_peak_vram_mb" in metrics
    assert "generation_runtime_headroom_mb" in metrics
    assert "training_time_ms" in metrics
    assert metrics["total_step_time_ms"] >= 0.0
